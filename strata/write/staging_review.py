"""Git-staging-branch review v2 (§6.6) — memory PRs as actual git branches.

The v1 review queue uses JSON files in .strata/review/. This v2 upgrade makes
each quarantined write a real git branch, so reviewers can:
- Use normal git tooling (git diff, GitHub PR review, etc.) to inspect changes
- CI/CD pipelines can run checks before accepting
- Every decision becomes a real git merge or branch deletion

Architecture:
    Quarantined op → write to staging branch → branch lives in .strata/staging/
    accept → git merge staging branch into HEAD → delete branch
    reject → delete staging branch (no merge)

Branch naming: strata/review/<item-id>
    e.g. strata/review/r-a1b2c3d4e5

This replaces ReviewQueue.add() / pop() when enabled via config:
    pipeline:
      review: gated
      review_backend: git  # new config key; default 'json' keeps v1 behavior

The JSON v1 queue and git v2 queue are independent; you can mix them
(some items in JSON, some in branches) during migration.
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Optional

from ..models import PageOp
from ..obs import Ledgers, log
from ..storage.git_backend import GitError, SubprocessGit, writer_lock
from ..storage.repo import Repo
from ..util import short_hash, utcnow_iso
from ..write.guard import find_injection
from ..write.resolve import apply_op
from ..schema import Schema


BRANCH_PREFIX = "strata/review/"


class StagingReviewQueue:
    """Git-branch-backed review queue.

    Each quarantined PageOp becomes:
    1. A git branch named strata/review/<item-id>
    2. The proposed page written to that branch
    3. A metadata file .strata/staging/<item-id>.json with the op + reasons

    Reviewers can inspect with: git diff main strata/review/<item-id>
    """

    def __init__(self, repo: Repo, git: SubprocessGit, ledgers: Ledgers,
                 schema: Schema):
        self.repo = repo
        self.git = git
        self.ledgers = ledgers
        self.schema = schema
        self.staging_dir = repo.strata_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Queue item lifecycle
    # ------------------------------------------------------------------

    def add(self, op: PageOp, raw_path: str, reasons: list[str]) -> str:
        """Write the proposed op to a staging branch and return the item_id."""
        item_id = f"r-{short_hash(raw_path + op.title + utcnow_iso(), 10)}"
        branch = BRANCH_PREFIX + item_id

        with self._lock:
            try:
                self._create_staging_branch(branch, op, raw_path)
            except GitError as e:
                log.warning("could not create staging branch %s: %s — falling back to JSON", branch, e)
                # Write metadata anyway so the item shows in list()
                pass

        # Always write metadata JSON (readable without git)
        payload = {
            "id": item_id,
            "queued_at": utcnow_iso(),
            "raw_path": raw_path,
            "reasons": reasons,
            "branch": branch,
            "op": op.model_dump(mode="json", exclude_none=True),
        }
        (self.staging_dir / f"{item_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("staged review item %s on branch %s", item_id, branch)
        return item_id

    def _create_staging_branch(self, branch: str, op: PageOp, raw_path: str) -> None:
        """Create a git branch with the proposed page change."""
        root = self.repo.root
        current = self._current_branch()

        # Create and checkout the staging branch
        self.git._run("checkout", "-b", branch)
        try:
            # Apply the op to the staging branch's working tree
            apply_op(self.repo, self.schema, op, raw_path)
            self.git._run("add", "-A")
            # Only commit if there's something staged
            status = self.git._run("status", "--porcelain").stdout.strip()
            if status:
                self.git._run("commit", "-q", "-m",
                              f"strata: review proposal — {op.type}/{op.title or 'page'}")
        finally:
            # Always return to the original branch
            self.git._run("checkout", current)

    def _current_branch(self) -> str:
        try:
            return self.git._run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        except GitError:
            return "main"

    def _branch_exists(self, branch: str) -> bool:
        result = self.git._run("branch", "--list", branch, check=False)
        return bool(result.stdout.strip())

    # ------------------------------------------------------------------
    # Queue inspection
    # ------------------------------------------------------------------

    def list(self) -> list[dict]:
        """Return all pending review items (from metadata JSON files)."""
        items = []
        if not self.staging_dir.exists():
            return items
        for path in sorted(self.staging_dir.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                # Annotate with whether the branch still exists
                item["branch_exists"] = self._branch_exists(item.get("branch", ""))
                items.append(item)
            except (json.JSONDecodeError, OSError):
                continue
        return items

    def get(self, item_id: str) -> Optional[dict]:
        path = self.staging_dir / f"{item_id}.json"
        if not path.exists():
            return None
        item = json.loads(path.read_text(encoding="utf-8"))
        item["branch_exists"] = self._branch_exists(item.get("branch", ""))
        return item

    def diff(self, item_id: str) -> str:
        """Return git diff between HEAD and the staging branch for human review."""
        item = self.get(item_id)
        if item is None:
            return f"no such item: {item_id}"
        branch = item.get("branch", "")
        if not self._branch_exists(branch):
            return f"branch {branch} no longer exists"
        try:
            result = self.git._run("diff", "HEAD", branch, check=False)
            return result.stdout or "(no diff)"
        except GitError as e:
            return f"diff failed: {e}"

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def accept(self, item_id: str) -> Optional[str]:
        """Merge the staging branch into HEAD. Returns the page_id written."""
        item = self.get(item_id)
        if item is None:
            return None

        branch = item.get("branch", "")
        page_id = None

        with self._lock:
            if self._branch_exists(branch):
                try:
                    # Merge the staging branch — this is the actual accept commit
                    self.git._run("merge", "--no-ff", branch,
                                  "-m", f"strata: review-accept {item_id}")
                    self.git._run("branch", "-d", branch)
                    # Extract page_id from the op
                    op = PageOp.model_validate(item["op"])
                    from ..write.resolve import canonical_page_id
                    page_id = canonical_page_id(op, self.repo)
                except GitError as e:
                    log.error("merge failed for %s: %s", item_id, e)
                    # Fall back: apply the op directly without the branch merge
                    op = PageOp.model_validate(item["op"])
                    result = apply_op(self.repo, self.schema, op, item.get("raw_path", ""))
                    self.git._run("commit", "-q", "-am",
                                  f"strata: review-accept {item_id} (fallback)")
                    page_id = result.page_id
            else:
                # Branch was deleted externally; apply the op directly
                op = PageOp.model_validate(item["op"])
                result = apply_op(self.repo, self.schema, op, item.get("raw_path", ""))
                self.git.commit_all(f"strata: review-accept {item_id} -> {result.page_id}")
                page_id = result.page_id

        self._cleanup(item_id, "accept", item)
        return page_id

    def reject(self, item_id: str) -> bool:
        """Delete the staging branch and discard the op."""
        item = self.get(item_id)
        if item is None:
            return False

        branch = item.get("branch", "")
        with self._lock:
            if self._branch_exists(branch):
                try:
                    self.git._run("branch", "-D", branch)
                except GitError as e:
                    log.warning("could not delete branch %s: %s", branch, e)

        self._cleanup(item_id, "reject", item)
        return True

    def _cleanup(self, item_id: str, decision: str, item: dict) -> None:
        """Remove metadata file and write to ops ledger."""
        meta_path = self.staging_dir / f"{item_id}.json"
        if meta_path.exists():
            meta_path.unlink()
        self.ledgers.record_op(
            "review_decision",
            item_id=item_id,
            decision=decision,
            raw_path=item.get("raw_path", ""),
            reasons=item.get("reasons", []),
            backend="git-branch",
        )
