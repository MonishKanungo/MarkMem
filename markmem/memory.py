"""Memory — the Mem0-shaped front door (§5.1 full parity).

    from markmem import Memory

    m = Memory(repo_path="./chat-memory")
    m.add("I'm vegetarian and prefer window seats", user_id="alice")
    m.search("alice food preference", user_id="alice")

Same integration shape as Mem0, different semantics where the architecture
demands honesty: `add()` enqueues (milliseconds) and compilation is eventual;
raw text is searchable immediately, compiled pages after the worker cycle or
an explicit `flush()`.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Optional

from .config import Config, load_config
from .models import Claim, Page, PageStatus, Provenance
from .obs import Ledgers, log
from .read.fts import Indexer
from .read.pack import pack
from .read.search import Searcher
from .schema import Schema, load_schema
from .storage.erasure import forget_user, merge_users
from .storage.git_backend import SubprocessGit, writer_lock
from .storage.repo import Repo
from .util import new_claim_id, today_iso, utcnow_iso
from .write.extractors.base import get_extractor
from .write.pii import apply_policy
from .write.pipeline import WritePipeline


def _messages_to_text(messages: str | list[dict]) -> str:
    if isinstance(messages, str):
        return messages
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):                 # anthropic-style content blocks
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class Memory:
    def __init__(self, repo_path: str | Path = "./markmem", auto_init: bool = True,
                 config: Optional[Config] = None, start_worker: bool = True,
                 force_heuristic: bool = False):
        self.repo = Repo(repo_path)
        if not self.repo.is_initialized:
            if not auto_init:
                raise FileNotFoundError(f"no markmem repo at {self.repo.root} (auto_init=False)")
            Repo.scaffold(self.repo.root)
        self.git = SubprocessGit(self.repo.root)
        self.git.init()
        self.config = config or load_config(self.repo.config_path)
        self.schema: Schema = load_schema(self.repo.schema_path)
        self.ledgers = Ledgers(self.repo.markmem_dir)
        self.indexer = Indexer(self.repo.markmem_dir)
        if not self.indexer.schema_current():          # one-time migration of the
            log.info("index schema outdated — rebuilding derived index")
            self.indexer.reindex(self.repo)            # derived cache (invariant §6.1)
        self.vector_index = self._make_vector_index()
        self.searcher = Searcher(self.repo, self.indexer, self.schema, self.config,
                                 self.vector_index)
        extractor = get_extractor(self.config, self.ledgers, force_heuristic=force_heuristic)
        self.pipeline = WritePipeline(
            self.repo, self.git, self.schema, self.config, extractor, self.ledgers,
            on_pages_written=self._on_pages_written,
            index_summary_provider=lambda: self.indexer.index_summary(),
        )
        if self.git.commit_all("markmem: init repo") is not None:
            log.info("initialized markmem repo at %s", self.repo.root)
        if start_worker:
            self.pipeline.start()

    def _make_vector_index(self):
        try:
            from .read.vectors import get_vector_index
            return get_vector_index(self.indexer.db_path)
        except Exception:
            return None

    def _on_pages_written(self, page_ids: list[str]) -> None:
        """Post-write hook: refresh the derived index for exactly the changed
        pages, then regenerate index.md from it (O(changed), not O(repo))."""
        self.indexer.index_pages(page_ids, self.repo)
        if self.vector_index is not None:
            items = []
            for pid in page_ids:
                parsed = self.repo.read_page(pid)
                if parsed:
                    page, body = parsed
                    items.append((pid, f"{page.title}\n{page.summary}\n{body}"[:2000]))
            try:
                self.vector_index.upsert(items)
            except Exception as e:
                log.warning("vector upsert failed: %s", e)
        self._write_index_md()

    def _write_index_md(self) -> None:
        from .storage.repo import format_index
        self.repo.index_path.write_text(format_index(self.indexer.index_entries()),
                                        encoding="utf-8")

    # ---------------- write ----------------

    def add(self, messages: str | list[dict], user_id: Optional[str] = None,
            agent_id: Optional[str] = None, run_id: Optional[str] = None,
            metadata: Optional[dict] = None, source_type: str = "conversation",
            origin: str = "") -> dict[str, Any]:
        """Append raw + enqueue for compilation. Returns in milliseconds; raises
        PIIBlockedError when config policy is `block` and PII was detected."""
        text = _messages_to_text(messages)
        if not text.strip():
            return {"status": "skipped", "reason": "empty input"}
        text, pii_types = apply_policy(text, self.config.pii.policy)
        entry = self.repo.append_raw(
            text, source_type=source_type, origin=origin, user_id=user_id,
            agent_id=agent_id, run_id=run_id, metadata=metadata, pii=pii_types,
        )
        queued_id = self.pipeline.enqueue(entry)
        self.indexer.index_raw(entry)                 # searchable before compile (§5.2)
        return {"status": "queued", "queued_id": queued_id,
                "raw_path": entry.path, "pii": pii_types}

    def flush(self, timeout_s: float = 300.0) -> int:
        """Force synchronous compilation of everything queued."""
        return self.pipeline.flush(timeout_s)

    # ---------------- read ----------------

    def search(self, query: str, user_id: Optional[str] = None,
               agent_id: Optional[str] = None, run_id: Optional[str] = None,
               top_k: Optional[int] = None, type: Optional[str] = None,
               as_of: Optional[str] = None, include_superseded: bool = False,
               metadata: Optional[dict] = None, format: Optional[str] = None):
        """Mem0-shaped search. format="context" returns one packed, budgeted
        string (standing context + hits) ready for a system prompt."""
        hits = self.searcher.search(
            query, user_id=user_id, type=type, top_k=top_k, as_of=as_of,
            include_superseded=include_superseded,
        )
        if agent_id or run_id or metadata:
            hits = [h for h in hits if self._row_matches(h.page_id, agent_id, run_id, metadata)]
        if format == "context":
            standing = self.searcher.standing_context(user_id)
            return pack(standing, hits, token_budget=self.config.search.token_budget)
        return [h.to_mem0() for h in hits]

    def _row_matches(self, page_id: str, agent_id, run_id, metadata) -> bool:
        row = self.indexer.page_row(page_id)
        if row is None:
            return False
        if agent_id and row["agent_id"] != agent_id:
            return False
        if run_id and row["run_id"] != run_id:
            return False
        if metadata:
            import json
            stored = json.loads(row["metadata"] or "{}")
            if any(stored.get(k) != v for k, v in metadata.items()):
                return False
        return True

    def get(self, memory_id: str) -> Optional[dict[str, Any]]:
        parsed = self.repo.read_page(memory_id)
        if parsed is None:
            return None
        return self._page_dict(*parsed)

    def get_all(self, user_id: Optional[str] = None, type: Optional[str] = None,
                agent_id: Optional[str] = None, run_id: Optional[str] = None,
                include_archived: bool = False) -> list[dict[str, Any]]:
        out = []
        for page, body in self.repo.list_pages(type=type, user_id=user_id):
            if not include_archived and page.status != PageStatus.active:
                continue
            if agent_id and page.agent_id != agent_id:
                continue
            if run_id and page.run_id != run_id:
                continue
            out.append(self._page_dict(page, body))
        return out

    @staticmethod
    def _page_dict(page: Page, body: str) -> dict[str, Any]:
        return {
            "id": page.id, "memory": page.summary or page.title, "user_id": page.user_id,
            "created_at": page.created, "updated_at": page.updated,
            "metadata": {"type": page.type, "status": page.status.value,
                         "tags": page.tags, "confidence": page.confidence,
                         "pinned": page.pinned, **page.metadata},
            "claims": [c.model_dump(mode="json", exclude_none=True) for c in page.claims],
            "body": body,
        }

    # ---------------- mutation (synchronous, user-initiated) ----------------

    def update(self, memory_id: str, data: str) -> dict[str, Any]:
        """Human correction: recorded as a human_edited claim (full trust) and
        as the page summary. History survives in the ledger + git."""
        parsed = self.repo.read_page(memory_id)
        if parsed is None:
            raise KeyError(f"no such memory: {memory_id}")
        page, body = parsed
        with writer_lock(self.repo.root):
            page.claims.append(Claim(
                id=new_claim_id(data), text=data, valid_from=today_iso(),
                recorded_at=utcnow_iso(), confidence=1.0,
                provenance=Provenance.human_edited, sources=["human:update"],
            ))
            page.summary = data
            page.updated = utcnow_iso()
            self.repo.write_page(page, body)
            self._on_pages_written([memory_id])
            self.git.commit_all(f"markmem: human update {memory_id}")
        return self._page_dict(page, body)

    def delete(self, memory_id: str, hard: bool = False) -> bool:
        parsed = self.repo.read_page(memory_id)
        if parsed is None:
            return False
        page, body = parsed
        with writer_lock(self.repo.root):
            if hard:
                self.git.rm([f"wiki/{memory_id}.md"])
                self.repo.delete_page_file(memory_id)
                self.indexer.remove_page(memory_id)
                if self.vector_index is not None:
                    self.vector_index.remove(memory_id)
                self._write_index_md()
            else:
                page.status = PageStatus.archived
                self.repo.write_page(page, body)
                self._on_pages_written([memory_id])
            self.git.commit_all(
                f"markmem: {'hard delete' if hard else 'archive'} {memory_id}")
        return True

    def delete_all(self, user_id: str, hard: bool = False) -> int:
        """Soft: archive every page of the user. Hard: provable path-prefix
        erasure via forget() (scrub mode)."""
        if hard:
            tombstone = self.forget(user_id, mode="scrub")
            return tombstone["pages_erased"]
        n = 0
        with writer_lock(self.repo.root):
            changed = []
            for page, body in self.repo.list_pages(user_id=user_id):
                if page.status != PageStatus.archived:
                    page.status = PageStatus.archived
                    self.repo.write_page(page, body)
                    changed.append(page.id)
                    n += 1
            if changed:
                self._on_pages_written(changed)
                self.git.commit_all(f"markmem: archive all pages of {user_id} ({n})")
        return n

    def forget(self, user_id: str, mode: str = "scrub") -> dict:
        """Compliance erasure (§4.1): scrub = delete from working tree + tombstone;
        rewrite = additionally purge from all git history (needs git-filter-repo)."""
        with writer_lock(self.repo.root):
            tombstone = forget_user(self.repo, self.git, user_id, mode=mode)  # type: ignore[arg-type]
            self.indexer.remove_user(user_id)
        return tombstone

    def merge_users(self, from_user: str, into_user: str) -> int:
        with writer_lock(self.repo.root):
            moved = merge_users(self.repo, self.git, from_user, into_user)
            self.indexer.reindex(self.repo)
        return moved

    def history(self, memory_id: str, include_diff: bool = False) -> list[dict[str, Any]]:
        """Better than Mem0's: it's literally `git log --follow` on the page."""
        commits = self.git.history(f"wiki/{memory_id}.md", include_diff=include_diff)
        return [{"commit": c.sha, "author": c.author, "date": c.date,
                 "message": c.message, **({"diff": c.diff} if include_diff else {})}
                for c in commits]

    def reset(self) -> None:
        """Wipe wiki/, raw/, queue, review and index; keep the repo + git history
        (the reset itself becomes an auditable commit). For provable per-user
        erasure use forget(); for a full purge delete the directory."""
        worker_was_running = self.pipeline._worker is not None and self.pipeline._worker.is_alive()
        self.pipeline.close()                          # release queue.db handles
        self.indexer.close()                           # release index.db handles
        with writer_lock(self.repo.root):
            for d in (self.repo.wiki_dir, self.repo.raw_dir):
                if d.exists():
                    shutil.rmtree(d)
            for name in ("queue.db", "index.db", "review"):
                target = self.repo.markmem_dir / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            Repo.scaffold(self.repo.root)
            self.git.commit_all("markmem: reset (wiki + raw wiped)")
        self.indexer = Indexer(self.repo.markmem_dir)
        self.searcher.indexer = self.indexer
        self.pipeline._init_db()
        self.ledgers.record_op("reset")
        if worker_was_running:
            self.pipeline.start()

    # ---------------- lifecycle & governance ----------------

    def maintenance(self, decay: bool = True, consolidate: bool = True,
                    retention: bool = True) -> dict[str, list[str]]:
        """Run the scheduled sweeps once (§6.5). Idempotent; one commit."""
        from .lifecycle import consolidation_sweep, decay_sweep, retention_sweep
        report: dict[str, list[str]] = {}
        with writer_lock(self.repo.root):
            if decay:
                report["archived"] = decay_sweep(self.repo, self.schema)
            if consolidate:
                report["consolidated"] = consolidation_sweep(
                    self.repo, self.config.pipeline.consolidate_after)
            if retention:
                report["deleted"] = retention_sweep(self.repo, self.schema)
            touched = sorted({pid for ids in report.values() for pid in ids})
            if touched:
                self.repo.regenerate_index()
                self.git.commit_all(
                    "markmem: maintenance ("
                    + ", ".join(f"{k}={len(v)}" for k, v in report.items() if v) + ")")
                self.indexer.reindex(self.repo)
        return report

    def lint(self):
        from .lifecycle import lint_repo
        return lint_repo(self.repo)

    def reindex(self) -> int:
        return self.indexer.reindex(self.repo)

    def stats(self) -> dict[str, Any]:
        return {
            "repo": str(self.repo.root),
            **self.indexer.counts(),
            "queue": self.pipeline.stats(),
            "review_pending": len(self.pipeline.review_queue.list()),
            "tokens": self.ledgers.token_totals(),
            "extractor": self.pipeline.extractor.name,
            "vector_search": self.vector_index is not None,
        }

    # ---------------- plumbing ----------------

    def close(self) -> None:
        self.pipeline.close()
        self.indexer.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncMemory:
    """Thin asyncio wrapper (§5.1) — every call runs in a worker thread."""

    def __init__(self, *args, **kwargs):
        self._m = Memory(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._m, name)
        if not callable(attr):
            return attr

        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)
        return wrapper

    async def __aenter__(self) -> "AsyncMemory":
        return self

    async def __aexit__(self, *exc) -> None:
        await asyncio.to_thread(self._m.close)
