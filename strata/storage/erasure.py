"""Erasure & identity — compliance-grade deletion (§4.1) and minimal identity ops (§4.6).

Two erasure modes ship in v1 (crypto-shredding is a documented Phase-2 option):

- ``scrub``   — delete the user's two path prefixes from the working tree and
                commit. Files are gone from HEAD; **git history still contains
                them** (stated honestly — this is retention-friendly audit mode).
- ``rewrite`` — additionally rewrite all history with git-filter-repo and
                aggressively gc: provable erasure. Invalidates clones/remotes.

Every erasure writes a tombstone to the ops ledger; the erasure commit itself
is the in-repo audit record.
"""
from __future__ import annotations

from typing import Literal

from ..models import Page
from ..obs import Ledgers, log
from ..util import utcnow_iso
from .git_backend import SubprocessGit
from .repo import Repo


def forget_user(repo: Repo, git: SubprocessGit, user_id: str,
                mode: Literal["scrub", "rewrite"] = "scrub") -> dict:
    """Erase all memory about ``user_id``. Returns a tombstone summary."""
    paths = repo.user_paths(user_id)
    n_pages = len(repo.list_pages(user_id=user_id))
    git.rm(paths)
    for p in paths:                                   # git rm misses untracked files
        target = repo.root / p
        if target.exists():
            import shutil
            shutil.rmtree(target)
    repo.regenerate_index()
    sha = git.commit_all(f"strata: forget user {user_id} (mode={mode}, {n_pages} pages)")
    if mode == "rewrite":
        git.filter_out_paths(paths)
        sha = "history-rewritten"
    tombstone = {
        "user_id": user_id, "mode": mode, "paths": paths,
        "pages_erased": n_pages, "erased_at": utcnow_iso(), "commit": sha,
    }
    Ledgers(repo.strata_dir).record_op("erasure", **tombstone)
    log.info("forgot user %s (mode=%s, %d pages)", user_id, mode, n_pages)
    return tombstone


def merge_users(repo: Repo, git: SubprocessGit, from_user: str, into_user: str) -> int:
    """Minimal identity resolution: move ``from_user``'s pages and raw sources
    under ``into_user``, rewrite frontmatter, record the alias. Returns pages moved."""
    from_prefix, into_prefix = repo.user_prefix(from_user), repo.user_prefix(into_user)
    if from_prefix == into_prefix:
        return 0
    moved = 0
    for page, body in repo.list_pages(user_id=from_user):
        # page.id is "u/<from>/<type>/<slug>" — re-root it under the target user
        new_id = f"{into_prefix}/{page.id.split('/', 2)[2]}"
        if repo.page_exists(new_id):                  # collision: keep both, disambiguate
            new_id = f"{new_id}-from-{from_prefix.split('/')[1]}"
        page.id, page.user_id = new_id, into_user
        if from_user not in page.aliases:
            page.aliases.append(from_user)
        repo.write_page(page, body)
        moved += 1
    # raw sources move wholesale; frontmatter user_id inside them stays as recorded
    # (raw is immutable history), the path move is what re-scopes erasure.
    src = repo.root / "raw" / from_prefix
    if src.exists():
        dst = repo.root / "raw" / into_prefix
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.rglob("*.md"):
            rel = child.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            n = 2
            while target.exists():
                target = target.with_name(f"{target.stem}-{n}{target.suffix}")
                n += 1
            child.replace(target)
        import shutil
        shutil.rmtree(src)
    old_wiki = repo.root / "wiki" / from_prefix
    if old_wiki.exists():
        import shutil
        shutil.rmtree(old_wiki)
    # note the alias on the target user's canonical profile page if one exists
    profile = repo.read_page(f"{into_prefix}/user/profile")
    if profile is not None:
        page, body = profile
        if from_user not in page.aliases:
            page.aliases.append(from_user)
            repo.write_page(page, body)
    repo.regenerate_index()
    git.commit_all(f"strata: merge user {from_user} into {into_user} ({moved} pages)")
    Ledgers(repo.strata_dir).record_op(
        "merge_users", from_user=from_user, into_user=into_user, pages_moved=moved,
    )
    return moved
