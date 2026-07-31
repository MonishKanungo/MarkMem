"""Retention sweep (§6.5.3) — enforce per-type retain_days from schema.md.

Pages whose last update is older than their type's retain_days are hard
deleted from the working tree (the deletion commit is the audit record; git
history keeps the content unless an erasure mode rewrites it — both behaviors
documented in README). Pinned pages are never swept.
"""
from __future__ import annotations

from ..obs import Ledgers, log
from ..schema import Schema
from ..storage.repo import Repo
from ..util import parse_iso, utcnow


def retention_sweep(repo: Repo, schema: Schema) -> list[str]:
    """Delete over-retention pages. Returns deleted ids; caller commits + reindexes."""
    deleted: list[str] = []
    now = utcnow()
    for page, _ in repo.iter_pages():
        if page.pinned:
            continue
        retain = schema.retain_days_for(page.type)
        if retain is None:
            continue
        try:
            age_days = (now - parse_iso(page.updated)).total_seconds() / 86400.0
        except (ValueError, TypeError):
            continue
        if age_days > retain:
            repo.delete_page_file(page.id)
            deleted.append(page.id)
    if deleted:
        Ledgers(repo.strata_dir).record_op("retention", pages_deleted=deleted)
        log.info("retention sweep deleted %d page(s)", len(deleted))
    return deleted
