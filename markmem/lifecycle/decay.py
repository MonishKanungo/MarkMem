"""Decay sweep (§6.5.1) — idempotent confidence half-life enforcement.

Stored ``confidence`` stays the base value set at last content update; the
sweep computes the decay-adjusted effective value and archives pages that fall
below their decay class's threshold. Running the sweep twice changes nothing
(effective confidence is derived, never compounded back into the base).
Fresh evidence on an archived page revives it (resolve.apply_op).
"""
from __future__ import annotations

from ..models import PageStatus
from ..obs import log
from ..read.search import effective_confidence
from ..schema import Schema
from ..storage.repo import Repo


def decay_sweep(repo: Repo, schema: Schema) -> list[str]:
    """Archive pages whose effective confidence fell below threshold.
    Returns archived page ids; caller commits + reindexes."""
    archived: list[str] = []
    for page, body in repo.iter_pages():
        if page.status != PageStatus.active or page.pinned:
            continue
        rule = schema.decay_for(page.type)
        if rule.archive_below_confidence is None:
            continue
        eff = effective_confidence(page.confidence, page.updated, rule.half_life_days)
        if eff < rule.archive_below_confidence:
            page.status = PageStatus.archived
            page.metadata["archived_reason"] = (
                f"decay: effective confidence {eff:.3f} < {rule.archive_below_confidence}"
            )
            repo.write_page(page, body)
            archived.append(page.id)
    if archived:
        log.info("decay sweep archived %d page(s)", len(archived))
    return archived
