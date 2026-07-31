"""Ledger resolution — where PageOps meet existing pages (§6.2, §6.4).

For each proposed claim against the target page's active ledger:
- same subject + same (normalized) text  → corroboration: confidence up, source added
- same subject + different text          → contradiction: old claim's valid_until is
                                           closed, new claim records `supersedes`
- otherwise                              → append as a new claim

Trust ceilings by provenance (schema.md) are enforced here, so no upstream
component can smuggle in an over-confident imported/inferred claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Claim, ClaimDraft, Page, PageOp, PageStatus
from ..schema import Schema
from ..storage.repo import Repo
from ..util import new_claim_id, today_iso, utcnow_iso


def _norm_text(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


@dataclass
class ResolveResult:
    page_id: str
    created: bool
    claims_added: int = 0
    claims_corroborated: int = 0
    supersessions: list[tuple[str, str]] = field(default_factory=list)   # (old_id, new_id)


def canonical_page_id(op: PageOp, repo: Repo) -> str:
    if op.page_id:
        return op.page_id
    if op.type == "user" and op.user_id:
        # one deterministic profile page per user — all person-facts converge here
        return f"u/{Repo.user_prefix(op.user_id).split('/', 1)[1]}/user/profile"
    return Repo.make_page_id(op.type, op.title or "page", op.user_id)


def merge_claim(page: Page, draft: ClaimDraft, source: str, schema: Schema,
                result: ResolveResult) -> None:
    ceiling = schema.ceiling_for(draft.provenance)
    confidence = min(draft.confidence, ceiling)
    norm = _norm_text(draft.text)

    for existing in page.active_claims():
        if _norm_text(existing.text) == norm:
            existing.confidence = min(max(existing.confidence, confidence) + 0.05,
                                      schema.ceiling_for(existing.provenance))
            if source not in existing.sources:
                existing.sources.append(source)
            result.claims_corroborated += 1
            return

    new_claim = Claim(
        id=new_claim_id(draft.text), text=draft.text, subject=draft.subject,
        valid_from=draft.valid_from or today_iso(), valid_until=None,
        recorded_at=utcnow_iso(), confidence=confidence,
        provenance=draft.provenance, sources=[source],
    )
    if draft.subject:
        for existing in page.active_claims():
            if existing.subject == draft.subject:
                existing.valid_until = new_claim.valid_from
                new_claim.supersedes = existing.id
                result.supersessions.append((existing.id, new_claim.id))
                break
    page.claims.append(new_claim)
    result.claims_added += 1


def apply_op(repo: Repo, schema: Schema, op: PageOp, source_path: str,
             pii_types: list[str] | None = None, agent_id: str | None = None,
             run_id: str | None = None) -> ResolveResult:
    """Apply one PageOp to the wiki (filesystem only — commit/index are the
    pipeline's job, batched). Returns what happened for logging/commit message."""
    page_id = canonical_page_id(op, repo)
    existing = repo.read_page(page_id)
    now = utcnow_iso()

    if existing is None:
        page = Page(
            type=op.type, id=page_id, title=op.title or page_id.rsplit("/", 1)[-1],
            user_id=op.user_id, agent_id=agent_id, run_id=run_id,
            tags=sorted(set(op.tags)), created=now, updated=now,
            confidence=op.confidence, summary=op.summary, sources=[],
            pii=sorted(pii_types or []),
        )
        body = op.body.strip()
        result = ResolveResult(page_id=page_id, created=True)
    else:
        page, body = existing
        page.updated = now
        page.tags = sorted(set(page.tags) | set(op.tags))
        if op.summary:
            page.summary = op.summary
        if op.title and not page.title:
            page.title = op.title
        for t in pii_types or []:
            if t not in page.pii:
                page.pii.append(t)
        if op.body.strip():
            body = f"{body.rstrip()}\n\n## Details (updated {now})\n\n{op.body.strip()}"
        result = ResolveResult(page_id=page_id, created=False)

    if source_path and source_path not in page.sources:
        page.sources.append(source_path)

    for draft in op.claims:
        merge_claim(page, draft, source_path, schema, result)

    if page.claims:
        active = page.active_claims()
        if active:
            page.confidence = round(sum(c.confidence for c in active) / len(active), 3)
    elif not result.created:
        page.confidence = max(page.confidence, min(op.confidence, 1.0))

    if page.status == PageStatus.archived:            # fresh evidence revives archived pages
        page.status = PageStatus.active

    if not body.strip():
        body = page.summary or page.title
    repo.write_page(page, body)
    return result
