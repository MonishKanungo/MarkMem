"""Context packer (§5.6) — retrieval returns pages; prompts need budgeted context.

Properties, in order:
1. Stable-first ordering (user profile → pinned → hits by score) so the
   downstream LLM call gets prompt-prefix cache hits (§9.6).
2. Dedupe by page id.
3. Every block carries id + effective confidence + provenance + date so the
   LLM can cite memory instead of asserting it.
4. Truncation happens at block/claim boundaries, never mid-sentence.
"""
from __future__ import annotations

from typing import Optional

from ..models import Page, SearchHit
from ..util import est_tokens

HEADER = "### Memory (cite ids when you rely on these)"


def _page_block(page: Page, body: str) -> str:
    lines = [f"[{page.id} | {page.type} | conf {page.confidence:.2f} | updated {page.updated[:10]}]"]
    if page.summary:
        lines.append(page.summary.strip())
    active = page.active_claims()
    if active:
        for c in sorted(active, key=lambda c: -c.confidence):
            lines.append(f"- ({c.provenance.value}, {c.confidence:.2f}) {c.text}")
    elif body.strip() and not page.summary:
        lines.append(body.strip().splitlines()[0])
    return "\n".join(lines)


def _hit_block(hit: SearchHit) -> str:
    prov = f" | {hit.provenance}" if hit.provenance else ""
    lines = [f"[{hit.page_id} | {hit.type} | conf {hit.confidence:.2f}{prov} | updated {hit.updated[:10]}]"]
    if hit.summary:
        lines.append(hit.summary.strip())
    if hit.claims:
        for c in sorted(hit.claims, key=lambda c: -c.confidence):
            if c.is_active:  # A2: never pack superseded claims — they are hallucination surface
                lines.append(f"- ({c.provenance.value}, {c.confidence:.2f}) {c.text}")
    # A3: always include snippet evidence alongside claims (not just as fallback)
    if hit.snippet:
        lines.extend(f"> {ln}" for ln in hit.snippet.splitlines() if ln.strip())
    return "\n".join(lines)


def _fit(block: str, remaining: int) -> Optional[str]:
    """Fit a block into the remaining budget, trimming trailing lines at
    claim/line boundaries; None if even the header line doesn't fit."""
    if est_tokens(block) <= remaining:
        return block
    lines = block.splitlines()
    while len(lines) > 1:
        lines.pop()
        candidate = "\n".join(lines)
        if est_tokens(candidate) <= remaining:
            return candidate
    return lines[0] if lines and est_tokens(lines[0]) <= remaining else None


def pack(standing: list[tuple[Page, str]], hits: list[SearchHit],
         token_budget: int = 2000) -> str:
    """Return one packed string ready to paste into a system prompt."""
    remaining = token_budget - est_tokens(HEADER)
    blocks: list[str] = []
    seen: set[str] = set()

    for page, body in standing:                       # stable prefix first
        if page.id in seen:
            continue
        fitted = _fit(_page_block(page, body), remaining)
        if fitted is None:
            break
        blocks.append(fitted)
        seen.add(page.id)
        remaining -= est_tokens(fitted) + 1

    for hit in hits:
        if hit.page_id in seen or remaining <= 0:
            continue
        fitted = _fit(_hit_block(hit), remaining)
        if fitted is None:
            continue                                  # a smaller later block may still fit
        blocks.append(fitted)
        seen.add(hit.page_id)
        remaining -= est_tokens(fitted) + 1

    if not blocks:
        return ""
    return "\n\n".join([HEADER, *blocks])
