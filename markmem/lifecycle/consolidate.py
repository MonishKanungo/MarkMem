"""Consolidation (§5.3) — append-only pages rot; rewrite them from the ledger.

The claim ledger is the machine-grade truth, so the body can be regenerated
from it deterministically: summary → current facts → history → one compacted
notes section from the accumulated append-updates. Git preserves every
pre-consolidation state, so this is always reversible. An LLM rewrite can slot
in later behind the same function; the deterministic version is the honest,
dependency-free default and is itself an accuracy feature (BM25 dilution ends).
"""
from __future__ import annotations

import re

from ..models import Page
from ..obs import log
from ..storage.repo import Repo

_UPDATE_SECTION = re.compile(r"^## Details \(updated .*\)$", re.MULTILINE)


def update_section_count(body: str) -> int:
    return len(_UPDATE_SECTION.findall(body))


def rewrite_body_from_ledger(page: Page, body: str) -> str:
    parts: list[str] = []
    if page.summary:
        parts.append(page.summary.strip())
    active = page.active_claims()
    if active:
        parts.append("## Current\n\n" + "\n".join(
            f"- {c.text} *(since {c.valid_from}, {c.provenance.value}, conf {c.confidence:.2f})*"
            for c in sorted(active, key=lambda c: c.valid_from or "")
        ))
    superseded = [c for c in page.claims if not c.is_active]
    if superseded:
        parts.append("## History\n\n" + "\n".join(
            f"- ~~{c.text}~~ *(held {c.valid_from} → {c.valid_until}"
            + (f", superseded by {nxt.id}" if (nxt := next(
                (n for n in page.claims if n.supersedes == c.id), None)) else "")
            + ")*"
            for c in sorted(superseded, key=lambda c: c.valid_until or "")
        ))
    # keep prose that never made it into claims, compacted into one section
    notes = _extract_notes(body, page)
    if notes:
        parts.append("## Notes\n\n" + notes)
    return "\n\n".join(parts).strip()


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _extract_notes(body: str, page: Page) -> str:
    """Prose worth keeping: everything except ledger-derived sections
    (## Current / ## History are regenerated from claims each time) and
    paragraphs that duplicate the summary or a claim. Idempotent: running
    consolidation twice yields the same Notes."""
    kept_chunks: list[str] = []
    drop_section = False
    current: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            kept_chunks.append("\n".join(current))
            current = []
            # Current/History are regenerated; Details/Notes headers are folded away
            drop_section = line.strip() in ("## Current", "## History")
            if not drop_section and not _UPDATE_SECTION.match(line.strip()) \
                    and line.strip() != "## Notes":
                current.append(line)                  # user's own heading: keep verbatim
            continue
        if not drop_section:
            current.append(line)
    kept_chunks.append("\n".join(current))

    ledger_texts = {_norm(c.text) for c in page.claims} | {_norm(page.summary or "")}
    seen: set[str] = set()
    out: list[str] = []
    for para in re.split(r"\n\s*\n", "\n\n".join(kept_chunks)):
        para = para.strip()
        key = _norm(para)
        if not para or not key or key in seen or key in ledger_texts:
            continue
        seen.add(key)
        out.append(para)
    return "\n\n".join(out)


def consolidate_page(repo: Repo, page: Page, body: str) -> bool:
    """Rewrite one page. Returns True if the body changed."""
    new_body = rewrite_body_from_ledger(page, body)
    if new_body.strip() == body.strip():
        return False
    repo.write_page(page, new_body)
    return True


def consolidation_sweep(repo: Repo, min_update_sections: int = 5) -> list[str]:
    """Consolidate pages with enough append-scar tissue. Returns page ids
    rewritten; caller commits + reindexes."""
    rewritten: list[str] = []
    for page, body in repo.iter_pages():
        if page.status.value != "active":
            continue
        if update_section_count(body) >= min_update_sections and consolidate_page(repo, page, body):
            rewritten.append(page.id)
    if rewritten:
        log.info("consolidated %d page(s)", len(rewritten))
    return rewritten
