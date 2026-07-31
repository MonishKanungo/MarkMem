"""Lint (§6.5.4) — make memory defects visible instead of silent.

Checks: broken [[wikilinks]], unsourced claims, injection-shaped content
(§4.5 poisoning defense), ledger/prose drift (superseded facts still stated
as current in the body), and near-duplicate titles (routing sprawl).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..storage.repo import Repo
from ..write.guard import find_injection

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


@dataclass
class LintFinding:
    page_id: str
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.page_id}: {self.detail}"


def _norm_title(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def _strip_history_section(body: str) -> str:
    out, in_history = [], False
    for line in body.splitlines():
        if line.startswith("## "):
            in_history = line.strip() == "## History"
        if not in_history:
            out.append(line)
    return "\n".join(out)


def lint_repo(repo: Repo) -> list[LintFinding]:
    findings: list[LintFinding] = []
    pages = list(repo.iter_pages())
    ids = {p.id for p, _ in pages}
    titles: dict[str, str] = {}

    for page, body in pages:
        # broken wikilinks
        for target in _WIKILINK.findall(body):
            target = target.strip()
            if target and target not in ids:
                findings.append(LintFinding(page.id, "broken-link", f"[[{target}]] does not exist"))

        # unsourced claims
        for c in page.claims:
            if not c.sources:
                findings.append(LintFinding(page.id, "unsourced-claim", f"{c.id}: {c.text[:60]!r}"))

        # injection-shaped content in what gets re-injected into prompts
        flags = find_injection("\n".join([body, page.summary,
                                          *(c.text for c in page.claims)]))
        for flag in flags:
            findings.append(LintFinding(page.id, "injection", flag))

        # ledger/prose drift: a superseded claim's text still present in body
        # prose *outside* the History section (where it legitimately lives)
        non_history = _strip_history_section(body)
        body_norm = re.sub(r"\W+", " ", non_history.lower())
        for c in page.claims:
            if not c.is_active:
                claim_norm = re.sub(r"\W+", " ", c.text.lower()).strip()
                if claim_norm and claim_norm in body_norm:
                    findings.append(LintFinding(
                        page.id, "ledger-drift",
                        f"superseded claim {c.id} still reads as current in body"))

        # near-duplicate titles within a type (routing sprawl, §10.7)
        if page.status.value == "active":
            key = f"{page.type}:{_norm_title(page.title)}"
            if key in titles and titles[key] != page.id:
                findings.append(LintFinding(
                    page.id, "duplicate-title",
                    f"same normalized title as {titles[key]} — consider merging"))
            else:
                titles[key] = page.id
    return findings
