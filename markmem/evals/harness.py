"""Application-level eval harness (§4.4) — "the only memory layer that ships
its own eval harness."

Domain evals are synthesized from the repo's own supersession history: every
superseded claim chain yields the question "what is the current <subject>?"
with mechanically-known ground truth (the active claim) and a known trap (the
superseded text). Two metrics per case:

- hit@k    — does the right page surface in the top k?
- current  — does the packed context contain the current fact and NOT present
             the superseded one as current? (temporal-correctness, the LoCoMo
             failure mode §10.4)

Results are written to evals/ inside the memory repo so quality itself is
git-tracked over time (§6.5.5). Public-benchmark adapters (LoCoMo/LongMemEval/
BEAM) are Phase-2 work; the interface here is what they will plug into.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..util import utcnow_iso


@dataclass
class EvalCase:
    question: str
    expected_text: str            # active claim — ground truth
    superseded_text: str          # must NOT be presented as current
    page_id: str
    subject: str
    user_id: str | None = None


@dataclass
class EvalResult:
    cases: int = 0
    hit_at_k: float = 0.0
    current_fact_rate: float = 0.0
    stale_leak_rate: float = 0.0
    k: int = 5
    ran_at: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)


def generate_domain_eval(memory) -> list[EvalCase]:
    """Mine supersession chains from the live claim ledger."""
    cases: list[EvalCase] = []
    for page, _ in memory.repo.iter_pages():
        for claim in page.active_claims():
            if not claim.supersedes or not claim.subject:
                continue
            old = page.get_claim(claim.supersedes)
            if old is None:
                continue
            topic = claim.subject.split(":", 1)[-1].replace("-", " ")
            who = page.user_id or "the user"
            cases.append(EvalCase(
                question=f"What is {who}'s current {topic}?",
                expected_text=claim.text, superseded_text=old.text,
                page_id=page.id, subject=claim.subject, user_id=page.user_id,
            ))
    return cases


def _norm(text: str) -> str:
    import re
    return re.sub(r"\W+", " ", text.lower()).strip()


def _non_episodic(packed: str) -> str:
    """Drop dated episodic blocks (session pages, raw snippets) from the
    stale-leak scan: an old statement inside a dated what-happened record is
    history, not a fact asserted as current. Semantic blocks (user/concept/
    entity/...) must never carry superseded text unmarked."""
    keep = []
    for block in packed.split("\n\n"):
        header = block.splitlines()[0] if block else ""
        if "| session |" in header or "| raw |" in header:
            continue
        keep.append(block)
    return "\n\n".join(keep)


def run_domain_eval(memory, cases: list[EvalCase] | None = None,
                    k: int = 5, save: bool = True) -> EvalResult:
    cases = cases if cases is not None else generate_domain_eval(memory)
    result = EvalResult(cases=len(cases), k=k, ran_at=utcnow_iso())
    if not cases:
        return result
    hits = current_ok = stale_leaks = 0
    for case in cases:
        found = memory.searcher.search(case.question, user_id=case.user_id, top_k=k)
        page_hit = any(h.page_id == case.page_id for h in found)
        packed = memory.search(case.question, user_id=case.user_id, format="context")
        has_current = _norm(case.expected_text) in _norm(packed)
        # stale leak = superseded text asserted as current in a *semantic* block
        semantic = _non_episodic(packed)
        stale = (_norm(case.superseded_text) in _norm(semantic)
                 and "superseded" not in semantic.lower())
        hits += page_hit
        current_ok += has_current
        stale_leaks += stale
        if not (page_hit and has_current and not stale):
            result.failures.append({
                "question": case.question, "page_id": case.page_id,
                "page_hit": page_hit, "current_fact": has_current, "stale_leak": stale,
            })
    result.hit_at_k = round(hits / len(cases), 4)
    result.current_fact_rate = round(current_ok / len(cases), 4)
    result.stale_leak_rate = round(stale_leaks / len(cases), 4)
    if save:
        evals_dir = memory.repo.root / "evals"
        evals_dir.mkdir(exist_ok=True)
        stamp = result.ran_at.replace(":", "-").replace("+", "-")
        (evals_dir / f"domain-{stamp}.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        memory.git.commit_all(
            f"markmem: eval run — hit@{k} {result.hit_at_k}, "
            f"current {result.current_fact_rate}, stale-leak {result.stale_leak_rate}")
    return result
