"""In-house evaluator — Strata's own supersession-based domain eval (§4.4).

Runs `strata.evals.run_domain_eval` either against a repo you point it at
(--inhouse-repo, evaluating YOUR memory) or against a deterministic built-in
scenario: 3 users × several contradiction chains, ingested fresh, so the run
is reproducible offline and exercises the exact temporal-correctness failure
mode the claim ledger exists to prevent.

Metrics differ from the QA benchmarks by design:
    hit@k             — right page in top-k
    current-fact rate — packed context contains the CURRENT claim
    stale-leak rate   — superseded claim asserted as current (lower = better)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from strata.evals import run_domain_eval

from .common import BenchmarkResult, CaseResult, fresh_memory

# (user, ordered statements) — later statements contradict earlier same-subject
# ones. Phrasing is deliberately crisp: the offline heuristic extractor keys
# contradictions on the object's head noun, so trailing adverb/preposition
# phrases ("...lately", "...in the morning") would split subjects. With the LLM
# extractor ([llm] extra) arbitrary phrasing resolves; this scenario tests the
# supersession/eval machinery at the dependency-free floor.
SCENARIO: list[tuple[str, list[str]]] = [
    ("alice", ["I prefer window seats.",
               "I live in Lisbon.",
               "Actually I prefer aisle seats.",
               "I live in Porto now."]),
    ("bob", ["I work at Initech.",
             "My favorite color is blue.",
             "I work at Acme Corp now.",
             "My favorite color is green."]),
    ("carol", ["I really like strong coffee.",
               "I hate coffee now."]),
]


def build_scenario(work_dir: Path):
    memory = fresh_memory(work_dir, "inhouse")
    for user, statements in SCENARIO:
        for s in statements:
            memory.add(s, user_id=user, origin="scenario")
        memory.flush()
    return memory


def run_inhouse(work_dir: Path, k: int = 5, repo: Optional[Path] = None,
                progress: Callable[[str], None] = print) -> BenchmarkResult:
    if repo is not None:
        from strata import Memory
        memory = Memory(repo_path=repo, start_worker=False)
        source = f"your repo: {repo}"
    else:
        memory = build_scenario(work_dir)
        source = "built-in deterministic scenario"
    try:
        eval_result = run_domain_eval(memory, k=k, save=False)
        progress(f"  in-house: {eval_result.cases} supersession case(s) from {source}")
        result = BenchmarkResult(
            name="In-house (claim ledger)",
            mode=f"supersession domain eval, {source}",
            notes=(f"hit@{k} {eval_result.hit_at_k} · current-fact "
                   f"{eval_result.current_fact_rate} · stale-leak {eval_result.stale_leak_rate}"),
        )
        # map onto the shared CaseResult shape so the combined table lines up:
        # in_context <- current fact present · in_pages <- page hit
        failures = {f["question"]: f for f in eval_result.failures}
        # run_domain_eval aggregates; reconstruct per-case booleans from failures
        from strata.evals import generate_domain_eval
        for case in generate_domain_eval(memory):
            f = failures.get(case.question)
            result.cases.append(CaseResult(
                qid=case.page_id + "/" + case.subject, category=case.subject.split(":")[0],
                in_context=(f is None or f.get("current_fact", False)),
                in_pages=(f is None or f.get("page_hit", False)),
                latency_ms=0.0, context_tokens=0,
            ))
        result.metrics = {
            "hit_at_k": eval_result.hit_at_k,
            "current_fact_rate": eval_result.current_fact_rate,
            "stale_leak_rate": eval_result.stale_leak_rate,
        }
        return result
    finally:
        memory.close()
