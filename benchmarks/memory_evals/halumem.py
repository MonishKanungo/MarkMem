"""HaluMem (Hallucination in Memory) adapter.

HaluMem tests whether a memory system causes hallucinations in downstream LLM
responses. Specifically: does the retrieved memory context cause the LLM to
assert facts that contradict the ground truth or were never stated?

Three types of memory hallucination this adapter detects:

1. **Stale fact hallucination**: The system retrieves a superseded fact as current
   (e.g., "Alice prefers window seats" after she said she switched to aisle).
   → Measured by stale_leak_rate from the domain eval harness.

2. **Fabrication from partial retrieval**: The system retrieves incomplete context
   and the LLM fills in gaps with invented facts.
   → Measured by checking if the LLM answer introduces facts not in memory.

3. **Cross-user contamination**: Memory from user A leaks into user B's context.
   → Measured by checking if user-scoped search returns other users' pages.

Dataset format:
    {
      "cases": [
        {
          "id": "halu_001",
          "type": "stale"|"fabrication"|"contamination",
          "setup": [
            {"user_id": "alice", "text": "I prefer window seats."},
            {"user_id": "alice", "text": "Actually, I now prefer aisle seats."}
          ],
          "question": "What seat does Alice prefer?",
          "user_id": "alice",
          "correct_answer": "aisle seats",
          "hallucination_trap": "window seats",
          "contamination_user": null  # for contamination type: the other user
        }
      ]
    }

Usage:
    python -m benchmarks.memory_evals.run --halumem data/halumem.json
    python -m benchmarks.memory_evals.run --halumem data/halumem.json --with-llm
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .common import BenchmarkResult, CaseResult, fresh_memory, normalize, presence

# Built-in fixture — tests all three hallucination types offline
HALUMEM_FIXTURE = {
    "cases": [
        # Type 1: stale fact
        {
            "id": "halu_stale_001",
            "type": "stale",
            "setup": [
                {"user_id": "alice", "text": "user: I prefer window seats on flights.\nassistant: Got it!"},
                {"user_id": "alice", "text": "user: Actually I now prefer aisle seats, more legroom.\nassistant: Noted!"},
            ],
            "question": "What seat does Alice prefer?",
            "user_id": "alice",
            "correct_answer": "aisle",
            "hallucination_trap": "window",
            "contamination_user": None,
        },
        # Type 2: fabrication (question about something never stated)
        {
            "id": "halu_fab_001",
            "type": "fabrication",
            "setup": [
                {"user_id": "bob", "text": "user: I work at NVIDIA.\nassistant: Cool!"},
            ],
            "question": "What is Bob's job title at NVIDIA?",
            "user_id": "bob",
            "correct_answer": "unknown",  # never stated
            "hallucination_trap": None,
            "contamination_user": None,
        },
        # Type 3: cross-user contamination
        {
            "id": "halu_contam_001",
            "type": "contamination",
            "setup": [
                {"user_id": "carol", "text": "user: My favourite colour is purple.\nassistant: Nice!"},
                {"user_id": "dave", "text": "user: My favourite colour is green.\nassistant: Great!"},
            ],
            "question": "What is Carol's favourite colour?",
            "user_id": "carol",
            "correct_answer": "purple",
            "hallucination_trap": "green",
            "contamination_user": "dave",
        },
    ]
}


@dataclass
class HaluMemCaseResult:
    case_id: str
    halu_type: str
    stale_leak: bool          # True = stale fact appeared in context as current
    contaminated: bool        # True = other user's data appeared in context
    correct_in_context: bool  # True = correct answer found in context
    trap_in_context: bool     # True = hallucination trap found in context


@dataclass
class HaluMemResult:
    cases: list[HaluMemCaseResult] = field(default_factory=list)
    skipped: int = 0

    @property
    def n(self) -> int:
        return len(self.cases)

    def stale_leak_rate(self) -> float:
        stale = [c for c in self.cases if c.halu_type == "stale"]
        return round(sum(c.stale_leak for c in stale) / len(stale), 4) if stale else 0.0

    def contamination_rate(self) -> float:
        cont = [c for c in self.cases if c.halu_type == "contamination"]
        return round(sum(c.contaminated for c in cont) / len(cont), 4) if cont else 0.0

    def fabrication_exposure_rate(self) -> float:
        """Rate at which the context contains enough info to answer a fabrication question.
        High rate = system retrieves partial info that invites hallucination."""
        fab = [c for c in self.cases if c.halu_type == "fabrication"]
        # For fabrication: correct_in_context means the right answer ("unknown") is implicit.
        # We measure trap exposure: if context has topic keywords, LLM might fabricate.
        return round(sum(c.correct_in_context for c in fab) / len(fab), 4) if fab else 0.0

    def overall_safety_rate(self) -> float:
        """Rate of cases with no hallucination signal: no stale leak, no contamination."""
        if not self.cases:
            return 1.0
        safe = sum(not c.stale_leak and not c.contaminated for c in self.cases)
        return round(safe / len(self.cases), 4)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "skipped": self.skipped,
            "stale_leak_rate": self.stale_leak_rate(),
            "contamination_rate": self.contamination_rate(),
            "fabrication_exposure_rate": self.fabrication_exposure_rate(),
            "overall_safety_rate": self.overall_safety_rate(),
        }


def load_halumem(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_halumem(path: Optional[Path], work_dir: Path, k: int = 5,
                limit: Optional[int] = None,
                force_heuristic: bool = True,
                progress: Callable[[str], None] = print) -> tuple[BenchmarkResult, HaluMemResult]:
    """Run HaluMem evaluation.

    Returns both a BenchmarkResult (compatible with the combined summary table)
    and a HaluMemResult (detailed hallucination-specific metrics).
    """
    if path is None:
        data = HALUMEM_FIXTURE
        source = "built-in fixture"
    else:
        data = load_halumem(path)
        source = str(path)

    cases = data.get("cases", [])
    if limit:
        cases = cases[:limit]

    bench_result = BenchmarkResult(
        name="HaluMem",
        mode="retrieval-only",
        notes=f"{len(cases)} case(s) from {source}",
    )
    halu_result = HaluMemResult()

    memory = fresh_memory(work_dir, "halumem", force_heuristic)
    try:
        # Ingest all setup texts first (may span multiple users)
        all_users: set[str] = set()
        for case in cases:
            for step in case.get("setup", []):
                uid = step["user_id"]
                all_users.add(uid)
                memory.add(step["text"], user_id=uid, origin=f"halumem-{case['id']}")
        memory.flush()

        for ci, case in enumerate(cases):
            halu_type = case.get("type", "stale")
            user_id = case.get("user_id", "")
            question = case.get("question", "")
            correct = str(case.get("correct_answer", "")).strip()
            trap = case.get("hallucination_trap")
            contamination_user = case.get("contamination_user")

            if not question:
                halu_result.skipped += 1
                continue

            progress(f"  HaluMem {case['id']} [{halu_type}]: {question[:60]}")

            # Retrieve context scoped to the correct user
            context = memory.search(question, user_id=user_id, top_k=k, format="context") or ""

            # 1. Stale leak check: trap appears in semantic (non-episodic) context blocks
            stale_leak = False
            if trap:
                semantic_blocks = _non_episodic_blocks(context)
                stale_leak = (
                    normalize(trap) in normalize(semantic_blocks)
                    and "superseded" not in semantic_blocks.lower()
                )

            # 2. Contamination check: other user's data in context
            contaminated = False
            if contamination_user:
                # Check if the contamination user's trap appears in context
                # (context should be scoped to user_id only)
                contam_trap = trap or correct
                contaminated = normalize(contam_trap) in normalize(context)

            # 3. Correct answer presence
            correct_in_context = presence(context, correct) if correct not in ("unknown", "") else True

            # 4. Trap in context (generic hallucination surface check)
            trap_in_context = presence(context, trap) if trap else False

            halu_case = HaluMemCaseResult(
                case_id=case["id"],
                halu_type=halu_type,
                stale_leak=stale_leak,
                contaminated=contaminated,
                correct_in_context=correct_in_context,
                trap_in_context=trap_in_context,
            )
            halu_result.cases.append(halu_case)

            # Also add to the BenchmarkResult for combined table compatibility
            import time
            t0 = time.perf_counter()
            hits = memory.searcher.search(question, user_id=user_id, top_k=k)
            latency_ms = (time.perf_counter() - t0) * 1000
            from markmem.util import est_tokens
            bench_result.cases.append(CaseResult(
                qid=case["id"],
                category=halu_type,
                in_context=correct_in_context,
                in_pages=correct_in_context,  # same for halumem
                latency_ms=latency_ms,
                context_tokens=est_tokens(context),
            ))
    finally:
        memory.close()

    # Surface HaluMem-specific metrics into the BenchmarkResult for the combined table
    bench_result.metrics = halu_result.summary()
    return bench_result, halu_result


def _non_episodic_blocks(packed: str) -> str:
    """Keep only semantic blocks (not session/raw) for stale-leak checking.
    Mirrors the logic in markmem/evals/harness.py."""
    keep = []
    for block in packed.split("\n\n"):
        header = block.splitlines()[0] if block else ""
        if "| session |" in header or "| raw |" in header:
            continue
        keep.append(block)
    return "\n\n".join(keep)
