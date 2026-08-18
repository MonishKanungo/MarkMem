"""BEAM (Benchmarking Episodic and Associative Memory) adapter.

BEAM tests structured memory retrieval: given a conversation, can the system
retrieve specific episodic facts at different temporal distances?

Dataset format (BEAM-style JSON):
    {
      "conversations": [
        {
          "id": "conv_001",
          "turns": [
            {"role": "user"|"assistant", "content": "...", "timestamp": "..."},
            ...
          ],
          "questions": [
            {
              "id": "q001",
              "question": "...",
              "answer": "...",
              "category": "episodic"|"associative"|"temporal"|"multi-hop",
              "turn_distance": 5   # how many turns back the answer was stated
            },
            ...
          ]
        }
      ]
    }

Categories:
- episodic:     Direct recall of stated facts
- associative:  Requires connecting two pieces of information
- temporal:     Requires knowing when something was stated or changed
- multi-hop:    Requires chaining multiple facts

Usage:
    python -m benchmarks.memory_evals.run --beam data/beam.json
    python -m benchmarks.memory_evals.run --beam data/beam.json --with-llm
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .common import BenchmarkResult, evaluate_question, fresh_memory

BEAM_CATEGORIES = {"episodic", "associative", "temporal", "multi-hop"}

# Synthetic mini-fixture for --self-test (no download needed)
BEAM_FIXTURE = {
    "conversations": [
        {
            "id": "beam_001",
            "turns": [
                {"role": "user", "content": "Hi, I'm Sarah. I work as a nurse in Boston."},
                {"role": "assistant", "content": "Nice to meet you, Sarah!"},
                {"role": "user", "content": "I have two cats named Pixel and Luna."},
                {"role": "assistant", "content": "Lovely names!"},
                {"role": "user", "content": "Actually I moved to Seattle last month for a new job."},
                {"role": "assistant", "content": "How exciting, a fresh start!"},
                {"role": "user", "content": "I now work as a software engineer instead of a nurse."},
            ],
            "questions": [
                {
                    "id": "q001",
                    "question": "What city does Sarah live in?",
                    "answer": "Seattle",
                    "category": "temporal",
                    "turn_distance": 2,
                },
                {
                    "id": "q002",
                    "question": "What are Sarah's cats named?",
                    "answer": "Pixel and Luna",
                    "category": "episodic",
                    "turn_distance": 4,
                },
                {
                    "id": "q003",
                    "question": "What is Sarah's current job?",
                    "answer": "software engineer",
                    "category": "temporal",
                    "turn_distance": 0,
                },
            ],
        }
    ]
}


def load_beam(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_beam(path: Optional[Path], work_dir: Path, k: int = 5,
             limit: Optional[int] = None, llm=None,
             force_heuristic: bool = True,
             progress: Callable[[str], None] = print) -> BenchmarkResult:
    """Run BEAM evaluation against MarkMem.

    Args:
        path: Path to BEAM JSON file. If None, uses the built-in fixture.
        work_dir: Temporary directory for MarkMem repos.
        k: Search top-k.
        limit: Max conversations to evaluate.
        llm: Optional NemotronClient for graded QA.
        force_heuristic: Use heuristic extractor (reproducible).
        progress: Callable for progress messages.

    Returns:
        BenchmarkResult with per-category breakdown.
    """
    if path is None:
        data = BEAM_FIXTURE
        source = "built-in fixture"
    else:
        data = load_beam(path)
        source = str(path)

    conversations = data.get("conversations", [])
    if limit:
        conversations = conversations[:limit]

    result = BenchmarkResult(
        name="BEAM",
        mode="LLM QA (Nemotron)" if llm else "retrieval-only",
        notes=f"{len(conversations)} conversation(s) from {source}",
    )

    memory = fresh_memory(work_dir, "beam", force_heuristic)
    try:
        for ci, conv in enumerate(conversations):
            user_id = f"beam-{conv.get('id', ci)}"

            # Ingest all turns as one raw add per conversation
            turns = conv.get("turns", [])
            if not turns:
                continue
            text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in turns
            )
            if text.strip():
                memory.add(text, user_id=user_id, origin=f"beam-conv-{ci}")
                memory.flush()

            questions = conv.get("questions", [])
            progress(f"  BEAM {user_id}: {len(turns)} turns, {len(questions)} questions")

            for q in questions:
                gold = str(q.get("answer", "")).strip()
                question = q.get("question", "")
                if not gold or not question:
                    result.skipped += 1
                    continue

                category = q.get("category", "unknown")
                turn_dist = q.get("turn_distance", 0)
                # Annotate category with turn distance for fine-grained analysis
                cat_label = f"{category}@d{turn_dist}"

                result.cases.append(evaluate_question(
                    memory, question, gold, user_id,
                    qid=f"{user_id}/{q.get('id', 'q')}",
                    category=cat_label,
                    k=k, llm=llm,
                ))
    finally:
        memory.close()

    # Compute BEAM-specific metrics: per-category and per-turn-distance breakdown
    result.metrics = _beam_metrics(result)
    return result


def _beam_metrics(result: BenchmarkResult) -> dict:
    """Compute BEAM-native metrics beyond the common BenchmarkResult aggregates."""
    by_base_cat: dict[str, list] = {}
    by_dist: dict[str, list] = {}
    for c in result.cases:
        # Split "episodic@d3" -> base="episodic", dist=3
        parts = c.category.rsplit("@d", 1)
        base = parts[0] if parts else c.category
        dist = int(parts[1]) if len(parts) == 2 else 0

        by_base_cat.setdefault(base, []).append(c.in_context)
        by_dist.setdefault(f"d{dist}", []).append(c.in_context)

    metrics: dict = {}
    for cat, hits in sorted(by_base_cat.items()):
        metrics[f"in_context_{cat}"] = round(sum(hits) / len(hits), 4) if hits else 0.0
    for dist, hits in sorted(by_dist.items()):
        metrics[f"in_context_{dist}"] = round(sum(hits) / len(hits), 4) if hits else 0.0
    return metrics
