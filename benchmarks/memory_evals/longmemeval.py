"""LongMemEval adapter (xiaowu0162/LongMemEval).

Dataset shape (defensively parsed):
    [{"question_id", "question_type", "question", "answer", "question_date",
      "haystack_session_ids": [...], "haystack_dates": [...],
      "haystack_sessions": [[{"role", "content"}, ...], ...],
      "answer_session_ids": [...]}]

The dataset is NOT auto-downloaded (hosted via the project's release channels;
`longmemeval_oracle.json` — evidence-only sessions — is the practical variant
for this harness; `_s`/`_m` also work but ingest much more). Get it from
https://github.com/xiaowu0162/LongMemEval and pass --longmemeval <path>.

Each instance becomes its own Strata user (sessions ingested, then the one
question asked). Abstention instances (question_id ending in `_abs`, gold =
unanswerable) are excluded from presence aggregates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .common import BenchmarkResult, evaluate_question, fresh_memory

INSTRUCTIONS = (
    "LongMemEval dataset not found. Download it from "
    "https://github.com/xiaowu0162/LongMemEval (longmemeval_oracle.json is the "
    "compact evidence-only variant) and pass --longmemeval <path-to-json>."
)


def load_longmemeval(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"unexpected LongMemEval format in {path} (expected a list)")
    return data


def _session_text(turns: list[dict], date: str) -> str:
    lines = [f"(chat session on {date})"] if date else []
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ingest_instance(memory, instance: dict, user_id: str) -> dict[str, str]:
    """Ingest haystack sessions; returns {session_id: raw_path} so
    answer_session_ids can be traced to the pages compiled from them."""
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    ids = instance.get("haystack_session_ids", [])
    raw_by_session: dict[str, str] = {}
    for i, turns in enumerate(sessions):
        date = dates[i] if i < len(dates) else ""
        origin = str(ids[i]) if i < len(ids) else f"session-{i}"
        text = _session_text(turns, date)
        if text.strip():
            result = memory.add(text, user_id=user_id, origin=origin)
            raw_by_session[origin] = result.get("raw_path", "")
    memory.flush()
    return raw_by_session


def run_longmemeval(path: Path, work_dir: Path, k: int = 5,
                    limit: Optional[int] = None, llm=None,
                    force_heuristic: bool = True,
                    progress: Callable[[str], None] = print) -> BenchmarkResult:
    instances = load_longmemeval(path)
    if limit:
        instances = instances[:limit]
    result = BenchmarkResult(
        name="LongMemEval", mode="LLM QA (Nemotron)" if llm else "retrieval-only",
        notes=f"{len(instances)} instance(s); abstention (_abs) excluded",
    )
    memory = fresh_memory(work_dir, "longmemeval", force_heuristic)
    try:
        for i, inst in enumerate(instances):
            qid = str(inst.get("question_id", i))
            user_id = f"lme-{qid}"
            gold = str(inst.get("answer", "")).strip()
            if qid.endswith("_abs") or not gold:
                result.skipped += 1
                continue
            raw_by_session = ingest_instance(memory, inst, user_id)
            if i % 10 == 0:
                progress(f"  LongMemEval {i + 1}/{len(instances)} ({len(raw_by_session)} sessions)")
            from .locomo import pages_by_raw_source
            pages_by_raw = pages_by_raw_source(memory, user_id)
            evidence = [pages_by_raw.get(raw_by_session[str(sid)], [])
                        for sid in inst.get("answer_session_ids", [])
                        if str(sid) in raw_by_session]
            result.cases.append(evaluate_question(
                memory, inst.get("question", ""), gold, user_id, qid=qid,
                category=inst.get("question_type", "unknown"), k=k, llm=llm,
                evidence_pages=evidence,
            ))
    finally:
        memory.close()
    return result
