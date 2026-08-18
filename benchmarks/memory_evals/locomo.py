"""LoCoMo adapter (snap-research/locomo, `data/locomo10.json`).

Dataset shape (defensively parsed):
    [{"sample_id": ..., "conversation": {"speaker_a", "speaker_b",
       "session_1": [{"speaker", "text", "blip_caption"?}, ...],
       "session_1_date_time": "...", ...},
      "qa": [{"question", "answer" | "adversarial_answer", "category", ...}]}]

Each conversation becomes one MarkMem user; each session one raw add() with its
date stamped into the text. Category 5 (adversarial / unanswerable) is excluded
from presence aggregates — its gold is "not answerable", which a substring
check can't score.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .common import BenchmarkResult, evaluate_question, fresh_memory, presence

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}
ADVERSARIAL_CATEGORY = 5

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def download_locomo(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading LoCoMo -> {dest} …")
    urllib.request.urlretrieve(LOCOMO_URL, dest)
    return dest


def load_locomo(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"unexpected LoCoMo format in {path} (expected a list)")
    return data


def _session_text(turns: list[dict], date: str) -> str:
    lines = [f"(conversation session on {date})"] if date else []
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text") or turn.get("clean_text") or ""
        if turn.get("blip_caption"):
            text = f"{text} [shares a photo: {turn['blip_caption']}]".strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def ingest_conversation(memory, sample: dict, user_id: str) -> dict[str, str]:
    """Ingest all sessions; returns {session_key: raw_path} so QA evidence ids
    ("D3:2" -> session_3) can be traced to the pages compiled from them."""
    conv = sample.get("conversation", {})
    session_keys = sorted(
        (k for k in conv if _SESSION_KEY.match(k) and isinstance(conv[k], list)),
        key=lambda k: int(_SESSION_KEY.match(k).group(1)),
    )
    raw_by_session: dict[str, str] = {}
    for key in session_keys:
        text = _session_text(conv[key], conv.get(f"{key}_date_time", ""))
        if text.strip():
            result = memory.add(text, user_id=user_id, origin=key)
            raw_by_session[key] = result.get("raw_path", "")
    memory.flush()
    return raw_by_session


def pages_by_raw_source(memory, user_id: str) -> dict[str, list[str]]:
    """raw_path -> compiled pages that cite it. With the heuristic extractor
    this is 1:1 (session page); with LLM extraction, content may be routed
    onto other topical pages — any citing page carries the evidence. The user
    profile is excluded: it aggregates sources from everywhere and would
    inflate recall."""
    mapping: dict[str, list[str]] = {}
    for page, _ in memory.repo.list_pages(user_id=user_id):
        if page.type == "user":
            continue
        for src in page.sources:
            mapping.setdefault(src, []).append(page.id)
    return mapping


_EVIDENCE_ID = re.compile(r"D(\d+)")


def evidence_groups_for(qa: dict, raw_by_session: dict[str, str],
                        pages_by_raw: dict[str, list[str]]) -> list[list[str]]:
    """One group of acceptable pages per evidence id ("D3:2" -> session 3).
    Empty groups (evidence never compiled) stay in — they count as misses."""
    groups: list[list[str]] = []
    seen: set[str] = set()
    for ev in qa.get("evidence") or []:
        m = _EVIDENCE_ID.search(str(ev))
        if not m:
            continue
        key = f"session_{m.group(1)}"
        if key in seen or key not in raw_by_session:
            continue
        seen.add(key)
        groups.append(pages_by_raw.get(raw_by_session[key], []))
    return groups


def run_locomo(path: Path, work_dir: Path, k: int = 5,
               limit_conversations: Optional[int] = None,
               limit_qa: Optional[int] = None, llm=None,
               force_heuristic: bool = True,
               progress: Callable[[str], None] = print) -> BenchmarkResult:
    samples = load_locomo(path)
    if limit_conversations:
        samples = samples[:limit_conversations]
    result = BenchmarkResult(
        name="LoCoMo", mode="LLM QA (Nemotron)" if llm else "retrieval-only",
        notes=f"{len(samples)} conversation(s); category 5 (adversarial) excluded",
    )
    memory = fresh_memory(work_dir, "locomo", force_heuristic)
    ceiling_hits = ceiling_total = 0
    try:
        for si, sample in enumerate(samples):
            user_id = f"locomo-{sample.get('sample_id', si)}"
            raw_by_session = ingest_conversation(memory, sample, user_id)
            pages_by_raw = pages_by_raw_source(memory, user_id)
            conv = sample.get("conversation", {})
            full_text = "\n".join(
                _session_text(conv[key], conv.get(f"{key}_date_time", ""))
                for key in raw_by_session)
            qa_items = sample.get("qa", [])
            if limit_qa:
                qa_items = qa_items[:limit_qa]
            progress(f"  LoCoMo {user_id}: {len(raw_by_session)} sessions, {len(qa_items)} questions")
            for qi, qa in enumerate(qa_items):
                category = qa.get("category")
                gold = qa.get("answer") or qa.get("adversarial_answer") or ""
                if category == ADVERSARIAL_CATEGORY or not str(gold).strip():
                    result.skipped += 1
                    continue
                ceiling_total += 1
                ceiling_hits += presence(full_text, str(gold))
                result.cases.append(evaluate_question(
                    memory, qa.get("question", ""), str(gold), user_id,
                    qid=f"{user_id}/q{qi}",
                    category=f"{category}:{CATEGORY_NAMES.get(category, 'unknown')}",
                    k=k, llm=llm,
                    evidence_pages=evidence_groups_for(qa, raw_by_session, pages_by_raw),
                ))
    finally:
        memory.close()
    if ceiling_total:
        # a perfect retriever cannot exceed this on the presence proxy: many
        # golds are derived (dates, aggregations) and never appear verbatim
        ceiling = round(ceiling_hits / ceiling_total, 4)
        result.metrics["answer_presence_ceiling"] = ceiling
        result.notes += f"; answer-presence ceiling (gold verbatim anywhere): {ceiling}"
    return result
