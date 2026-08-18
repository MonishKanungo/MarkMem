"""MarkMemBench — MarkMem's own open, hand-authored memory benchmark.

WHY THIS EXISTS
───────────────
Public benchmarks (LoCoMo, LongMemEval) measure answer-presence with paraphrased
golds, so their scores conflate three separate things: retrieval quality,
extraction quality, and string-matching luck. They also don't label which
session actually contains the answer, so true R@k is not computable.

MarkMemBench fixes that. Every question declares:
  • the exact session(s) that contain the answer  → true R@1 / R@5 / MRR / P@5
  • the exact gold answer span                    → answer presence, EM, F1
  • the exact stale trap (for superseded facts)   → stale-leak rate
  • a category                                    → per-capability breakdown

Everything is hand-verifiable: read the scenario, read the question, confirm the
label. No dataset download. Fully deterministic offline.

WHAT IT MEASURES (10 capabilities)
──────────────────────────────────
  simple_recall      fact stated once, asked directly
  supersession       fact later changed — current value must win
  point_in_time      as_of=<date> must return what was true then
  multi_fact         answer needs 2+ sessions
  long_range         fact from the first session, asked after many more
  distractor         another user holds a similar-but-wrong fact
  isolation          same question, different users, different answers
  abstention         never stated — must NOT be answerable
  temporal_order     requires knowing which fact came first
  pii_handling       fact containing PII is stored and retrievable

Run:
    python -m benchmarks.memory_evals.markmembench            # offline
    python -m benchmarks.memory_evals.markmembench --with-llm # + EM/F1
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .common import BenchmarkResult, fresh_memory, normalize, presence


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO — three users, chronological sessions.
#  Each session is one add() call with origin=<session id>.
#  Facts are deliberately plain first-person so the offline heuristic extractor
#  can compile them; the LLM extractor handles the same text strictly better.
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS: list[dict] = [
    # ---- alice: changes job AND seat preference (supersession chains) ----
    {"id": "a1", "user": "alice", "date": "2026-01-05",
     "text": "I am Alice. I work at NVIDIA. I am a software engineer."},
    {"id": "a2", "user": "alice", "date": "2026-01-12",
     "text": "I prefer window seats. I am vegetarian."},
    {"id": "a3", "user": "alice", "date": "2026-02-20",
     "text": "I live in San Francisco. My cat is named Luna."},
    {"id": "a4", "user": "alice", "date": "2026-03-10",
     "text": "Actually I prefer aisle seats."},
    {"id": "a5", "user": "alice", "date": "2026-04-01",
     "text": "I work at Google now."},
    {"id": "a6", "user": "alice", "date": "2026-05-15",
     "text": "My email is alice.chen@example.com. I am learning Spanish."},

    # ---- bob: holds facts that are DISTRACTORS for alice's questions ----
    {"id": "b1", "user": "bob", "date": "2026-01-08",
     "text": "I am Bob. I work at Google. I am a data scientist."},
    {"id": "b2", "user": "bob", "date": "2026-02-14",
     "text": "I prefer aisle seats. I love steak."},
    {"id": "b3", "user": "bob", "date": "2026-03-22",
     "text": "My dog is named Max."},

    # ---- carol: minimal footprint, used for abstention ----
    {"id": "c1", "user": "carol", "date": "2026-01-20",
     "text": "I am Carol. I live in Lisbon."},
]


# ─────────────────────────────────────────────────────────────────────────────
#  QUESTIONS — every one declares its gold evidence session(s) so true R@k is
#  computable, plus the gold answer span and (for superseded facts) the trap.
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS: list[dict] = [
    # ---------- simple_recall ----------
    {"id": "q01", "user": "alice", "category": "simple_recall",
     "q": "What is Alice's job title?", "gold": "software engineer",
     "evidence": ["a1"]},
    {"id": "q02", "user": "alice", "category": "simple_recall",
     "q": "What is the name of Alice's cat?", "gold": "Luna",
     "evidence": ["a3"]},
    {"id": "q03", "user": "bob", "category": "simple_recall",
     "q": "What is the name of Bob's dog?", "gold": "Max",
     "evidence": ["b3"]},
    {"id": "q04", "user": "carol", "category": "simple_recall",
     "q": "Which city does Carol live in?", "gold": "Lisbon",
     "evidence": ["c1"]},

    # ---------- supersession (current value must win, old must not) ----------
    {"id": "q05", "user": "alice", "category": "supersession",
     "q": "What seat does Alice prefer?", "gold": "aisle",
     "evidence": ["a4"], "trap": "window"},
    {"id": "q06", "user": "alice", "category": "supersession",
     "q": "Which company does Alice work at?", "gold": "Google",
     "evidence": ["a5"], "trap": "NVIDIA"},

    # ---------- point_in_time (as_of) ----------
    {"id": "q07", "user": "alice", "category": "point_in_time",
     "q": "What seat does Alice prefer?", "gold": "window",
     "evidence": ["a2"], "as_of": "2026-02-01"},
    {"id": "q08", "user": "alice", "category": "point_in_time",
     "q": "Which company does Alice work at?", "gold": "NVIDIA",
     "evidence": ["a1"], "as_of": "2026-03-01"},

    # ---------- multi_fact (needs 2+ sessions) ----------
    {"id": "q09", "user": "alice", "category": "multi_fact",
     "q": "What are Alice's dietary and seating preferences?",
     "gold": "vegetarian", "evidence": ["a2", "a4"]},

    # ---------- long_range (first session, asked after 5 more) ----------
    {"id": "q10", "user": "alice", "category": "long_range",
     "q": "What is this user's name?", "gold": "Alice",
     "evidence": ["a1"]},

    # ---------- distractor (bob also prefers aisle / also at Google) ----------
    {"id": "q11", "user": "bob", "category": "distractor",
     "q": "What food does Bob love?", "gold": "steak",
     "evidence": ["b2"], "trap": "vegetarian"},
    {"id": "q12", "user": "bob", "category": "distractor",
     "q": "What is Bob's job title?", "gold": "data scientist",
     "evidence": ["b1"], "trap": "software engineer"},

    # ---------- isolation (same question, different correct answers) ----------
    {"id": "q13", "user": "alice", "category": "isolation",
     "q": "What is this user's diet?", "gold": "vegetarian",
     "evidence": ["a2"], "trap": "steak"},
    {"id": "q14", "user": "bob", "category": "isolation",
     "q": "What is this user's diet?", "gold": "steak",
     "evidence": ["b2"], "trap": "vegetarian"},

    # ---------- abstention (never stated — must not be answerable) ----------
    {"id": "q15", "user": "carol", "category": "abstention",
     "q": "What is Carol's job title?", "gold": "unknown", "evidence": []},
    {"id": "q16", "user": "alice", "category": "abstention",
     "q": "What is Alice's phone number?", "gold": "unknown", "evidence": []},

    # ---------- temporal_order ----------
    {"id": "q17", "user": "alice", "category": "temporal_order",
     "q": "Where did Alice work before Google?", "gold": "NVIDIA",
     "evidence": ["a1", "a5"]},

    # ---------- pii_handling ----------
    {"id": "q18", "user": "alice", "category": "pii_handling",
     "q": "What is Alice's email address?", "gold": "alice.chen@example.com",
     "evidence": ["a6"]},
]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-case result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    qid: str
    category: str
    # true retrieval metrics (None when the question has no gold evidence)
    r_at_1: Optional[bool] = None
    r_at_5: Optional[bool] = None
    r_at_5_all: Optional[bool] = None
    mrr: Optional[float] = None
    p_at_5: Optional[float] = None
    # answer-level metrics
    answer_in_context: bool = False
    # trap leaked into a SEMANTIC block (asserted as current) — the real defect
    trap_in_context: Optional[bool] = None
    # trap present anywhere, including episodic history blocks (informational:
    # retaining what was said is correct behaviour, not a leak)
    trap_anywhere: Optional[bool] = None
    abstained: Optional[bool] = None          # LLM said "unknown" (abstention only)
    em: Optional[bool] = None
    f1: Optional[float] = None
    prediction: str = ""
    context_tokens: int = 0
    latency_ms: float = 0.0
    retrieved: list[str] = field(default_factory=list)


def _rate(cases: list[BenchCase], attr: str) -> Optional[float]:
    """Mean over cases where the metric APPLIES. Returns None when it applies to
    none of them (e.g. R@k on abstention questions, which have no gold evidence)
    so the report can render '—' instead of a misleading 0.000."""
    vals = [getattr(c, attr) for c in cases if getattr(c, attr) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def _semantic_blocks(packed: str) -> str:
    """Drop dated episodic blocks (session pages, raw snippets) — same semantics
    as markmem/evals/harness.py. An old statement inside a dated what-happened
    record is HISTORY, not a fact asserted as current. Semantic blocks
    (user/concept/entity) must never carry a superseded fact unmarked."""
    keep = []
    for block in packed.split("\n\n"):
        header = block.splitlines()[0] if block else ""
        if "| session |" in header or "| raw |" in header:
            continue
        keep.append(block)
    return "\n\n".join(keep)


def _build_evidence_map(memory) -> dict[str, set[str]]:
    """session_id -> {page_ids whose sources include that session's raw entry}.

    A session's content may be routed onto several pages (a session page plus
    the user's profile), so evidence is a GROUP: retrieving any member counts.
    """
    ev: dict[str, set[str]] = {}
    for page, _ in memory.repo.iter_pages():
        for src in page.sources:
            for sess in SESSIONS:
                sid = sess["id"]
                if f"__{sid}." in src or src.endswith(f"__{sid}.md"):
                    ev.setdefault(sid, set()).add(page.id)
    return ev


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def ingest_scenario(memory, progress: Callable[[str], None] = print) -> None:
    """Ingest every session in chronological order, one add() each."""
    for sess in SESSIONS:
        memory.add(sess["text"], user_id=sess["user"], origin=sess["id"])
        memory.flush()      # compile per session so later turns see earlier facts
    progress(f"  ingested {len(SESSIONS)} sessions "
             f"for {len({s['user'] for s in SESSIONS})} users")


def _score_one(memory, q: dict, evidence_map: dict[str, set[str]],
               k: int, llm=None) -> BenchCase:
    from markmem.util import est_tokens
    from .common import exact_match, token_f1

    case = BenchCase(qid=q["id"], category=q["category"])

    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"user_id": q["user"], "top_k": k}
    if q.get("as_of"):
        kwargs["as_of"] = q["as_of"]
    context = memory.search(q["q"], format="context", **kwargs) or ""
    hits = memory.searcher.search(q["q"], **kwargs)
    case.latency_ms = (time.perf_counter() - t0) * 1000
    case.context_tokens = est_tokens(context)
    case.retrieved = [h.page_id for h in hits]

    # ---- true retrieval metrics (only when gold evidence is declared) ----
    groups = [evidence_map.get(sid, set()) for sid in q["evidence"]]
    if groups:
        ranked = case.retrieved
        topk = set(ranked[:k])
        relevant = set().union(*groups) if groups else set()

        case.r_at_1 = bool(ranked) and ranked[0] in relevant
        case.r_at_5 = any(g & topk for g in groups)
        case.r_at_5_all = all(g & topk for g in groups)
        case.p_at_5 = round(len(topk & relevant) / max(len(topk), 1), 4)
        case.mrr = 0.0
        for rank, pid in enumerate(ranked, start=1):
            if pid in relevant:
                case.mrr = round(1.0 / rank, 4)
                break

    # ---- answer-level metrics ----
    case.answer_in_context = presence(context, q["gold"]) \
        if q["gold"] != "unknown" else True
    if q.get("trap"):
        semantic = _semantic_blocks(context)
        # a leak only counts if the stale fact is asserted as current in a
        # semantic block AND is not explicitly marked superseded
        case.trap_in_context = (presence(semantic, q["trap"])
                                and "superseded" not in semantic.lower())
        case.trap_anywhere = presence(context, q["trap"])

    if llm is not None:
        from .common import answer_with_llm
        pred = answer_with_llm(llm, context, q["q"])
        case.prediction = pred
        if q["gold"] == "unknown":
            case.abstained = "unknown" in normalize(pred)
            case.em = case.abstained
            case.f1 = 1.0 if case.abstained else 0.0
        else:
            case.em = exact_match(pred, q["gold"])
            case.f1 = token_f1(pred, q["gold"])
    return case


def run_markmembench(work_dir: Path, k: int = 5, llm=None,
                    force_heuristic: bool = True,
                    progress: Callable[[str], None] = print) -> dict:
    """Run the full MarkMemBench suite. Returns a metrics dict."""
    memory = fresh_memory(work_dir, "markmembench", force_heuristic)
    try:
        ingest_scenario(memory, progress)
        evidence_map = _build_evidence_map(memory)
        progress(f"  evidence map: {len(evidence_map)}/{len(SESSIONS)} sessions "
                 f"traced to pages")

        cases = [_score_one(memory, q, evidence_map, k, llm) for q in QUESTIONS]
        extractor = memory.pipeline.extractor.name
        vector = memory.stats()["vector_search"]
    finally:
        memory.close()

    # ---- aggregate ----
    retrievable = [c for c in cases if c.r_at_5 is not None]
    trapped = [c for c in cases if c.trap_in_context is not None]
    abst = [c for c in cases if c.abstained is not None]

    by_cat: dict[str, dict[str, Any]] = {}
    for c in cases:
        b = by_cat.setdefault(c.category, {"n": 0, "cases": []})
        b["n"] += 1
        b["cases"].append(c)
    for cat, b in by_cat.items():
        cc = b.pop("cases")
        b["r_at_5"] = _rate(cc, "r_at_5")
        b["mrr"] = _rate(cc, "mrr")
        b["answer_in_context"] = _rate(cc, "answer_in_context")
        if any(x.f1 is not None for x in cc):
            b["em"] = _rate(cc, "em")
            b["f1"] = _rate(cc, "f1")

    out: dict[str, Any] = {
        "questions": len(cases),
        "retrievable_questions": len(retrievable),
        "extractor": extractor,
        "vector_search": vector,
        # true retrieval metrics — the headline numbers
        "r_at_1": _rate(cases, "r_at_1"),
        "r_at_5": _rate(cases, "r_at_5"),
        "r_at_5_all_evidence": _rate(cases, "r_at_5_all"),
        "mrr": _rate(cases, "mrr"),
        "precision_at_5": _rate(cases, "p_at_5"),
        # answer-level
        "answer_in_context": _rate(cases, "answer_in_context"),
        # the real defect: stale fact asserted as current in a semantic block
        "stale_leak_rate": _rate(trapped, "trap_in_context"),
        # informational: stale text present anywhere incl. episodic history
        "stale_present_incl_history": _rate(trapped, "trap_anywhere"),
        "context_tokens_p50": sorted(c.context_tokens for c in cases)[len(cases) // 2],
        "latency_ms_p50": round(sorted(c.latency_ms for c in cases)[len(cases) // 2], 2),
        "by_category": by_cat,
    }
    if abst:
        out["abstention_rate"] = _rate(abst, "abstained")
    if any(c.f1 is not None for c in cases):
        out["em"] = _rate(cases, "em")
        out["f1"] = _rate(cases, "f1")
    out["failures"] = [
        {"qid": c.qid, "category": c.category, "r_at_5": c.r_at_5,
         "answer_in_context": c.answer_in_context, "trap_leaked": c.trap_in_context,
         "prediction": c.prediction[:80]}
        for c in cases
        if (c.r_at_5 is False) or (not c.answer_in_context) or c.trap_in_context
        or (c.abstained is False)
    ]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(r: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    c = Console()

    c.rule("[bold green]MarkMemBench — hand-authored, ground-truth benchmark[/bold green]")
    c.print(f"[dim]extractor={r['extractor']}  vector={r['vector_search']}  "
            f"questions={r['questions']} ({r['retrievable_questions']} with gold evidence)[/dim]\n")

    t = Table(title="Retrieval (true R@k — gold evidence pages are labelled)",
              header_style="bold cyan")
    t.add_column("metric"); t.add_column("MarkMem", justify="right")
    t.add_column("reference", justify="right")
    t.add_row("R@1",                    _fmt(r["r_at_1"]),  "—")
    t.add_row("R@5",                    _fmt(r["r_at_5"]),  "Mem0 0.952 / Khoj 0.832")
    t.add_row("R@5 (all evidence)",     _fmt(r["r_at_5_all_evidence"]), "—")
    t.add_row("MRR",                    _fmt(r["mrr"]),     "—")
    t.add_row("Precision@5",            _fmt(r["precision_at_5"]), "—")
    c.print(t)

    t2 = Table(title="Answer quality & safety", header_style="bold cyan")
    t2.add_column("metric"); t2.add_column("MarkMem", justify="right")
    t2.add_column("ideal", justify="right")
    t2.add_row("answer-in-context",  _fmt(r["answer_in_context"]), "1.000")
    t2.add_row("stale-leak (asserted as current)", _fmt(r["stale_leak_rate"]), "0.000")
    t2.add_row("  + stale text present incl. history",
               _fmt(r["stale_present_incl_history"]), "(informational)")
    if r.get("abstention_rate") is not None:
        t2.add_row("abstention rate", _fmt(r["abstention_rate"]), "1.000")
    if r.get("f1") is not None:
        t2.add_row("EM (LLM graded)", _fmt(r["em"]), "1.000")
        t2.add_row("F1 (LLM graded)", _fmt(r["f1"]), "1.000")
    t2.add_row("ctx tokens p50",     str(r["context_tokens_p50"]), "—")
    t2.add_row("latency p50 (ms)",   f"{r['latency_ms_p50']}", "—")
    c.print(t2)
    c.print("[dim]  stale-leak counts a superseded fact only when it is asserted as "
            "current in a semantic block. Retaining it inside a dated session record "
            "is history, not a leak.[/dim]")

    t3 = Table(title="Per-capability breakdown", header_style="bold cyan")
    for col in ("capability", "n", "R@5", "MRR", "answer-in-ctx", "F1"):
        t3.add_column(col, justify="right" if col != "capability" else "left")
    for cat, b in sorted(r["by_category"].items()):
        t3.add_row(cat, str(b["n"]), _fmt(b["r_at_5"]), _fmt(b["mrr"]),
                   _fmt(b["answer_in_context"]), _fmt(b.get("f1")))
    c.print(t3)
    c.print("[dim]  '—' = metric does not apply (abstention questions have no gold "
            "evidence page, so R@k is undefined for them).[/dim]")

    if r["failures"]:
        c.print(f"\n[yellow]{len(r['failures'])} failing case(s):[/yellow]")
        for f in r["failures"]:
            c.print(f"  [red]{f['qid']}[/red] ({f['category']}) "
                    f"R@5={f['r_at_5']} in_ctx={f['answer_in_context']} "
                    f"trap={f['trap_leaked']}"
                    + (f" pred={f['prediction']!r}" if f["prediction"] else ""))
    else:
        c.print("\n[green]All cases passed.[/green]")


def main(argv: list[str] | None = None) -> dict:
    import argparse, json, tempfile
    ap = argparse.ArgumentParser(description="MarkMemBench — MarkMem's own benchmark")
    ap.add_argument("--k", type=int, default=5, help="search top-k")
    ap.add_argument("--with-llm", action="store_true", help="grade EM/F1 with an LLM")
    ap.add_argument("--llm-extract", action="store_true",
                    help="compile memory with the configured LLM extractor")
    ap.add_argument("--out", type=Path, help="write results JSON here")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(".env")

    llm = None
    if args.with_llm:
        import os
        from chatbot.llm import NemotronClient
        llm = NemotronClient(model=os.environ.get("NEMOTRON_JUDGE_MODEL") or None)
        if not llm.available:
            print("no LLM key available — running retrieval-only")
            llm = None

    work = Path(tempfile.mkdtemp(prefix="markmembench-"))
    result = run_markmembench(work, k=args.k, llm=llm,
                             force_heuristic=not args.llm_extract)
    print_report(result)

    out = args.out or (Path(__file__).resolve().parents[2] / "benchmarks" /
                       "results" / f"markmembench-{time.strftime('%Y-%m-%dT%H-%M-%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults saved -> {out}")
    return result


if __name__ == "__main__":
    main()
