"""Benchmark runner — LoCoMo + LongMemEval + Strata's in-house evaluator,
results in labelled tables + a combined summary, saved as JSON.

    python -m benchmarks.memory_evals.run --self-test          # offline, fixtures
    python -m benchmarks.memory_evals.run --download-locomo    # fetch + run LoCoMo
    python -m benchmarks.memory_evals.run --locomo data/locomo10.json --limit-qa 30
    python -m benchmarks.memory_evals.run --longmemeval longmemeval_oracle.json --limit 50
    python -m benchmarks.memory_evals.run --with-llm           # Nemotron answers + EM/F1
    python -m benchmarks.memory_evals.run --inhouse-repo ./chat-memory

Honesty note (printed with results): retrieval-only numbers are answer-presence
recall proxies computed with Strata's default offline stack — they are NOT
comparable to published LoCoMo/LongMemEval leaderboard scores, which use full
LLM QA pipelines. Use --with-llm for graded QA (still your stack, your numbers).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .common import BenchmarkResult
from .inhouse import run_inhouse
from .locomo import LOCOMO_URL, download_locomo, run_locomo
from .longmemeval import INSTRUCTIONS as LME_INSTRUCTIONS
from .longmemeval import run_longmemeval
from .beam import run_beam, BEAM_FIXTURE
from .halumem import run_halumem

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RESULTS_DIR = ROOT / "benchmarks" / "results"

console = Console()


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_benchmark(result: BenchmarkResult) -> None:
    console.rule(f"[bold]{result.name}[/bold] — {result.mode}")
    summary = result.summary()
    native = "hit_at_k" in result.metrics  # in-house: show its own metrics only
    keys = (("cases", "skipped", "hit_at_k", "current_fact_rate", "stale_leak_rate")
            if native else
            ("cases", "skipped", "r_at_k", "r_at_k_all", "answer_in_context",
             "answer_in_pages", "answer_presence_ceiling",
             "em", "f1", "context_tokens_p50", "latency_ms_p50"))
    table = Table(title=f"{result.name}: overall", show_header=True)
    table.add_column("metric"); table.add_column("value", justify="right")
    for key in keys:
        if key in summary and summary[key] != "":
            table.add_row(key.replace("_", " "), _fmt(summary[key]))
    console.print(table)
    if result.notes:
        console.print(f"[dim]{result.notes}[/dim]")

    by_cat = result.by_category()
    if len(by_cat) > 1:
        cat_table = Table(title=f"{result.name}: by category")
        cat_table.add_column("category")
        cat_table.add_column("n", justify="right")
        has_r = any("r_at_k" in v for v in by_cat.values())
        if has_r:
            cat_table.add_column("R@k", justify="right")
        cat_table.add_column("current fact" if native else "answer in context", justify="right")
        cat_table.add_column("page hit" if native else "answer in pages", justify="right")
        if any("f1" in v for v in by_cat.values()):
            cat_table.add_column("f1", justify="right")
        for cat, vals in by_cat.items():
            row = [cat, str(vals["n"])]
            if has_r:
                row.append(_fmt(vals.get("r_at_k", "")))
            row += [_fmt(vals["in_context"]), _fmt(vals["in_pages"])]
            if any("f1" in v for v in by_cat.values()):
                row.append(_fmt(vals.get("f1", "")))
            cat_table.add_row(*row)
        console.print(cat_table)


def print_combined(results: list[BenchmarkResult]) -> None:
    console.rule("[bold]Combined summary[/bold]")
    table = Table(show_header=True)
    for col in ("benchmark", "mode", "cases", "primary metric", "secondary metric",
                "ctx tokens p50", "latency p50 (ms)"):
        table.add_column(col, justify="right" if "p50" in col or col == "cases" else "left")
    for r in results:
        s = r.summary()
        if "hit_at_k" in r.metrics:                     # in-house native metrics
            primary = f"current-fact {_fmt(r.metrics['current_fact_rate'])}"
            secondary = (f"hit@k {_fmt(r.metrics['hit_at_k'])} · "
                         f"stale-leak {_fmt(r.metrics['stale_leak_rate'])}")
            tokens = latency = "—"
        else:
            if "r_at_k" in s:
                primary = f"R@k (evidence) {_fmt(s['r_at_k'])}"
                secondary = f"answer-in-context {_fmt(s['answer_in_context'])}"
            else:
                primary = f"answer-in-context {_fmt(s['answer_in_context'])}"
                secondary = f"answer-in-pages {_fmt(s['answer_in_pages'])}"
            if "f1" in s:
                secondary += f" · EM {_fmt(s['em'])} · F1 {_fmt(s['f1'])}"
            tokens = _fmt(s["context_tokens_p50"])
            latency = _fmt(s["latency_ms_p50"])
        table.add_row(r.name, r.mode, str(s["cases"]), primary, secondary, tokens, latency)
    console.print(table)
    console.print(
        "[dim]Retrieval-only numbers are answer-presence recall proxies on Strata's "
        "offline stack — not comparable to published leaderboard scores (§14 of the "
        "spec: never quote incumbents' numbers without reproduction).[/dim]"
    )


def save_results(results: list[BenchmarkResult], out: Path | None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    path = out or (RESULTS_DIR / f"memory-evals-{stamp}.json")
    payload = [{**r.summary(), "by_category": r.by_category()} for r in results]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> list[BenchmarkResult]:
    ap = argparse.ArgumentParser(
        description="Run Strata against LoCoMo, LongMemEval and the in-house evaluator.")
    ap.add_argument("--locomo", type=Path, help="path to locomo10.json")
    ap.add_argument("--download-locomo", action="store_true",
                    help=f"download LoCoMo to benchmarks/datasets/ from {LOCOMO_URL}")
    ap.add_argument("--longmemeval", type=Path, help="path to longmemeval_*.json")
    ap.add_argument("--beam", type=Path, default=None,
                    help="path to BEAM JSON file (omit to use built-in fixture)")
    ap.add_argument("--halumem", type=Path, default=None,
                    help="path to HaluMem JSON file (omit to use built-in fixture)")
    ap.add_argument("--skip-beam", action="store_true", help="skip BEAM evaluation")
    ap.add_argument("--skip-halumem", action="store_true", help="skip HaluMem evaluation")
    ap.add_argument("--inhouse-repo", type=Path,
                    help="evaluate an existing Strata repo instead of the built-in scenario")
    ap.add_argument("--limit", type=int, default=None,
                    help="max conversations (LoCoMo) / instances (LongMemEval)")
    ap.add_argument("--limit-qa", type=int, default=None, help="max questions per LoCoMo conversation")
    ap.add_argument("--k", type=int, default=5, help="search top-k")
    ap.add_argument("--with-llm", action="store_true",
                    help="answer with Nemotron + grade EM/F1 (needs NVIDIA_API_KEY in .env)")
    ap.add_argument("--llm-extract", action="store_true",
                    help="compile memory with the LLM extractor (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--self-test", action="store_true",
                    help="run all three benchmarks on tiny bundled format fixtures (offline)")
    ap.add_argument("--out", type=Path, help="results JSON path")
    args = ap.parse_args(argv)

    llm = None
    if args.with_llm:
        import os
        from chatbot.llm import NemotronClient
        # judge model can be overridden separately from the chat/extraction
        # model via NEMOTRON_JUDGE_MODEL; unset -> NemotronClient's own default
        llm = NemotronClient(model=os.environ.get("NEMOTRON_JUDGE_MODEL") or None)
        if not llm.available:
            ap.error("--with-llm needs NVIDIA_API_KEY set in .env")

    locomo_path = args.locomo
    lme_path = args.longmemeval
    if args.self_test:
        locomo_path = locomo_path or FIXTURES / "locomo_sample.json"
        lme_path = lme_path or FIXTURES / "longmemeval_sample.json"
        console.print("[yellow]--self-test: using tiny bundled format fixtures, "
                      "not the real datasets[/yellow]")
    if args.download_locomo and locomo_path is None:
        locomo_path = download_locomo(ROOT / "benchmarks" / "datasets" / "locomo10.json")

    work_dir = Path(tempfile.mkdtemp(prefix="strata-membench-"))
    force_heuristic = not args.llm_extract
    results: list[BenchmarkResult] = []

    if locomo_path:
        console.print(f"[bold]LoCoMo[/bold] ({locomo_path})")
        results.append(run_locomo(locomo_path, work_dir, k=args.k,
                                  limit_conversations=args.limit, limit_qa=args.limit_qa,
                                  llm=llm, force_heuristic=force_heuristic,
                                  progress=console.print))
    else:
        console.print(f"[dim]LoCoMo skipped — pass --locomo <path> or --download-locomo[/dim]")

    if lme_path:
        console.print(f"[bold]LongMemEval[/bold] ({lme_path})")
        results.append(run_longmemeval(lme_path, work_dir, k=args.k, limit=args.limit,
                                       llm=llm, force_heuristic=force_heuristic,
                                       progress=console.print))
    else:
        console.print(f"[dim]LongMemEval skipped — {LME_INSTRUCTIONS}[/dim]")

    console.print("[bold]In-house evaluator[/bold]")
    results.append(run_inhouse(work_dir, k=args.k, repo=args.inhouse_repo,
                               progress=console.print))

    if not args.skip_beam:
        console.print(f"[bold]BEAM[/bold] ({'built-in fixture' if not args.beam else args.beam})")
        results.append(run_beam(
            args.beam, work_dir, k=args.k, limit=args.limit,
            llm=llm, force_heuristic=force_heuristic, progress=console.print,
        ))
    else:
        console.print("[dim]BEAM skipped (--skip-beam)[/dim]")

    if not args.skip_halumem:
        console.print(f"[bold]HaluMem[/bold] ({'built-in fixture' if not args.halumem else args.halumem})")
        bench_r, halu_r = run_halumem(
            args.halumem, work_dir, k=args.k, limit=args.limit,
            force_heuristic=force_heuristic, progress=console.print,
        )
        results.append(bench_r)
        # Print HaluMem-specific safety summary
        console.print(f"  [bold]HaluMem safety:[/bold] stale-leak={halu_r.stale_leak_rate():.3f} "
                      f"contamination={halu_r.contamination_rate():.3f} "
                      f"overall-safe={halu_r.overall_safety_rate():.3f}")
    else:
        console.print("[dim]HaluMem skipped (--skip-halumem)[/dim]")

    for r in results:
        print_benchmark(r)
    print_combined(results)
    path = save_results(results, args.out)
    console.print(f"\nresults saved -> [green]{path}[/green]")
    return results


if __name__ == "__main__":
    main()
