"""`strata` CLI — init, ingest, search, governance, lifecycle, interop, evals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Strata — git-native memory layer (markdown + git, Mem0-shaped API).",
                  no_args_is_help=True)
console = Console()

_REPO_OPT = typer.Option(".", "--repo", "-r", help="Path to the memory repo")


def _memory(repo: str, start_worker: bool = False):
    from .memory import Memory
    return Memory(repo_path=repo, start_worker=start_worker)


@app.command()
def init(path: str = typer.Argument(".", help="Directory to initialize")):
    """Scaffold a new memory repo (schema.md, config.yaml, wiki/, raw/, git)."""
    m = _memory(path)
    m.close()
    console.print(f"[green]Initialized strata repo at[/green] {Path(path).resolve()}")


@app.command()
def ingest(text: str = typer.Argument(..., help="Raw text to ingest"),
           repo: str = _REPO_OPT,
           user: Optional[str] = typer.Option(None, "--user", "-u"),
           type: str = typer.Option("conversation", "--type", "-t"),
           origin: str = typer.Option("", "--origin"),
           no_wait: bool = typer.Option(False, "--no-wait", help="Enqueue only, don't compile")):
    """Ingest raw text and (by default) compile it synchronously."""
    with _memory(repo) as m:
        result = m.add(text, user_id=user, source_type=type, origin=origin)
        if not no_wait and result["status"] == "queued":
            m.flush()
        console.print(result)


@app.command()
def search(query: str, repo: str = _REPO_OPT,
           user: Optional[str] = typer.Option(None, "--user", "-u"),
           top_k: int = typer.Option(5, "--top-k", "-k"),
           as_of: Optional[str] = typer.Option(None, help="Point-in-time (YYYY-MM-DD)"),
           context: bool = typer.Option(False, "--context", help="Packed context format")):
    """Search compiled memory (BM25 + claims, RRF-fused when vectors installed)."""
    with _memory(repo) as m:
        if context:
            console.print(m.search(query, user_id=user, top_k=top_k, as_of=as_of,
                                   format="context") or "[dim](no context)[/dim]")
            return
        hits = m.search(query, user_id=user, top_k=top_k, as_of=as_of)
        if not hits:
            console.print("[dim]no results[/dim]")
            return
        table = Table("score", "id", "memory", "conf", "prov")
        for h in hits:
            md = h["metadata"]
            table.add_row(f"{h['score']:.4f}", h["id"], (h["memory"] or "")[:70],
                          f"{md['confidence']:.2f}", md.get("provenance") or "-")
        console.print(table)


@app.command("list")
def list_cmd(repo: str = _REPO_OPT,
             type: Optional[str] = typer.Option(None, "--type", "-t"),
             user: Optional[str] = typer.Option(None, "--user", "-u"),
             all: bool = typer.Option(False, "--all", help="Include archived/superseded")):
    """List pages."""
    with _memory(repo) as m:
        pages = m.get_all(user_id=user, type=type, include_archived=all)
        table = Table("id", "memory", "status", "conf", "claims", "updated")
        for p in pages:
            table.add_row(p["id"], (p["memory"] or "")[:50], p["metadata"]["status"],
                          f"{p['metadata']['confidence']:.2f}", str(len(p["claims"])),
                          p["updated_at"][:10])
        console.print(table)


@app.command()
def read(page_id: str, repo: str = _REPO_OPT):
    """Print one page (frontmatter + body)."""
    with _memory(repo) as m:
        parsed = m.repo.read_page(page_id)
        if parsed is None:
            console.print(f"[red]no such page:[/red] {page_id}")
            raise typer.Exit(1)
        from .storage.repo import dump_page_text
        console.print(dump_page_text(*parsed))


@app.command()
def history(page_id: str, repo: str = _REPO_OPT,
            diff: bool = typer.Option(False, "--diff", "-d")):
    """Audit trail of one page — git log --follow underneath."""
    with _memory(repo) as m:
        for entry in m.history(page_id, include_diff=diff):
            console.print(f"[yellow]{entry['commit'][:8]}[/yellow] {entry['date']} "
                          f"{entry['author']}: {entry['message']}")
            if diff and entry.get("diff"):
                console.print(entry["diff"], highlight=False)


@app.command()
def reindex(repo: str = _REPO_OPT):
    """Rebuild .strata/index.db entirely from the markdown (the invariant)."""
    with _memory(repo) as m:
        n = m.reindex()
        console.print(f"[green]reindexed[/green] {n} pages")


@app.command()
def flush(repo: str = _REPO_OPT):
    """Compile everything still queued."""
    with _memory(repo) as m:
        console.print(f"processed {m.flush()} queued entries")


@app.command()
def sweep(repo: str = _REPO_OPT,
          no_decay: bool = typer.Option(False), no_consolidate: bool = typer.Option(False),
          no_retention: bool = typer.Option(False)):
    """Run lifecycle jobs: decay, consolidation, retention (idempotent, committed)."""
    with _memory(repo) as m:
        report = m.maintenance(decay=not no_decay, consolidate=not no_consolidate,
                               retention=not no_retention)
        console.print(report)


@app.command()
def lint(repo: str = _REPO_OPT):
    """Check memory hygiene: broken links, unsourced claims, injection, drift."""
    with _memory(repo) as m:
        findings = m.lint()
        if not findings:
            console.print("[green]clean[/green]")
            return
        for f in findings:
            console.print(f"[yellow]{f.check}[/yellow] {f.page_id}: {f.detail}")
        raise typer.Exit(1)


@app.command()
def review(repo: str = _REPO_OPT,
           accept: Optional[str] = typer.Option(None, help="Accept a queued item id"),
           reject: Optional[str] = typer.Option(None, help="Reject a queued item id")):
    """Memory PRs: list/accept/reject quarantined write operations."""
    with _memory(repo) as m:
        if accept:
            page_id = m.pipeline.review_accept(accept)
            console.print(f"[green]applied ->[/green] {page_id}" if page_id
                          else f"[red]no such item:[/red] {accept}")
            return
        if reject:
            ok = m.pipeline.review_reject(reject)
            console.print("[green]rejected[/green]" if ok else f"[red]no such item:[/red] {reject}")
            return
        items = m.pipeline.review_queue.list()
        if not items:
            console.print("[dim]review queue empty[/dim]")
            return
        for item in items:
            op = item["op"]
            console.print(f"[yellow]{item['id']}[/yellow] ({', '.join(item['reasons'])})")
            console.print(f"  {op.get('op')} {op.get('type')}/{op.get('title', '')!r} "
                          f"claims={len(op.get('claims', []))} raw={item['raw_path']}")


@app.command()
def forget(user: str = typer.Argument(..., help="user_id to erase"),
           repo: str = _REPO_OPT,
           rewrite: bool = typer.Option(False, "--rewrite",
                                        help="Also purge from all git history (git-filter-repo)"),
           yes: bool = typer.Option(False, "--yes", "-y")):
    """Compliance erasure of one user (§4.1). scrub by default; --rewrite is provable."""
    if not yes:
        typer.confirm(f"Erase ALL memory of user {user!r}?", abort=True)
    with _memory(repo) as m:
        tombstone = m.forget(user, mode="rewrite" if rewrite else "scrub")
        console.print(tombstone)


@app.command("merge-users")
def merge_users_cmd(from_user: str, into_user: str, repo: str = _REPO_OPT):
    """Minimal identity resolution: fold one user id into another (§4.6)."""
    with _memory(repo) as m:
        moved = m.merge_users(from_user, into_user)
        console.print(f"moved {moved} pages from {from_user} -> {into_user}")


@app.command()
def stats(repo: str = _REPO_OPT):
    """Pages by type/status, claims, queue, review backlog, token spend."""
    with _memory(repo) as m:
        console.print_json(json.dumps(m.stats()))


@app.command()
def doctor(repo: str = typer.Option(None, "--repo", "-r")):
    """Environment + repo health checks."""
    import sqlite3

    from .storage.git_backend import git_available

    checks: list[tuple[str, bool, str]] = []
    fts_ok = "ENABLE_FTS5" in [r[0] for r in sqlite3.connect(":memory:").execute("PRAGMA compile_options")]
    checks.append(("SQLite FTS5", fts_ok, "required for search"))
    checks.append(("git on PATH", git_available(), "required"))
    import importlib.util as iu
    for mod, label in (("anthropic", "LLM extraction [llm]"), ("mcp", "MCP server [mcp]"),
                       ("model2vec", "vector search [vector]"), ("presidio_analyzer", "PII [pii]")):
        checks.append((label, iu.find_spec(mod) is not None, "optional"))
    import os
    checks.append(("ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY")),
                   "optional — heuristic extractor without it"))
    if repo:
        from .storage.repo import Repo
        r = Repo(repo)
        checks.append((f"repo at {r.root}", r.is_initialized, "run `strata init`"))
        if r.is_initialized:
            try:
                from .schema import load_schema
                load_schema(r.schema_path)
                checks.append(("schema.md parses", True, ""))
            except Exception as e:
                checks.append(("schema.md parses", False, str(e)))
    table = Table("check", "status", "note")
    ok = True
    for name, passed, note in checks:
        table.add_row(name, "[green]ok[/green]" if passed else "[red]MISSING[/red]", note)
        if not passed and note == "required":
            ok = False
    console.print(table)
    if not ok:
        raise typer.Exit(1)


@app.command("export")
def export_cmd(to: str = typer.Option("jsonl", help="jsonl | mem0 | memory-md"),
               out: str = typer.Option(..., "--out", "-o"),
               repo: str = _REPO_OPT,
               user: Optional[str] = typer.Option(None, "--user", "-u")):
    """Export memory to a portable format (§4.2)."""
    from .interop import export_jsonl, export_mem0, export_memory_md
    with _memory(repo) as m:
        if to == "jsonl":
            n = export_jsonl(m.repo, Path(out))
        elif to == "mem0":
            n = export_mem0(m.repo, Path(out))
        elif to == "memory-md":
            n = export_memory_md(m.repo, Path(out), user_id=user)
        else:
            console.print(f"[red]unknown format:[/red] {to}")
            raise typer.Exit(1)
        console.print(f"exported {n} records to {out}")


@app.command("import")
def import_cmd(path: str = typer.Argument(...),
               from_: str = typer.Option("jsonl", "--from", help="jsonl | mem0"),
               repo: str = _REPO_OPT):
    """Import memory from Mem0's export or Strata JSONL."""
    from .interop import import_jsonl, import_mem0
    with _memory(repo) as m:
        if from_ == "jsonl":
            written = import_jsonl(m.repo, Path(path))
        elif from_ == "mem0":
            written = import_mem0(m.repo, m.schema, Path(path))
        else:
            console.print(f"[red]unknown format:[/red] {from_}")
            raise typer.Exit(1)
        m.git.commit_all(f"strata: import {len(written)} page(s) from {from_}")
        m.reindex()
        console.print(f"imported {len(written)} pages")


@app.command("eval")
def eval_cmd(repo: str = _REPO_OPT, k: int = typer.Option(5, "--top-k", "-k")):
    """Domain eval generated from your own supersession history (§4.4)."""
    from .evals import run_domain_eval
    with _memory(repo) as m:
        result = run_domain_eval(m, k=k)
        if result.cases == 0:
            console.print("[dim]no supersession chains yet — ingest contradicting facts first[/dim]")
            return
        console.print(f"cases: {result.cases}  hit@{k}: {result.hit_at_k}  "
                      f"current-fact: {result.current_fact_rate}  "
                      f"stale-leak: {result.stale_leak_rate}")
        for f in result.failures[:10]:
            console.print(f"  [yellow]fail[/yellow] {f}")


@app.command()
def claim(page_id: str = typer.Argument(...),
          add: Optional[str] = typer.Option(None, help="Claim text to add"),
          subject: Optional[str] = typer.Option(None),
          close: Optional[str] = typer.Option(None, help="Claim id to close (valid_until=today)"),
          repo: str = _REPO_OPT):
    """Hand-edit the claim ledger without fighting YAML (§6.4 mitigation)."""
    from .models import Claim, Provenance
    from .util import new_claim_id, today_iso, utcnow_iso
    with _memory(repo) as m:
        parsed = m.repo.read_page(page_id)
        if parsed is None:
            console.print(f"[red]no such page:[/red] {page_id}")
            raise typer.Exit(1)
        page, body = parsed
        if add:
            page.claims.append(Claim(
                id=new_claim_id(add), text=add, subject=subject, valid_from=today_iso(),
                recorded_at=utcnow_iso(), confidence=1.0,
                provenance=Provenance.human_edited, sources=["human:cli"]))
            console.print("[green]claim added[/green]")
        if close:
            c = page.get_claim(close)
            if c is None:
                console.print(f"[red]no such claim:[/red] {close}")
                raise typer.Exit(1)
            c.valid_until = today_iso()
            console.print("[green]claim closed[/green]")
        if add or close:
            page.updated = utcnow_iso()
            m.repo.write_page(page, body)
            m.repo.regenerate_index()
            m.git.commit_all(f"strata: claim edit on {page_id}")
            m.indexer.index_pages([page_id], m.repo)
        else:
            for c in page.claims:
                mark = "" if c.is_active else f"  [dim](until {c.valid_until})[/dim]"
                console.print(f"[yellow]{c.id}[/yellow] ({c.provenance.value}, "
                              f"{c.confidence:.2f}) {c.text}{mark}")


@app.command()
def serve(repo: str = _REPO_OPT):
    """Run the MCP server (stdio) — needs `pip install strata-memory[mcp]`."""
    from .mcp_server import main as mcp_main
    mcp_main(repo)


@app.command()
def api(repo: str = _REPO_OPT,
        host: str = typer.Option("0.0.0.0", help="Bind host"),
        port: int = typer.Option(8000, help="Bind port"),
        reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)")):
    """Run the FastAPI REST server — needs `pip install strata-memory[api]`."""
    from .api import main as api_main
    api_main(repo_path=repo, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
