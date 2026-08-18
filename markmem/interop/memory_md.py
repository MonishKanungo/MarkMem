"""Claude Code MEMORY.md export (§4.2) — one memory file per page plus the
index format Claude Code's auto-memory already understands."""
from __future__ import annotations

from pathlib import Path

from ..storage.repo import Repo
from ..util import slugify


def export_memory_md(repo: Repo, out_dir: Path, user_id: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# Memory Index", ""]
    n = 0
    for page, body in repo.iter_pages():
        if page.status.value != "active":
            continue
        if user_id and page.user_id not in (None, user_id):
            continue
        name = f"{slugify(page.type)}-{slugify(page.title)}.md"
        lines = [
            "---",
            f"name: {slugify(page.title)}",
            f'description: "{(page.summary or page.title)[:150].replace(chr(34), chr(39))}"',
            "metadata:",
            f"  type: {page.type}",
            "---",
            "",
        ]
        active = page.active_claims()
        if active:
            lines += [f"- {c.text}" for c in active] + [""]
        if body.strip():
            lines.append(body.strip())
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        hook = (page.summary or page.title).splitlines()[0][:100]
        index_lines.append(f"- [{page.title}]({name}) — {hook}")
        n += 1
    (out_dir / "MEMORY.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return n
