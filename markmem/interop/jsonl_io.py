"""JSONL interchange (§4.2) — the lossless roundtrip format.

One JSON object per line, each a full MARKMEM-FORMAT page record (frontmatter +
body). `export → import` into a fresh repo reproduces every page and its
entire claim ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import FORMAT_VERSION, Page
from ..storage.repo import Repo


def export_jsonl(repo: Repo, out_path: Path) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"markmem_format": FORMAT_VERSION, "kind": "header"}) + "\n")
        for page, body in repo.iter_pages():
            record = {"kind": "page", **page.model_dump(mode="json", exclude_none=True),
                      "body": body}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def import_jsonl(repo: Repo, in_path: Path) -> list[str]:
    """Import pages; existing pages with the same id are overwritten (the git
    diff of the import commit is the review surface). Returns written ids."""
    written: list[str] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("kind") == "header":
            continue
        record.pop("kind", None)
        body = record.pop("body", "")
        page = Page.model_validate(record)
        repo.write_page(page, body)
        written.append(page.id)
    if written:
        repo.regenerate_index()
    return written
