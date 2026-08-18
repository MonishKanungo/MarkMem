"""Mem0 import/export (§4.2) — "leave Mem0 without losing your memory."

Mem0's export shape: a JSON array (or {"results": [...]} envelope) of
{"id", "memory", "user_id", "created_at", "updated_at", "metadata"} records.
Each becomes a claim with provenance=imported (trust-ceiling capped, §4.5) on
the user's profile page — imported facts never masquerade as user statements.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import ClaimDraft, PageOp, Provenance
from ..schema import Schema
from ..storage.repo import Repo
from ..write.resolve import apply_op


def import_mem0(repo: Repo, schema: Schema, in_path: Path,
                default_user: str = "imported") -> list[str]:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results") or data.get("memories") or []
    written: list[str] = []
    for rec in data:
        text = (rec.get("memory") or rec.get("text") or "").strip()
        if not text:
            continue
        user_id = rec.get("user_id") or default_user
        op = PageOp(
            op="update", type="user", user_id=user_id, title=user_id,
            summary=f"Profile of {user_id}", confidence=0.6,
            claims=[ClaimDraft(
                text=text, provenance=Provenance.imported, confidence=0.6,
                valid_from=(rec.get("created_at") or "")[:10] or None,
            )],
        )
        result = apply_op(repo, schema, op, source_path=f"import:mem0:{rec.get('id', '?')}")
        written.append(result.page_id)
    if written:
        repo.regenerate_index()
    return sorted(set(written))


def export_mem0(repo: Repo, out_path: Path) -> int:
    """Flatten pages to Mem0-shaped records (one per active claim; pages
    without claims export their summary). Lossy by design — use JSONL for
    full fidelity."""
    records = []
    for page, _ in repo.iter_pages():
        if page.status.value != "active":
            continue
        active = page.active_claims()
        if active:
            for c in active:
                records.append({
                    "id": c.id, "memory": c.text, "user_id": page.user_id,
                    "created_at": c.recorded_at, "updated_at": page.updated,
                    "metadata": {"page_id": page.id, "type": page.type,
                                 "provenance": c.provenance.value,
                                 "confidence": c.confidence},
                })
        elif page.summary:
            records.append({
                "id": page.id, "memory": page.summary, "user_id": page.user_id,
                "created_at": page.created, "updated_at": page.updated,
                "metadata": {"page_id": page.id, "type": page.type},
            })
    out_path.write_text(json.dumps({"results": records}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return len(records)
