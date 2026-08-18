"""Observability: structured logger, token ledger, ops log.

- tokens.jsonl — one line per LLM call: model, input/output tokens, operation.
- ops.jsonl    — one line per governance event: erasure tombstones, review
                 decisions, retention deletions. This is the non-git half of the
                 audit trail (git log is the other half).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from .util import utcnow_iso

log = logging.getLogger("markmem")

_write_lock = threading.Lock()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    record = {"ts": utcnow_iso(), **record}
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class Ledgers:
    """Per-repo JSONL ledgers living under .markmem/."""

    def __init__(self, markmem_dir: Path):
        self.tokens_path = markmem_dir / "tokens.jsonl"
        self.ops_path = markmem_dir / "ops.jsonl"

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int, op: str) -> None:
        _append_jsonl(self.tokens_path, {
            "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "op": op,
        })

    def record_op(self, event: str, **fields: Any) -> None:
        _append_jsonl(self.ops_path, {"event": event, **fields})

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def token_totals(self) -> dict[str, int]:
        totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        for rec in self._read(self.tokens_path):
            totals["input_tokens"] += int(rec.get("input_tokens", 0))
            totals["output_tokens"] += int(rec.get("output_tokens", 0))
            totals["calls"] += 1
        return totals

    def ops(self) -> list[dict[str, Any]]:
        return self._read(self.ops_path)
