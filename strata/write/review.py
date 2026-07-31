"""Review workflow — "memory PRs" (§6.6).

Policy (config `pipeline.review`):
- off   — everything the extractor emits is written directly.
- auto  — (default) lint-gated: ops carrying injection-shaped content or
          below min_confidence are quarantined to the review queue; the rest
          auto-merge. Low friction, catches the dangerous cases.
- gated — every op waits for human review.

Queued items are JSON files under .strata/review/ — inspectable with any
editor, decided with `strata review`. Every decision lands in the ops ledger,
and accepted ops are committed like any other write: the audit story is free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..models import PageOp
from ..obs import Ledgers
from ..util import short_hash, utcnow_iso
from .guard import find_injection


@dataclass
class GateDecision:
    apply: bool
    reasons: list[str]


def gate(op: PageOp, config: Config) -> GateDecision:
    mode = config.pipeline.review
    if mode == "off":
        return GateDecision(True, [])
    if mode == "gated":
        return GateDecision(False, ["review mode is 'gated'"])
    reasons = []
    flags = find_injection(op.body + "\n" + op.summary + "\n" +
                           "\n".join(c.text for c in op.claims))
    if flags:
        reasons.append(f"injection-shaped content: {', '.join(flags)}")
    if op.confidence < config.pipeline.min_confidence:
        reasons.append(f"confidence {op.confidence:.2f} < min {config.pipeline.min_confidence:.2f}")
    return GateDecision(not reasons, reasons)


class ReviewQueue:
    def __init__(self, strata_dir: Path, ledgers: Ledgers):
        self.dir = strata_dir / "review"
        self.ledgers = ledgers

    def add(self, op: PageOp, raw_path: str, reasons: list[str]) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        item_id = f"r-{short_hash(raw_path + op.title + utcnow_iso(), 10)}"
        payload = {
            "id": item_id, "queued_at": utcnow_iso(), "raw_path": raw_path,
            "reasons": reasons, "op": op.model_dump(mode="json", exclude_none=True),
        }
        (self.dir / f"{item_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return item_id

    def list(self) -> list[dict]:
        if not self.dir.exists():
            return []
        items = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return items

    def get(self, item_id: str) -> dict | None:
        path = self.dir / f"{item_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def pop(self, item_id: str, decision: str) -> PageOp | None:
        """Remove the item and return its op (accept) or None if missing.
        The caller applies + commits accepted ops."""
        item = self.get(item_id)
        if item is None:
            return None
        (self.dir / f"{item_id}.json").unlink()
        self.ledgers.record_op("review_decision", item_id=item_id, decision=decision,
                               raw_path=item["raw_path"], reasons=item["reasons"])
        return PageOp.model_validate(item["op"])
