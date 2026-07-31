"""Batch LLM extraction (§9.8) — amortize API call overhead across many raw entries.

The default extractor processes one raw entry per API call. At high volume this
means N calls for N entries — each with its own prompt overhead (~800 tokens of
system prompt + index summary). Batching packs B entries into one API call and
splits the response, reducing cost by ~(B-1)/B and latency by serialization.

Design decisions:
- Batch size B is config-controlled (pipeline.batch_extract_size, default 4)
- Each entry in the batch is clearly delimited so the LLM can reference them
- The response is a list of PageOpBatch, one per entry (positional mapping)
- If the response has fewer items than entries, remaining entries fall back
  to per-entry extraction (never silently dropped)
- Token ledger records the actual batch call, not B fake solo calls

Usage:
    The WritePipeline uses this automatically when pipeline.batch_extract_size > 1
    and the extractor is AnthropicExtractor. Heuristic extractor doesn't batch
    (it's fast enough that API overhead is irrelevant).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import ValidationError

from ..config import Config
from ..models import PageOp, PageOpBatch, RawEntry
from ..obs import Ledgers, log
from ..schema import Schema
from .extractors.base import ExtractionError, Extractor
from .extractors.anthropic_llm import SYSTEM_PROMPT, TOOL_NAME, _tool_schema


# Envelope for multi-entry extraction: the LLM returns a list of PageOpBatch,
# one element per input entry.
_BATCH_TOOL_NAME = "emit_batch_page_ops"

_BATCH_TOOL_DESCRIPTION = (
    "Emit page operations for EACH raw entry in the batch. "
    "The `batches` list must have exactly one PageOpBatch per input entry, "
    "in the same order as the entries were presented."
)


class BatchExtractor:
    """Wraps AnthropicExtractor to process multiple RawEntries in one API call.

    Falls back gracefully to single-entry extraction when:
    - The batch response is malformed
    - The response has fewer items than entries
    - Any individual item fails validation
    """

    name = "anthropic-batch"

    def __init__(self, config: Config, ledgers: Optional[Ledgers] = None, client: Any = None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = config.llm.compile_model
        self.ledgers = ledgers
        self.batch_size = config.pipeline.batch_extract_size

    def _batch_tool_schema(self) -> dict[str, Any]:
        """JSON schema for the batch envelope: {batches: [PageOpBatch, ...]}"""
        single_schema = PageOpBatch.model_json_schema()
        return {
            "type": "object",
            "properties": {
                "batches": {
                    "type": "array",
                    "description": "One PageOpBatch per input entry, in order.",
                    "items": single_schema,
                }
            },
            "required": ["batches"],
            "additionalProperties": False,
        }

    def _call_batch(self, prompt: str) -> Any:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,  # larger budget for batch output
            system=SYSTEM_PROMPT,
            tools=[{
                "name": _BATCH_TOOL_NAME,
                "description": _BATCH_TOOL_DESCRIPTION,
                "input_schema": self._batch_tool_schema(),
            }],
            tool_choice={"type": "tool", "name": _BATCH_TOOL_NAME},
            messages=[{"role": "user", "content": prompt}],
        )
        if self.ledgers and getattr(response, "usage", None):
            self.ledgers.record_tokens(
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                "batch-extract",
            )
        return response

    @staticmethod
    def _tool_input(response: Any) -> dict:
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _BATCH_TOOL_NAME:
                return block.input
        raise ExtractionError("batch: model returned no tool_use block")

    def _build_batch_prompt(self, entries: list[RawEntry], index_summary: str,
                             schema: Schema) -> str:
        type_lines = "\n".join(
            f"- {name} (decay: {rule.decay})" for name, rule in schema.page_types.items()
        ) or "- session"

        parts = [
            f"Page types available:\n{type_lines}\n",
            f"Existing pages (id | title | tags):\n{index_summary or '(none yet)'}\n",
            f"You will receive {len(entries)} raw entries. "
            f"Emit exactly {len(entries)} PageOpBatch objects in `batches`, one per entry.\n",
        ]
        for i, raw in enumerate(entries, 1):
            parts.append(
                f"--- ENTRY {i} of {len(entries)} "
                f"(source_type={raw.source_type}, user_id={raw.user_id or 'none'}, "
                f"recorded {raw.created}) ---\n{raw.text}\n"
            )
        return "\n".join(parts)

    def extract_batch(self, entries: list[RawEntry], index_summary: str,
                      schema: Schema) -> list[list[PageOp]]:
        """Extract PageOps for a list of entries in one API call.

        Returns a list parallel to `entries`; each element is the list of
        PageOps for that entry. Falls back per-entry on any failure.
        """
        if not entries:
            return []

        prompt = self._build_batch_prompt(entries, index_summary, schema)

        try:
            response = self._call_batch(prompt)
            data = self._tool_input(response)
            batches_raw = data.get("batches", [])

            results: list[list[PageOp]] = []
            for i, batch_data in enumerate(batches_raw):
                try:
                    results.append(PageOpBatch.model_validate(batch_data).ops)
                except ValidationError as e:
                    log.warning("batch entry %d validation failed: %s", i, e)
                    results.append([])  # will trigger per-entry fallback

            # If fewer batches than entries, pad with empty (fallback handles them)
            while len(results) < len(entries):
                results.append([])

            return results[:len(entries)]

        except Exception as e:
            log.error("batch extraction failed: %s — falling back to per-entry", e)
            return [[] for _ in entries]  # all fall back to per-entry

    def extract(self, raw: RawEntry, index_summary: str, schema: Schema) -> list[PageOp]:
        """Single-entry interface — delegates to the single AnthropicExtractor."""
        from .extractors.anthropic_llm import AnthropicExtractor
        from ..config import Config
        # Reuse the single extractor for consistency
        single = AnthropicExtractor.__new__(AnthropicExtractor)
        single.client = self.client
        single.model = self.model
        single.ledgers = self.ledgers
        return single.extract(raw, index_summary, schema)


class BatchWritePipeline:
    """Mixin that overrides _process_rows to use batch extraction.

    Mix this into WritePipeline when pipeline.batch_extract_size > 1 and
    the extractor is BatchExtractor.

    Usage: WritePipeline already calls process_batch() which calls _process_rows().
    This mixin replaces _process_rows with a batch-aware version.
    """

    def _process_rows_batch(self, rows: list[tuple[int, str]],
                             batch_extractor: "BatchExtractor",
                             schema: Schema) -> int:
        """Batch-extract all rows in one API call, then resolve individually."""
        from .resolve import apply_op
        from .review import gate

        # Load all raw entries first
        raws: list[tuple[int, str, Any]] = []  # (row_id, raw_path, raw_entry | None)
        for row_id, raw_path in rows:
            raw = self.repo.read_raw(raw_path)
            raws.append((row_id, raw_path, raw))

        # Separate valid from missing
        valid = [(rid, rp, r) for rid, rp, r in raws if r is not None]
        missing = [(rid, rp) for rid, rp, r in raws if r is None]

        for row_id, raw_path in missing:
            self._mark(row_id, "failed", f"raw entry missing: {raw_path}")

        if not valid:
            return len(rows)

        index_summary = self.index_summary()
        entries = [r for _, _, r in valid]

        # One API call for all entries
        all_ops = batch_extractor.extract_batch(entries, index_summary, schema)

        written: list[str] = []
        reviewed = failed = 0

        for (row_id, raw_path, raw), ops in zip(valid, all_ops):
            if not ops:
                # Batch failed for this entry — fall back to single extraction
                try:
                    ops = batch_extractor.extract(raw, index_summary, schema)
                except Exception as e:
                    self.repo.write_failed(
                        json.dumps({"raw_path": raw_path, "error": str(e),
                                    "extractor": batch_extractor.name,
                                    "failed_at": utcnow_iso()}, indent=2),
                        reason=type(e).__name__,
                    )
                    self._mark(row_id, "failed", str(e))
                    failed += 1
                    continue

            for op in ops:
                decision = gate(op, self.config)
                if decision.apply:
                    result = apply_op(self.repo, self.schema, op, raw_path, raw.pii,
                                      agent_id=raw.agent_id, run_id=raw.run_id)
                    written.append(result.page_id)
                else:
                    self.review_queue.add(op, raw_path, decision.reasons)
                    reviewed += 1
            self._mark(row_id, "done")

        if written:
            import json as _json
            unique_pages = sorted(set(written))
            self.on_pages_written(unique_pages)
            self.git.commit_all(
                f"strata: batch-compile {len(valid)} entr"
                f"{'y' if len(valid) == 1 else 'ies'} -> {len(unique_pages)} page(s)"
                + (f", {reviewed} queued for review" if reviewed else "")
            )

        return len(rows)
