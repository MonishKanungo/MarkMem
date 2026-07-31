"""LiteLLM extractor — compile memory with ANY of 100+ LLM providers.

LiteLLM normalises OpenAI, Anthropic, Gemini, Ollama, Groq, Bedrock, Azure,
Together, NVIDIA NIM and ~90 more behind one `completion()` call, so Strata
does not need a bespoke extractor per vendor.

Install:
    pip install "strata-memory[litellm]"

Configure — config.yaml:
    llm:
      provider: litellm
      compile_model: gpt-4o-mini              # any litellm model string
      litellm_api_base: http://localhost:11434 # optional (Ollama / proxies)

Configure — environment (overrides config.yaml):
    STRATA_LLM_PROVIDER=litellm
    STRATA_LLM_COMPILE_MODEL=groq/llama-3.1-8b-instant
    GROQ_API_KEY=...            # or OPENAI_API_KEY / GEMINI_API_KEY / ...

Model strings (provider prefix determines routing):
    gpt-4o-mini                     OPENAI_API_KEY
    claude-haiku-4-5-20251001       ANTHROPIC_API_KEY
    groq/llama-3.1-8b-instant       GROQ_API_KEY          (fastest)
    gemini/gemini-1.5-flash         GEMINI_API_KEY
    ollama/llama3.1                 none — local, set litellm_api_base
    nvidia_nim/meta/llama-3.1-8b-instruct   NVIDIA_API_KEY
    azure/<deployment>              AZURE_API_KEY + AZURE_API_BASE
    bedrock/anthropic.claude-3-haiku-20240307-v1:0   AWS credentials
    Full list: https://docs.litellm.ai/docs/providers

Structured output:
    Models with function-calling get forced tool-use (highest fidelity).
    Everything else falls back to JSON mode with a schema-guided prompt.
    Either way the result is validated against PageOpBatch; one retry on
    failure, then ExtractionError so the pipeline dead-letters instead of
    silently dropping the entry.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from pydantic import ValidationError

from ...config import Config
from ...models import PageOp, PageOpBatch, RawEntry
from ...obs import Ledgers, log
from ...schema import Schema
from .base import ExtractionError
from .anthropic_llm import SYSTEM_PROMPT

TOOL_NAME = "emit_page_ops"


# Model families known to support function/tool calling through LiteLLM.
# Matched as substrings against the lowercased model string.
_TOOL_USE_PATTERNS = (
    "gpt-4", "gpt-3.5-turbo", "gpt-5", "o1", "o3",
    "claude-", "anthropic/",
    "gemini/gemini-1.5", "gemini/gemini-2", "gemini/gemini-3",
    "groq/", "azure/", "bedrock/anthropic", "mistral-large",
    "nvidia_nim/",
)

_JSON_MODE_SUFFIX = """

Return ONLY a JSON object matching this schema — no prose, no markdown fences:

{schema}

The object MUST have an "ops" array. Each element MUST have:
  "op"         : "create" or "update"
  "type"       : one of the available page types
  "title"      : string
  "summary"    : string
  "body"       : string
  "confidence" : number between 0 and 1
  "claims"     : array of {{"text", "subject", "provenance", "confidence"}}
"""


def _tool_schema() -> dict[str, Any]:
    schema = PageOpBatch.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def supports_tool_use(model: str) -> bool:
    """True when the model family is known to support function calling."""
    m = model.lower()
    return any(p in m for p in _TOOL_USE_PATTERNS)


class LiteLLMExtractor:
    """Universal LLM extractor. Same contract as every other Extractor:
    ``extract(raw, index_summary, schema) -> list[PageOp]``."""

    name = "litellm"

    def __init__(self, config: Config, ledgers: Optional[Ledgers] = None,
                 client: Any = None):
        try:
            import litellm
        except ImportError as exc:                      # optional [litellm] extra
            raise ImportError(
                'LiteLLM extractor needs the extra: pip install "strata-memory[litellm]"'
            ) from exc
        self._litellm = client or litellm

        self.model = os.environ.get("STRATA_LLM_COMPILE_MODEL") or config.llm.compile_model
        self.api_base = getattr(config.llm, "litellm_api_base", None)
        self.ledgers = ledgers
        self._use_tools = supports_tool_use(self.model)

        # LiteLLM is chatty by default; keep Strata's logs readable.
        if not os.environ.get("LITELLM_LOG"):
            for attr, val in (("suppress_debug_info", True), ("verbose", False)):
                try:
                    setattr(self._litellm, attr, val)
                except Exception:
                    pass
        log.info("litellm extractor: model=%s tool_use=%s", self.model, self._use_tools)

    # ---------------- transport ----------------

    def _common_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self.model, "max_tokens": 4096, "timeout": 60}
        if self.api_base:
            kw["api_base"] = self.api_base
        return kw

    def _record(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if self.ledgers is not None and usage is not None:
            self.ledgers.record_tokens(
                self.model,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                "extract",
            )

    def _call_tools(self, messages: list[dict]) -> dict:
        response = self._litellm.completion(
            **self._common_kwargs(),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tools=[{"type": "function", "function": {
                "name": TOOL_NAME,
                "description": "Emit the page operations extracted from the raw entry.",
                "parameters": _tool_schema(),
            }}],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        self._record(response)
        calls = getattr(response.choices[0].message, "tool_calls", None)
        if not calls:
            raise ExtractionError("litellm: model returned no tool_calls")
        return json.loads(calls[0].function.arguments)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Some models wrap JSON in ```json fences despite instructions."""
        text = text.strip()
        if not text.startswith("```"):
            return text
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()

    def _call_json(self, prompt: str) -> dict:
        system = SYSTEM_PROMPT + _JSON_MODE_SUFFIX.format(
            schema=json.dumps(PageOpBatch.model_json_schema(), indent=2))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]

        # response_format is unsupported by some providers — degrade, don't fail.
        try:
            response = self._litellm.completion(
                **self._common_kwargs(), messages=messages,
                response_format={"type": "json_object"})
        except Exception as exc:
            log.debug("litellm: json response_format rejected (%s), retrying plain", exc)
            response = self._litellm.completion(**self._common_kwargs(), messages=messages)

        self._record(response)
        return json.loads(self._strip_fences(response.choices[0].message.content or "{}"))

    # ---------------- Extractor protocol ----------------

    @staticmethod
    def _build_prompt(raw: RawEntry, index_summary: str, schema: Schema) -> str:
        type_lines = "\n".join(
            f"- {name} (decay: {rule.decay})" for name, rule in schema.page_types.items()
        ) or "- session"
        return (
            f"Page types available:\n{type_lines}\n\n"
            f"Existing pages (id | title | tags | active_subjects):\n"
            f"{index_summary or '(none yet)'}\n\n"
            f"Raw entry (source_type={raw.source_type}, "
            f"user_id={raw.user_id or 'none'}, recorded {raw.created}):\n"
            f"<raw>\n{raw.text}\n</raw>"
        )

    def extract(self, raw: RawEntry, index_summary: str, schema: Schema) -> list[PageOp]:
        prompt = self._build_prompt(raw, index_summary, schema)
        messages: list[dict] = [{"role": "user", "content": prompt}]
        last_error = ""

        for attempt in (1, 2):
            try:
                data = self._call_tools(messages) if self._use_tools \
                    else self._call_json(prompt)
                return PageOpBatch.model_validate(data).ops
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                log.warning("litellm extraction invalid (attempt %d/2): %s",
                            attempt, last_error)
                messages = [{"role": "user", "content": prompt}, {
                    "role": "user",
                    "content": f"Your previous output failed validation:\n{last_error}\n"
                               f"Re-emit valid output conforming exactly to the schema.",
                }]
            except Exception as exc:                     # transport / provider error
                last_error = str(exc)
                log.error("litellm extraction error (attempt %d/2, model=%s): %s",
                          attempt, self.model, last_error)

        raise ExtractionError(
            f"litellm extraction failed after retry (model={self.model}): {last_error}")

    def extract_batch(self, entries: list[RawEntry], index_summary: str,
                      schema: Schema) -> list[list[PageOp]]:
        """Concurrent per-entry extraction. Returns a list parallel to `entries`;
        an empty element signals failure so the pipeline can dead-letter it."""
        import concurrent.futures

        results: list[list[PageOp]] = [[] for _ in entries]

        def _one(idx: int, entry: RawEntry) -> None:
            try:
                results[idx] = self.extract(entry, index_summary, schema)
            except Exception as exc:
                log.error("litellm batch entry %s failed: %s", entry.path, exc)

        workers = min(len(entries), 8) or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            concurrent.futures.wait(
                [pool.submit(_one, i, e) for i, e in enumerate(entries)])
        return results
