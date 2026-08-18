"""config.yaml — runtime knobs (models, pipeline, search, PII, review).

Env overrides: any key can be overridden with MARKMEM_<SECTION>_<KEY>,
e.g. MARKMEM_PIPELINE_REVIEW=gated, MARKMEM_PII_POLICY=mask.
API keys (ANTHROPIC_API_KEY) are read from the environment only, never stored.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = os.environ.get("MARKMEM_LLM_PROVIDER", "anthropic")
    compile_model: str = "claude-haiku-4-5-20251001"    # cheap, high-volume extraction
    judgment_model: str = "claude-sonnet-5"             # contradiction/lint judgment calls


class PipelineConfig(BaseModel):
    batch_size: int = 8            # raw entries per worker cycle
    interval_s: float = 2.0        # worker poll interval
    review: Literal["auto", "gated", "off"] = "auto"
    review_backend: Literal["json", "git"] = "json"   # v1=json files, v2=git branches
    min_confidence: float = 0.3    # ops below this are queued for review in auto mode
    consolidate_after: int = 5     # appended Details sections before consolidation kicks in
    batch_extract_size: int = 8    # entries per LLM call (1=per-entry, >1=batch mode §9.8)


class SearchConfig(BaseModel):
    token_budget: int = 2000
    top_k: int = 5
    provenance_weighting: bool = True
    rrf_k: int = 60


class PIIConfig(BaseModel):
    policy: Literal["off", "tag", "mask", "block"] = "tag"


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")


def _apply_env_overrides(data: dict) -> dict:
    prefix = "MARKMEM_"
    for name, value in os.environ.items():
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix):].lower().split("_", 1)
        if len(parts) != 2 or parts[0] not in Config.model_fields:
            continue
        section, key = parts
        data.setdefault(section, {})
        if isinstance(data[section], dict):
            val = yaml.safe_load(value)
            if val is False and value.lower() == "off":
                val = "off"
            data[section][key] = val
    return data


def load_config(path: Path | None) -> Config:
    data: dict = {}
    if path is not None and path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    return Config.model_validate(_apply_env_overrides(data))
