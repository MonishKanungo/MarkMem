"""schema.md parsing — the user-editable ontology.

The schema is data, not code: prose for humans on top, one fenced ```yaml block
at the bottom that the library actually parses. Adding a page type never
requires a code change.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .models import Provenance

DEFAULT_TRUST_CEILINGS: dict[str, float] = {
    Provenance.user_stated.value: 1.0,
    Provenance.human_edited.value: 1.0,
    Provenance.tool_derived.value: 0.9,
    Provenance.agent_inferred.value: 0.8,
    Provenance.imported.value: 0.6,
}


class PageTypeRule(BaseModel):
    decay: str = "medium"
    retain_days: Optional[int] = None      # None → keep forever


class DecayRule(BaseModel):
    half_life_days: float = 90
    archive_below_confidence: Optional[float] = 0.2   # None → never auto-archived


class Schema(BaseModel):
    page_types: dict[str, PageTypeRule] = Field(default_factory=dict)
    decay_rules: dict[str, DecayRule] = Field(default_factory=dict)
    relationships: list[str] = Field(
        default_factory=lambda: ["relates_to", "supersedes", "part_of", "depends_on", "contradicts"]
    )
    trust_ceilings: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_TRUST_CEILINGS))

    def decay_for(self, page_type: str) -> DecayRule:
        cls = self.page_types.get(page_type, PageTypeRule()).decay
        return self.decay_rules.get(cls, DecayRule())

    def retain_days_for(self, page_type: str) -> Optional[int]:
        return self.page_types.get(page_type, PageTypeRule()).retain_days

    def ceiling_for(self, provenance: Provenance | str) -> float:
        key = provenance.value if isinstance(provenance, Provenance) else provenance
        return self.trust_ceilings.get(key, 1.0)

    def type_names(self) -> list[str]:
        return list(self.page_types) or ["session"]


_FENCE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)


class SchemaError(ValueError):
    pass


def parse_schema(text: str) -> Schema:
    """Parse the last fenced yaml block of schema.md into a Schema."""
    blocks = _FENCE.findall(text)
    if not blocks:
        raise SchemaError("schema.md has no fenced ```yaml block")
    try:
        data = yaml.safe_load(blocks[-1]) or {}
    except yaml.YAMLError as e:
        raise SchemaError(f"schema.md yaml block is invalid: {e}") from e
    if not isinstance(data, dict):
        raise SchemaError("schema.md yaml block must be a mapping")
    return Schema.model_validate(data)


def load_schema(path: Path) -> Schema:
    return parse_schema(path.read_text(encoding="utf-8"))
