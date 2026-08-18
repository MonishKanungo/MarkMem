"""Pydantic data models — the vocabulary shared by every layer.

These models define the MARKMEM-FORMAT frontmatter contract (see MARKMEM-FORMAT.md).
YAML hands back date/datetime objects for unquoted ISO strings, so every temporal
field coerces to ISO strings on input.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .util import to_iso

FORMAT_VERSION = "1.0"


class Provenance(str, Enum):
    user_stated = "user_stated"
    agent_inferred = "agent_inferred"
    tool_derived = "tool_derived"
    imported = "imported"
    human_edited = "human_edited"


class PageStatus(str, Enum):
    active = "active"
    superseded = "superseded"
    archived = "archived"


class Claim(BaseModel):
    """One bi-temporal, provenanced fact in a page's ledger (§6.4 of the architecture)."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    text: str
    subject: Optional[str] = None          # normalized key used for contradiction matching
    valid_from: Optional[str] = None       # event time: when the fact became true
    valid_until: Optional[str] = None      # event time: closed when superseded
    recorded_at: str                       # record time: when we learned it
    confidence: float = 0.7
    provenance: Provenance = Provenance.agent_inferred
    sources: list[str] = Field(default_factory=list)
    supersedes: Optional[str] = None       # id of the claim this one replaced

    _iso = field_validator("valid_from", "valid_until", "recorded_at", mode="before")(to_iso)

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    def valid_at(self, as_of: str) -> bool:
        """True if the claim was believed true on the given ISO date/timestamp."""
        frm = self.valid_from or self.recorded_at
        if frm and frm[:10] > as_of[:10]:
            return False
        return self.valid_until is None or self.valid_until[:10] > as_of[:10]


class Page(BaseModel):
    """A compiled wiki page. Frontmatter fields only; the body travels separately."""

    model_config = ConfigDict(validate_assignment=True)

    type: str
    id: str                                # path under wiki/ without .md, e.g. u/alice/user/profile
    title: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created: str
    updated: str
    confidence: float = 0.7                # page-level base confidence at last update
    status: PageStatus = PageStatus.active
    pinned: bool = False
    sources: list[str] = Field(default_factory=list)
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    pii: list[str] = Field(default_factory=list)   # PII entity types detected at ingest
    format: str = FORMAT_VERSION

    _iso = field_validator("created", "updated", mode="before")(to_iso)

    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_active]

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return next((c for c in self.claims if c.id == claim_id), None)

    def to_frontmatter(self) -> dict[str, Any]:
        """Ordered, minimal frontmatter dict (empty optionals omitted for hand-editability)."""
        d: dict[str, Any] = {
            "type": self.type, "id": self.id, "title": self.title,
        }
        for key in ("user_id", "agent_id", "run_id"):
            if getattr(self, key):
                d[key] = getattr(self, key)
        d.update({
            "tags": self.tags, "created": self.created, "updated": self.updated,
            "confidence": round(self.confidence, 3), "status": self.status.value,
        })
        if self.pinned:
            d["pinned"] = True
        d["sources"] = self.sources
        d["summary"] = self.summary
        if self.claims:
            d["claims"] = [c.model_dump(mode="json", exclude_none=True) for c in self.claims]
        if self.aliases:
            d["aliases"] = self.aliases
        if self.metadata:
            d["metadata"] = self.metadata
        if self.pii:
            d["pii"] = self.pii
        d["format"] = self.format
        return d


class RawEntry(BaseModel):
    """An immutable raw-source record under raw/."""

    path: str                              # repo-relative, e.g. raw/conversations/2026-...md
    text: str
    source_type: str = "conversation"
    origin: str = ""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created: str = ""
    pii: list[str] = Field(default_factory=list)

    _iso = field_validator("created", mode="before")(to_iso)


class ClaimDraft(BaseModel):
    """A claim proposed by an extractor, before ledger resolution."""

    text: str
    subject: Optional[str] = None
    provenance: Provenance = Provenance.agent_inferred
    confidence: float = 0.7
    valid_from: Optional[str] = None

    _iso = field_validator("valid_from", mode="before")(to_iso)


class PageOp(BaseModel):
    """One extractor-proposed operation against the wiki."""

    op: Literal["create", "update"]
    page_id: Optional[str] = None          # required for update; None → derived for create
    type: str = "session"
    title: str = ""
    user_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    summary: str = ""
    body: str = ""                         # prose to write/append
    confidence: float = 0.7
    claims: list[ClaimDraft] = Field(default_factory=list)


class PageOpBatch(BaseModel):
    """Structured-output envelope for LLM extraction (§5.7)."""

    ops: list[PageOp] = Field(default_factory=list)


class SearchHit(BaseModel):
    """One search result, with everything the packer and integrators need to cite it."""

    page_id: str
    score: float
    title: str = ""
    summary: str = ""
    snippet: str = ""
    type: str = ""
    user_id: Optional[str] = None
    status: str = "active"
    confidence: float = 0.7                # effective (decay-adjusted) confidence
    provenance: Optional[str] = None       # dominant provenance among matched claims
    updated: str = ""
    tier: str = "L1"                       # which read tier produced it (L0/L1/L2)
    claims: list[Claim] = Field(default_factory=list)   # matched/active claims

    def to_mem0(self) -> dict[str, Any]:
        """Mem0-shaped result dict."""
        return {
            "id": self.page_id,
            "memory": self.summary or self.snippet or self.title,
            "score": round(self.score, 6),
            "metadata": {
                "type": self.type, "status": self.status, "tier": self.tier,
                "confidence": round(self.confidence, 3), "provenance": self.provenance,
                "updated": self.updated, "title": self.title, "snippet": self.snippet,
            },
            "user_id": self.user_id,
        }
