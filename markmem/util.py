"""Small shared helpers: time, ids, slugs, token estimation."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import date, datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def today_iso() -> str:
    return utcnow().date().isoformat()


def to_iso(value: str | date | datetime | None) -> str | None:
    """Coerce YAML-parsed dates (which arrive as date/datetime objects) to ISO strings."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def parse_iso(value: str) -> datetime:
    """Parse an ISO timestamp or date into an aware UTC datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def new_claim_id(text: str) -> str:
    # random suffix: identical texts written in the same second must not collide
    return f"c-{today_iso()}-{short_hash(text, 4)}{uuid.uuid4().hex[:6]}"


_slug_strip = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60, fallback: str = "page") -> str:
    """Filesystem- and id-safe slug. Non-ascii is transliterated where possible;
    if sanitizing loses information, a short hash of the original keeps it unique."""
    original = text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _slug_strip.sub("-", text.lower()).strip("-")
    slug = slug[:max_len].strip("-")
    if not slug:
        return f"{fallback}-{short_hash(original)}" if original else fallback
    if slug != _slug_strip.sub("-", original.lower()).strip("-")[:max_len].strip("-"):
        # lossy sanitization (e.g. unicode user ids) — disambiguate
        slug = f"{slug}-{short_hash(original, 6)}"
    return slug


def est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — used only for context budgeting."""
    return max(1, len(text) // 4)
