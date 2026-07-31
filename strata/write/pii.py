"""PII gate on the ingest path (§4.1).

Zero-dependency regex scanner in core; Presidio slots in behind the same
``scan()`` shape via the [pii] extra. Policies:

- off   — no scanning
- tag   — store as-is, record detected entity types in frontmatter/raw meta
- mask  — replace detected spans with [REDACTED:<TYPE>] *before* anything is
          persisted (raw/ is immutable, so masking must happen at the gate)
- block — reject the add() outright
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class PIIMatch:
    type: str
    start: int
    end: int
    value: str


class PIIScanner(Protocol):
    def scan(self, text: str) -> list[PIIMatch]: ...


class PIIBlockedError(ValueError):
    """Raised when policy=block and PII was detected."""

    def __init__(self, types: list[str]):
        self.types = types
        super().__init__(f"input blocked by PII policy (detected: {', '.join(types)})")


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(
        r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?![\w.])"
    )),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


class RegexScanner:
    """The dependency-free floor. Presidio (extra) is strictly better; this
    catches the classic high-signal shapes with low false-positive risk."""

    def scan(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for ptype, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(0)
                if ptype == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", value)
                    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                        continue
                if ptype == "PHONE" and sum(c.isdigit() for c in value) < 7:
                    continue
                matches.append(PIIMatch(ptype, m.start(), m.end(), value))
        # drop matches fully contained in an earlier, longer one (e.g. phone inside card)
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        kept: list[PIIMatch] = []
        for m in matches:
            if not any(k.start <= m.start and m.end <= k.end for k in kept):
                kept.append(m)
        return kept


def get_scanner() -> PIIScanner:
    try:
        from .presidio_scanner import PresidioScanner   # optional extra
        return PresidioScanner()
    except ImportError:
        return RegexScanner()


def apply_policy(text: str, policy: str, scanner: PIIScanner | None = None) -> tuple[str, list[str]]:
    """Returns (possibly-masked text, sorted detected types). Raises PIIBlockedError
    when policy=block and anything was found."""
    if policy == "off" or not text:
        return text, []
    scanner = scanner or get_scanner()
    matches = scanner.scan(text)
    types = sorted({m.type for m in matches})
    if not matches:
        return text, []
    if policy == "block":
        raise PIIBlockedError(types)
    if policy == "mask":
        out, last = [], 0
        for m in sorted(matches, key=lambda m: m.start):
            out.append(text[last:m.start])
            out.append(f"[REDACTED:{m.type}]")
            last = m.end
        out.append(text[last:])
        return "".join(out), types
    return text, types          # tag
