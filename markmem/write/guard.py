"""Injection lint — memory is an injection surface (§4.5).

A manipulated conversation can plant instruction-shaped content that gets
compiled into pages and re-injected into every future prompt. This flags the
known shapes so the review policy can quarantine them instead of storing
silently. Pattern list is deliberately high-precision: false positives cost
review friction, false negatives cost trust.
"""
from __future__ import annotations

import re

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override",
     re.compile(r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|prior|above|all)\b.{0,20}\binstructions?\b", re.I | re.S)),
    ("role_reassignment",
     re.compile(r"\byou (?:are now|must now act as|will now behave as)\b", re.I)),
    ("system_prompt_probe",
     re.compile(r"\b(?:reveal|print|repeat|output)\b.{0,30}\bsystem prompt\b", re.I)),
    ("tool_invocation_syntax",
     re.compile(r"<(?:antml:)?(?:invoke|function_calls|tool_use)\b|\"tool_use\"\s*:", re.I)),
    ("exfiltration_url",
     re.compile(r"\b(?:send|post|forward|upload)\b.{0,40}\bhttps?://", re.I)),
    ("shell_destruction",
     re.compile(r"\brm\s+-rf\s+[~/.]", re.I)),
]


def find_injection(text: str) -> list[str]:
    """Return the names of injection patterns present in the text (empty = clean)."""
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]
