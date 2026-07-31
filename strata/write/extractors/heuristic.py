"""Heuristic extractor — the zero-dependency, zero-API-key floor.

Deliberately modest (documented as such): one episodic session page per raw
entry, plus pattern-matched first-person facts as user_stated claims on the
user's profile page. Subject keys are normalized so the resolver can detect
same-subject contradictions offline (e.g. "prefers window seats" →
"preference:seat" → superseded by "prefers aisle seats").
"""
from __future__ import annotations

import re
from collections import Counter

from ...models import ClaimDraft, PageOp, Provenance, RawEntry
from ...schema import Schema
from ...util import slugify

_STOPWORDS = frozenset("""
a an and are as at be but by for from has have i if in into is it its my not of
on or our so that the their them they this to was we were will with you your me
now really very just also am
""".split())

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

# (subject-prefix, pattern) — patterns capture the object phrase as 'val'
_FACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("preference", re.compile(r"\bi\s+(?:now\s+|currently\s+)?prefer(?:s)?\s+(?P<val>[^.!?,;]+)", re.I)),
    ("taste", re.compile(r"\bi\s+(?:really\s+)?(?:like|love|hate|dislike|enjoy)\s+(?P<val>[^.!?,;]+)", re.I)),
    ("attribute", re.compile(r"\bmy\s+(?P<key>[\w ]{2,30}?)\s+is\s+(?P<val>[^.!?,;]+)", re.I)),
    ("identity", re.compile(r"\bi\s*(?:'m|\s+am)\s+(?:a\s+|an\s+)?(?P<val>[^.!?,;]+)", re.I)),
    ("employer", re.compile(r"\bi\s+work\s+(?:at|for)\s+(?P<val>[^.!?,;]+)", re.I)),
    ("location", re.compile(r"\bi\s+live\s+in\s+(?P<val>[^.!?,;]+)", re.I)),
]

_TEMPORAL_LEADIN = re.compile(r"^(?:now|currently|these days|from now on)\s+", re.I)


def _head_noun(phrase: str) -> str:
    """Last content word of the object phrase, crudely singularized — a stable
    key for 'same thing being talked about'."""
    words = [w.lower() for w in _WORD.findall(phrase) if w.lower() not in _STOPWORDS]
    if not words:
        return slugify(phrase, max_len=20, fallback="fact")
    head = words[-1]
    if len(head) > 3 and head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head


def _user_lines(text: str) -> str:
    """If the text looks like a role-tagged transcript, keep only user turns so
    assistant phrasing can't become user_stated claims."""
    lines = text.splitlines()
    tagged = [ln for ln in lines if re.match(r"^\s*(user|assistant|system)\s*:", ln, re.I)]
    if not tagged:
        return text
    kept = [re.sub(r"^\s*user\s*:\s*", "", ln, flags=re.I)
            for ln in lines if re.match(r"^\s*user\s*:", ln, re.I)]
    return "\n".join(kept)


_CLAUSE_SPLIT = re.compile(r"\s+(?:and|but)\s+|;\s*", re.I)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|my)\b", re.I)
_VERB_LEAD = re.compile(
    r"^(?:now\s+|currently\s+|also\s+|really\s+)?"
    r"(?:prefer|like|love|hate|dislike|enjoy|work|live|am)\b", re.I)


def _clauses(sentence: str) -> list[str]:
    """Split on conjunctions, carrying the first-person subject forward so
    'I am vegetarian and prefer window seats' yields two matchable clauses."""
    out, first_person_seen = [], False
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip().rstrip(".")
        if not clause:
            continue
        if _FIRST_PERSON.search(clause):
            first_person_seen = True
        elif first_person_seen and _VERB_LEAD.match(clause):
            clause = f"I {clause}"
        out.append(clause)
    return out


def extract_fact_claims(text: str) -> list[ClaimDraft]:
    claims: list[ClaimDraft] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence or len(sentence) > 300:
            continue
        for clause in _clauses(sentence):
            for prefix, pattern in _FACT_PATTERNS:
                m = pattern.search(clause)
                if not m:
                    continue
                val = _TEMPORAL_LEADIN.sub("", m.group("val").strip().rstrip("."))
                if not val:
                    continue
                if prefix == "attribute":
                    subject = f"attribute:{slugify(m.group('key'), max_len=30)}"
                elif prefix in ("employer", "location"):
                    subject = prefix
                elif prefix == "identity":
                    subject = f"identity:{_head_noun(val)}"
                else:
                    subject = f"{prefix}:{_head_noun(val)}"
                key = (subject, val.lower())
                if key in seen:
                    continue
                seen.add(key)
                claims.append(ClaimDraft(
                    text=clause if len(clause) <= 140 else f"{prefix}: {val}",
                    subject=subject,
                    provenance=Provenance.user_stated,
                    confidence=0.85,
                ))
                break                 # one claim per clause: first pattern wins
    return claims


def _keywords(text: str, n: int = 5) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    counts = Counter(w for w in words if w not in _STOPWORDS and len(w) > 2)
    return [w for w, _ in counts.most_common(n)]


class HeuristicExtractor:
    name = "heuristic"

    def extract(self, raw: RawEntry, index_summary: str, schema: Schema) -> list[PageOp]:
        text = raw.text.strip()
        if not text:
            return []
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "session")
        title = re.sub(r"^\s*(user|assistant)\s*:\s*", "", first_line, flags=re.I)[:60]
        summary = re.sub(r"\s+", " ", text)[:200]
        ops = [PageOp(
            op="create", type="session" if "session" in schema.type_names() else schema.type_names()[0],
            title=title or "session", user_id=raw.user_id, tags=_keywords(text),
            summary=summary, body=text, confidence=0.5,
        )]
        if raw.user_id:
            claims = extract_fact_claims(_user_lines(text))
            if claims:
                ops.append(PageOp(
                    op="update", page_id=None, type="user", title=raw.user_id,
                    user_id=raw.user_id, tags=[], confidence=0.85,
                    summary=f"Profile of {raw.user_id}",
                    body="", claims=claims,
                ))
        return ops
