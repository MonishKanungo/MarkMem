"""Tiered read path (§6.3) with rank-only RRF fusion and honest scoring.

- L0 standing context: the user's profile page + pinned pages, no search.
- L1 lexical: FTS5 BM25 over pages and claims.
- L2 hybrid: vector candidates fused with L1 via RRF — only if [vector] extras
  are installed; otherwise search stays lexical, silently and correctly.

Final ordering = RRF positional score × provenance weight × freshness factor
(decay-adjusted effective confidence). Rank-only fusion means no score
calibration between BM25 and cosine distance is ever needed.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from ..config import Config
from ..models import Claim, Page, SearchHit
from ..schema import Schema
from ..storage.repo import Repo
from ..util import parse_iso, utcnow
from .fts import Indexer

PROVENANCE_WEIGHT = {
    "user_stated": 1.0, "human_edited": 1.0, "tool_derived": 0.92,
    "agent_inferred": 0.85, "imported": 0.75,
}


def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal-rank fusion: rank-only, calibration-free (§6.3 L2).
    Each list contributes at most once per item (first/best rank wins)."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        seen: set[str] = set()
        pos = 0
        for item in lst:
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos + 1)
            pos += 1
    return scores


def effective_confidence(confidence: float, updated: str, half_life_days: float) -> float:
    """Decay-adjusted confidence: conf * 0.5^(age/half_life)."""
    try:
        age_days = max(0.0, (utcnow() - parse_iso(updated)).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return confidence
    return confidence * math.pow(0.5, age_days / max(half_life_days, 0.001))


class Searcher:
    def __init__(self, repo: Repo, indexer: Indexer, schema: Schema, config: Config,
                 vector_index: Any = None):
        self.repo, self.indexer, self.schema, self.config = repo, indexer, schema, config
        self.vector_index = vector_index

    # ---------------- L0 ----------------

    def standing_context(self, user_id: Optional[str]) -> list[tuple[Page, str]]:
        """The always-relevant pages: user profile first, then pinned (§6.3 L0)."""
        out: list[tuple[Page, str]] = []
        if user_id:
            profile_id = f"u/{Repo.user_prefix(user_id).split('/', 1)[1]}/user/profile"
            parsed = self.repo.read_page(profile_id)
            if parsed and parsed[0].status.value == "active":
                out.append(parsed)
        for row in self.indexer.list_rows(pinned=True, status="active"):
            if row["user_id"] not in (None, user_id):
                continue
            parsed = self.repo.read_page(row["id"])
            if parsed and all(p.id != parsed[0].id for p, _ in out):
                out.append(parsed)
        return out

    # ---------------- L1/L2 ----------------

    def search(self, query: str, user_id: Optional[str] = None, type: Optional[str] = None,
               top_k: Optional[int] = None, as_of: Optional[str] = None,
               include_superseded: bool = False, include_raw: bool = False) -> list[SearchHit]:
        top_k = top_k or self.config.search.top_k
        statuses: tuple[str, ...] = ("active",) if not include_superseded \
            else ("active", "superseded", "archived")

        page_hits = self.indexer.search_pages(
            query, type=type, user_id=user_id, statuses=statuses, limit=max(top_k * 6, 30))
        claim_hits = self.indexer.search_claims(
            query, user_id=user_id, as_of=as_of,
            include_superseded=include_superseded, limit=50)
        chunk_hits = self.indexer.search_chunks(
            query, type=type, user_id=user_id, statuses=statuses, limit=max(top_k * 8, 40))

        # raw-source signal: immutable raw text keeps the verbatim terms that
        # LLM-compiled summaries lose; a raw match votes for the pages compiled
        # from it (mapping is one indexed lookup, no repo IO).
        raw_rows = self.indexer.search_raw(query, user_id=user_id, limit=max(top_k * 8, 40))
        citing = self.indexer.pages_citing([r["path"] for r in raw_rows])
        raw_ranked: list[str] = []
        for r in raw_rows:                             # rank order preserved
            raw_ranked.extend(citing.get(r["path"], []))

        # chunk-level matching is the precision signal for long episodic pages
        # (whole-page BM25 dilutes); it is listed twice = rank-only 2x weight in
        # RRF, justified by eval ablation (see PROJECT-STATE §6). rrf() dedupes
        # repeats within each list.
        chunk_ranked = [h["page_id"] for h in chunk_hits]
        ranked_lists = [[h["id"] for h in page_hits],
                        [h["page_id"] for h in claim_hits],
                        chunk_ranked, chunk_ranked,
                        raw_ranked, raw_ranked]        # raw = same 2x as chunks:
        # it carries the verbatim wording that summarized pages lose
        if self.vector_index is not None:
            try:
                ranked_lists.append([pid for pid, _ in self.vector_index.query(query, k=20)])
            except Exception:
                pass                                   # degrade to lexical, never fail a read
        fused = rrf(ranked_lists, k=self.config.search.rrf_k)

        by_page: dict[str, dict[str, Any]] = {h["id"]: h for h in page_hits}
        claims_by_page: dict[str, list[dict[str, Any]]] = {}
        for ch in claim_hits:
            claims_by_page.setdefault(ch["page_id"], []).append(ch)
        best_chunk: dict[str, str] = {}
        chunk_count: dict[str, int] = {}
        for ch in chunk_hits:                          # rows arrive rank-ordered
            n = chunk_count.get(ch["page_id"], 0)
            if n == 0:
                best_chunk[ch["page_id"]] = ch["chunk"]
            elif n == 1:                               # second-best adds recall in
                best_chunk[ch["page_id"]] += "\n…\n" + ch["chunk"]   # packed context
            chunk_count[ch["page_id"]] = n + 1

        hits: list[SearchHit] = []
        for page_id, base in sorted(fused.items(), key=lambda kv: -kv[1]):
            row = by_page.get(page_id) or self._row_for(page_id, statuses, type, user_id)
            if row is None:
                continue
            matched = claims_by_page.get(page_id, [])
            prov = self._dominant_provenance(matched, row)
            half_life = self.schema.decay_for(row["type"]).half_life_days
            eff = effective_confidence(row["confidence"] or 0.7, row["updated"] or "", half_life)
            score = base
            if self.config.search.provenance_weighting:
                score *= PROVENANCE_WEIGHT.get(prov or "", 0.85)
            score *= 0.6 + 0.4 * eff                  # freshness/trust nudge, never a veto
            # the best-matching chunk is richer evidence than the 12-token
            # FTS snippet — the packer can put the actual lines into context
            snippet = best_chunk.get(page_id) or row.get("snip") or ""
            hits.append(SearchHit(
                page_id=page_id, score=score, title=row["title"] or "",
                summary=row["summary"] or "", snippet=snippet,
                type=row["type"] or "", user_id=row["user_id"], status=row["status"] or "active",
                confidence=round(eff, 3), provenance=prov, updated=row["updated"] or "",
                tier="L2" if self.vector_index is not None else "L1",
                claims=[self._claim_from_row(c) for c in matched],
            ))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:top_k]

        if include_raw:
            for rh in self.indexer.search_raw(query, user_id=user_id, limit=3):
                hits.append(SearchHit(
                    page_id=rh["path"], score=0.0, title="(uncompiled raw entry)",
                    snippet=rh["snip"], type="raw", user_id=rh["user_id"], tier="raw",
                ))
        return hits

    def _row_for(self, page_id: str, statuses: tuple[str, ...], type: Optional[str],
                 user_id: Optional[str]) -> Optional[dict[str, Any]]:
        row = self.indexer.page_row(page_id)
        if row is None:
            return None
        d = dict(row)
        if d["status"] not in statuses:
            return None
        if type and d["type"] != type:
            return None
        if user_id and d["user_id"] not in (None, user_id):
            return None
        return d

    @staticmethod
    def _dominant_provenance(matched_claims: list[dict[str, Any]],
                             row: dict[str, Any]) -> Optional[str]:
        if not matched_claims:
            return None
        best = max(matched_claims,
                   key=lambda c: PROVENANCE_WEIGHT.get(c["provenance"], 0.0))
        return best["provenance"]

    @staticmethod
    def _claim_from_row(c: dict[str, Any]) -> Claim:
        return Claim(
            id=c["id"], text=c["text"], subject=c["subject"],
            valid_from=c["valid_from"], valid_until=c["valid_until"],
            recorded_at=c["recorded_at"] or "1970-01-01T00:00:00+00:00",
            confidence=c["confidence"] or 0.7, provenance=c["provenance"],
            supersedes=c["supersedes"],
        )
