"""SQLite FTS5 index — the DERIVED cache (never the source of truth).

Invariant: delete .strata/index.db and `reindex()` rebuilds everything from
wiki/*.md and raw/. Four FTS tables:

- pages_fts  — compiled pages (title/tags/summary/body)
- chunks_fts — page bodies split into small turn-window chunks. Whole-page BM25
               dilutes when the relevant lines are a tiny slice of a long
               episodic page; chunk-level matching fixes that (§10.1) and the
               best-matching chunk doubles as the evidence excerpt the packer
               puts into context.
- claims_fts — individual ledger claims (enables temporal + provenance search)
- raw_fts    — raw entries; the stopgap that makes writes searchable *before*
               async compilation lands (§5.2)

INDEX_SCHEMA_VERSION guards migrations: when the on-disk layout is older,
Memory triggers one automatic full reindex (cheap, derived data only).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..db import ConnectionPool
from ..models import Claim, Page, RawEntry
from ..obs import log
from ..storage.repo import Repo

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY, type TEXT, user_id TEXT, agent_id TEXT, run_id TEXT,
    title TEXT, tags TEXT, status TEXT, confidence REAL,
    created TEXT, updated TEXT, summary TEXT, pinned INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    id UNINDEXED, title, tags, summary, body, tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, page_id TEXT, subject TEXT, text TEXT,
    provenance TEXT, confidence REAL,
    valid_from TEXT, valid_until TEXT, recorded_at TEXT, supersedes TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_page ON claims(page_id);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id UNINDEXED, text, tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS raw_fts USING fts5(
    path UNINDEXED, user_id UNINDEXED, text, tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    page_id UNINDEXED, seq UNINDEXED, text, tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS page_sources (
    page_id TEXT NOT NULL, raw_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_raw ON page_sources(raw_path);
CREATE INDEX IF NOT EXISTS idx_sources_page ON page_sources(page_id);
"""

INDEX_SCHEMA_VERSION = "4"          # bumped when tables/semantics change

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Query-side stopwords: dropped from OR-joined match expressions so BM25 ranks
# on content words instead of "what/did/the" noise (also fewer terms = faster).
# Small and English-only by design — if EVERY token is a stopword we fall back
# to using them all, so nothing becomes unsearchable.
_QUERY_STOPWORDS = frozenset("""
a an and are as at be but by did do does for from had has have how i in into is
it its me my of on or our s so that the their them they this to was we were
what when where which who whom why will with you your
""".split())


def fts_query(query: str) -> Optional[str]:
    """Build a safe OR-joined FTS5 match expression (partial-overlap queries
    still surface results; special characters can't break the parser).
    Stopwords are dropped when content words remain."""
    tokens = _TOKEN.findall(query.lower())
    if not tokens:
        return None
    content = [t for t in tokens if t not in _QUERY_STOPWORDS]
    return " OR ".join(f'"{t}"' for t in (content or tokens)[:32])


# ---------------- body chunking ----------------

_CHUNK_TARGET_CHARS = 280           # ~2-3 dialog turns; finer granularity beats
                                    # whole-page BM25 dilution (eval-tuned)
_CHUNK_MAX_PER_PAGE = 400           # safety cap for pathological pages
_DATE_LINE = re.compile(r"^\(.{0,60}\bon .{4,40}\)$")   # "(chat session on 8 May, 2023)"


def chunk_body(body: str) -> list[str]:
    """Split a page body into line-grouped chunks of ~_CHUNK_TARGET_CHARS.
    A leading date/context line like "(conversation session on 8 May, 2023)"
    is carried into every chunk so temporal queries match at chunk level."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return []
    prefix = ""
    if _DATE_LINE.match(lines[0].strip()):
        prefix = lines[0].strip()
        lines = lines[1:]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        # very long single lines are split on their own boundaries
        while len(line) > _CHUNK_TARGET_CHARS * 2:
            head, line = line[:_CHUNK_TARGET_CHARS * 2], line[_CHUNK_TARGET_CHARS * 2:]
            current.append(head)
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
        if size >= _CHUNK_TARGET_CHARS:
            chunks.append("\n".join(current))
            current, size = [], 0
    if current:
        chunks.append("\n".join(current))
    if prefix:
        chunks = [f"{prefix}\n{c}" for c in chunks] or [prefix]
    return chunks[:_CHUNK_MAX_PER_PAGE]


class Indexer:
    def __init__(self, strata_dir: Path):
        self.db_path = strata_dir / "index.db"
        self._pool = ConnectionPool(self.db_path, row_factory=sqlite3.Row)
        self._init_db()

    def _db(self):
        return self._pool.tx()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_DDL)

    def close(self) -> None:
        """Release all index-db handles (Windows needs this before index.db
        can be deleted)."""
        self._pool.close_all()

    # ---------------- writes ----------------

    def index_page(self, page: Page, body: str) -> None:
        with self._db() as conn:
            self._index_page_conn(conn, page, body)

    def _index_page_conn(self, conn: sqlite3.Connection, page: Page, body: str) -> None:
        conn.execute("DELETE FROM pages WHERE id=?", (page.id,))
        conn.execute("DELETE FROM pages_fts WHERE id=?", (page.id,))
        conn.execute(
            "DELETE FROM claims_fts WHERE claim_id IN (SELECT id FROM claims WHERE page_id=?)",
            (page.id,))
        conn.execute("DELETE FROM claims WHERE page_id=?", (page.id,))
        conn.execute(
            "INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (page.id, page.type, page.user_id, page.agent_id, page.run_id, page.title,
             " ".join(page.tags), page.status.value, page.confidence, page.created,
             page.updated, page.summary, int(page.pinned), json.dumps(page.metadata)),
        )
        conn.execute(
            "INSERT INTO pages_fts (id, title, tags, summary, body) VALUES (?,?,?,?,?)",
            (page.id, page.title, " ".join(page.tags), page.summary, body),
        )
        conn.execute("DELETE FROM chunks_fts WHERE page_id=?", (page.id,))
        for seq, chunk in enumerate(chunk_body(body)):
            conn.execute("INSERT INTO chunks_fts (page_id, seq, text) VALUES (?,?,?)",
                         (page.id, seq, chunk))
        conn.execute("DELETE FROM page_sources WHERE page_id=?", (page.id,))
        for src in page.sources:
            conn.execute("INSERT INTO page_sources (page_id, raw_path) VALUES (?,?)",
                         (page.id, src))
        for c in page.claims:
            conn.execute(
                "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?)",
                (c.id, page.id, c.subject, c.text, c.provenance.value, c.confidence,
                 c.valid_from, c.valid_until, c.recorded_at, c.supersedes),
            )
            conn.execute("INSERT INTO claims_fts (claim_id, text) VALUES (?,?)", (c.id, c.text))

    def remove_page(self, page_id: str) -> None:
        with self._db() as conn:
            self._remove_page_conn(conn, page_id)

    def index_pages(self, page_ids: list[str], repo: Repo) -> None:
        """Incremental update after a write batch (§9.5)."""
        for pid in page_ids:
            parsed = repo.read_page(pid)
            if parsed is None:
                self.remove_page(pid)
            else:
                self.index_page(*parsed)

    def index_raw(self, entry: RawEntry) -> None:
        """Raw entries are indexed CHUNKED (multiple rows per path): whole-entry
        BM25 dilutes exactly like whole-page BM25 did, and raw text is the only
        place verbatim wording survives when an LLM extractor summarizes.
        One executemany in one transaction keeps the add() hot path flat."""
        chunks = chunk_body(entry.text) or [entry.text]
        with self._db() as conn:
            conn.executemany(
                "INSERT INTO raw_fts (path, user_id, text) VALUES (?,?,?)",
                [(entry.path, entry.user_id, c) for c in chunks])

    def remove_user(self, user_id: str) -> None:
        with self._db() as conn:
            for (pid,) in conn.execute(
                    "SELECT id FROM pages WHERE user_id=?", (user_id,)).fetchall():
                self._remove_page_conn(conn, pid)
            conn.execute("DELETE FROM raw_fts WHERE user_id=?", (user_id,))

    def _remove_page_conn(self, conn: sqlite3.Connection, page_id: str) -> None:
        conn.execute("DELETE FROM pages WHERE id=?", (page_id,))
        conn.execute("DELETE FROM pages_fts WHERE id=?", (page_id,))
        conn.execute("DELETE FROM chunks_fts WHERE page_id=?", (page_id,))
        conn.execute("DELETE FROM page_sources WHERE page_id=?", (page_id,))
        conn.execute(
            "DELETE FROM claims_fts WHERE claim_id IN (SELECT id FROM claims WHERE page_id=?)",
            (page_id,))
        conn.execute("DELETE FROM claims WHERE page_id=?", (page_id,))

    def reindex(self, repo: Repo) -> int:
        """Full rebuild from markdown — the recovery path proving the invariant."""
        n = 0
        with self._db() as conn:
            for table in ("pages", "pages_fts", "claims", "claims_fts", "raw_fts",
                          "chunks_fts", "page_sources"):
                conn.execute(f"DELETE FROM {table}")
            for page, body in repo.iter_pages():
                self._index_page_conn(conn, page, body)
                n += 1
            if repo.raw_dir.exists():
                for path in sorted(repo.raw_dir.rglob("*.md")):
                    if "failed" in path.relative_to(repo.raw_dir).parts:
                        continue
                    entry = repo.read_raw(repo.rel(path))
                    if entry is not None:
                        conn.executemany(
                            "INSERT INTO raw_fts (path, user_id, text) VALUES (?,?,?)",
                            [(entry.path, entry.user_id, c)
                             for c in (chunk_body(entry.text) or [entry.text])])
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('index_schema', ?)",
                         (INDEX_SCHEMA_VERSION,))
        log.info("reindexed %d pages", n)
        return n

    def schema_current(self) -> bool:
        """False when the on-disk index predates this code's schema (e.g. no
        chunks yet) and a one-time reindex is needed."""
        with self._db() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='index_schema'").fetchone()
            if row and row["value"] == INDEX_SCHEMA_VERSION:
                return True
            # fresh empty DBs are current by definition — stamp and move on
            has_pages = conn.execute("SELECT 1 FROM pages LIMIT 1").fetchone()
            if not has_pages:
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('index_schema', ?)",
                             (INDEX_SCHEMA_VERSION,))
                return True
        return False

    # ---------------- queries ----------------

    def page_row(self, page_id: str) -> Optional[sqlite3.Row]:
        with self._db() as conn:
            return conn.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()

    def list_rows(self, type: Optional[str] = None, user_id: Optional[str] = None,
                  status: Optional[str] = None, pinned: Optional[bool] = None) -> list[sqlite3.Row]:
        clauses, params = [], []
        for col, val in (("type", type), ("user_id", user_id), ("status", status)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        if pinned is not None:
            clauses.append("pinned=?")
            params.append(int(pinned))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._db() as conn:
            return conn.execute(
                f"SELECT * FROM pages {where} ORDER BY updated DESC", params).fetchall()

    def search_pages(self, query: str, type: Optional[str] = None,
                     user_id: Optional[str] = None, statuses: tuple[str, ...] = ("active",),
                     limit: int = 20) -> list[dict[str, Any]]:
        """BM25-ranked page hits (rank position is what matters downstream)."""
        match = fts_query(query)
        filters, params = ["p.status IN (%s)" % ",".join("?" * len(statuses))], list(statuses)
        if type:
            filters.append("p.type=?")
            params.append(type)
        if user_id:
            filters.append("(p.user_id=? OR p.user_id IS NULL)")   # user + shared knowledge
            params.append(user_id)
        where = " AND ".join(filters)
        with self._db() as conn:
            if match is None:
                return []
            # rank in a slim inner query (rowid+rank only), snippet() outside:
            # the sorter otherwise materializes snippet() for EVERY matching
            # row, which is what made broad queries cost ~15ms at 1K pages.
            rows = conn.execute(
                f"""SELECT p2.*, snippet(pages_fts, 3, '[', ']', '…', 12) AS snip,
                           r.rank AS rank
                    FROM (
                        SELECT pages_fts.rowid AS rid, bm25(pages_fts) AS rank,
                               p.id AS pid
                        FROM pages_fts JOIN pages p ON p.id = pages_fts.id
                        WHERE pages_fts MATCH ? AND {where}
                        ORDER BY rank LIMIT ?
                    ) r
                    JOIN pages_fts ON pages_fts.rowid = r.rid
                    JOIN pages p2 ON p2.id = r.pid
                    ORDER BY r.rank""",
                [match, *params, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def search_chunks(self, query: str, type: Optional[str] = None,
                      user_id: Optional[str] = None, statuses: tuple[str, ...] = ("active",),
                      limit: int = 24) -> list[dict[str, Any]]:
        """BM25-ranked chunk hits joined to their pages. Small rows make this
        the precision signal for long episodic pages; the top chunk per page is
        the evidence excerpt the packer can show."""
        match = fts_query(query)
        if match is None:
            return []
        filters, params = ["p.status IN (%s)" % ",".join("?" * len(statuses))], list(statuses)
        if type:
            filters.append("p.type=?")
            params.append(type)
        if user_id:
            filters.append("(p.user_id=? OR p.user_id IS NULL)")
            params.append(user_id)
        where = " AND ".join(filters)
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT chunks_fts.page_id AS page_id, chunks_fts.text AS chunk,
                           bm25(chunks_fts) AS rank
                    FROM chunks_fts JOIN pages p ON p.id = chunks_fts.page_id
                    WHERE chunks_fts MATCH ? AND {where}
                    ORDER BY rank LIMIT ?""",
                [match, *params, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    # a page citing more raws than this is an aggregator (profile/entity/topic
    # hub): it gets found via its own chunks/claims, never via the raw vote —
    # otherwise it squats the top-k on every query (measured on LoCoMo).
    _RAW_VOTE_MAX_SOURCES = 3

    def pages_citing(self, raw_paths: list[str]) -> dict[str, list[str]]:
        """raw_path -> the SPECIFIC pages compiled from it. This is what lets a
        lexical match on immutable raw text vote for the page that summarized
        it — essential when an LLM extractor compresses sessions into clean
        prose that no longer contains the verbatim terms (§5.2 raw stopgap,
        promoted to a ranking signal). User profiles and aggregator pages
        (> _RAW_VOTE_MAX_SOURCES sources) are excluded."""
        if not raw_paths:
            return {}
        placeholders = ",".join("?" * len(raw_paths))
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT ps.raw_path, ps.page_id FROM page_sources ps
                    JOIN pages p ON p.id = ps.page_id
                    WHERE ps.raw_path IN ({placeholders}) AND p.type != 'user'
                      AND (SELECT COUNT(*) FROM page_sources ps2
                           WHERE ps2.page_id = ps.page_id) <= ?""",
                [*raw_paths, self._RAW_VOTE_MAX_SOURCES],
            ).fetchall()
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["raw_path"], []).append(r["page_id"])
        return out

    def search_claims(self, query: str, user_id: Optional[str] = None,
                      as_of: Optional[str] = None, include_superseded: bool = False,
                      limit: int = 50) -> list[dict[str, Any]]:
        match = fts_query(query)
        if match is None:
            return []
        filters, params = [], []
        if user_id:
            filters.append("(p.user_id=? OR p.user_id IS NULL)")
            params.append(user_id)
        if as_of:
            # bi-temporal point-in-time: believed true on that date (§4.3)
            filters.append("(COALESCE(substr(c.valid_from,1,10), substr(c.recorded_at,1,10)) <= ?)")
            params.append(as_of[:10])
            filters.append("(c.valid_until IS NULL OR substr(c.valid_until,1,10) > ?)")
            params.append(as_of[:10])
        elif not include_superseded:
            filters.append("c.valid_until IS NULL")
        where = ("AND " + " AND ".join(filters)) if filters else ""
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT c.*, p.type AS page_type, p.title AS page_title,
                           p.user_id AS page_user_id, p.status AS page_status,
                           p.summary AS page_summary, p.updated AS page_updated,
                           bm25(claims_fts) AS rank
                    FROM claims_fts
                    JOIN claims c ON c.id = claims_fts.claim_id
                    JOIN pages p ON p.id = c.page_id
                    WHERE claims_fts MATCH ? {where}
                    ORDER BY rank LIMIT ?""",
                [match, *params, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def search_raw(self, query: str, user_id: Optional[str] = None,
                   limit: int = 10) -> list[dict[str, Any]]:
        match = fts_query(query)
        if match is None:
            return []
        where, params = "", []
        if user_id:
            where = "AND (user_id=? OR user_id IS NULL)"
            params.append(user_id)
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT path, user_id, snippet(raw_fts, 2, '[', ']', '…', 12) AS snip,
                           bm25(raw_fts) AS rank
                    FROM raw_fts WHERE raw_fts MATCH ? {where}
                    ORDER BY rank LIMIT ?""",
                [match, *params, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def index_entries(self) -> list[tuple]:
        """(type, id, title, summary, confidence, status) for index.md
        regeneration from the (already fresh) derived index — O(1) markdown parses."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT type, id, title, summary, confidence, status FROM pages "
                "ORDER BY type, id").fetchall()
        return [tuple(r) for r in rows]

    def index_summary(self, max_chars: int = 4000) -> str:
        """SQLite-backed equivalent of Repo.index_summary (extractor routing)."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, title, tags FROM pages WHERE status='active' ORDER BY id").fetchall()
            claims = conn.execute(
                "SELECT page_id, subject FROM claims WHERE valid_until IS NULL").fetchall()
        
        subjects_by_page = {}
        for c in claims:
            if c['subject']:
                subjects_by_page.setdefault(c['page_id'], set()).add(c['subject'])

        lines = []
        for r in rows:
            pid = r['id']
            tags = ','.join((r['tags'] or '').split()[:5])
            subs = ','.join(sorted(list(subjects_by_page.get(pid, []))))
            lines.append(f"{pid} | {r['title']} | {tags} | {subs}")
        return "\n".join(lines)[:max_chars]

    def counts(self) -> dict[str, Any]:
        with self._db() as conn:
            by_type = dict(conn.execute(
                "SELECT type, COUNT(*) FROM pages GROUP BY type").fetchall())
            by_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM pages GROUP BY status").fetchall())
            claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            active_claims = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE valid_until IS NULL").fetchone()[0]
        return {"pages_by_type": by_type, "pages_by_status": by_status,
                "claims": claims, "active_claims": active_claims}
