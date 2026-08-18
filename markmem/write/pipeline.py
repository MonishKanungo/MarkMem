"""Async write pipeline (§6.2) — the fix for the spec's biggest flaw.

`add()` appends raw + enqueues and returns in milliseconds. A background
worker thread batches pending entries, extracts, resolves against the claim
ledger, gates through review policy, writes pages, and makes ONE git commit
per cycle (debounced batch commits, §5.4).

Durability: the queue is a SQLite table in .markmem/queue.db (WAL). Rows stuck
in 'processing' after a crash are reset to 'pending' on startup. Extraction
failures are dead-lettered to raw/failed/ and marked 'failed' — never dropped.

Consistency model (documented, embraced): read-your-writes is EVENTUAL for
compiled pages; raw text is searchable immediately via the raw FTS table.
`flush()` compiles synchronously for tests and turn-boundary use.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

from filelock import Timeout

from ..config import Config
from ..db import ConnectionPool
from ..models import RawEntry
from ..obs import Ledgers, log
from ..schema import Schema
from ..storage.git_backend import SubprocessGit, writer_lock
from ..storage.repo import Repo
from ..util import utcnow_iso
from .extractors.base import Extractor
from .resolve import apply_op
from .review import ReviewQueue, gate

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS adds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_path TEXT NOT NULL,
    pii TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | done | failed
    error TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adds_status ON adds(status);
"""


class WritePipeline:
    def __init__(self, repo: Repo, git: SubprocessGit, schema: Schema, config: Config,
                 extractor: Extractor, ledgers: Ledgers,
                 on_pages_written: Optional[Callable[[list[str]], None]] = None,
                 index_summary_provider: Optional[Callable[[], str]] = None):
        """on_pages_written runs after pages hit disk and BEFORE the batch
        commit — Memory uses it to update the derived index (incl. index.md)
        so one commit captures everything. index_summary_provider supplies the
        routing summary cheaply (Memory wires the SQLite one; the fallback
        parses the markdown)."""
        self.repo, self.git, self.schema, self.config = repo, git, schema, config
        self.extractor, self.ledgers = extractor, ledgers
        self.on_pages_written = on_pages_written or (lambda ids: None)
        self.index_summary = index_summary_provider or repo.index_summary
        # Choose review queue backend: v1 JSON files or v2 git branches
        if config.pipeline.review_backend == "git":
            from .staging_review import StagingReviewQueue
            self.review_queue = StagingReviewQueue(repo, git, ledgers, schema)
        else:
            self.review_queue = ReviewQueue(repo.markmem_dir, ledgers)
        self.db_path = repo.markmem_dir / "queue.db"
        self._pool = ConnectionPool(self.db_path)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._init_db()
        self._recover()

    # ---------------- queue (durable, crash-safe) ----------------

    def _db(self):
        return self._pool.tx()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA_SQL)

    def _recover(self) -> None:
        with self._db() as conn:
            n = conn.execute(
                "UPDATE adds SET status='pending', updated=? WHERE status='processing'",
                (utcnow_iso(),),
            ).rowcount
        if n:
            log.info("recovered %d in-flight queue entries after unclean shutdown", n)

    def enqueue(self, raw: RawEntry) -> int:
        now = utcnow_iso()
        with self._db() as conn:
            cur = conn.execute(
                "INSERT INTO adds (raw_path, pii, created, updated) VALUES (?,?,?,?)",
                (raw.path, json.dumps(raw.pii), now, now),
            )
            return cur.lastrowid

    def _claim_batch(self, limit: int) -> list[tuple[int, str]]:
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, raw_path FROM adds WHERE status='pending' ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            if rows:
                conn.executemany(
                    "UPDATE adds SET status='processing', updated=? WHERE id=?",
                    [(utcnow_iso(), r[0]) for r in rows],
                )
        return rows

    def _mark(self, row_id: int, status: str, error: str = "") -> None:
        with self._db() as conn:
            conn.execute("UPDATE adds SET status=?, error=?, updated=? WHERE id=?",
                         (status, error or None, utcnow_iso(), row_id))

    def pending_count(self) -> int:
        with self._db() as conn:
            return conn.execute("SELECT COUNT(*) FROM adds WHERE status='pending'").fetchone()[0]

    def stats(self) -> dict[str, int]:
        with self._db() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM adds GROUP BY status").fetchall()
        return {status: count for status, count in rows}

    # ---------------- processing ----------------

    def process_batch(self) -> int:
        """One worker cycle: claim → extract → gate → resolve → one commit → reindex hook.
        Safe to call from multiple threads: row-claiming is atomic and the writer
        lock serializes the repo mutation."""
        rows = self._claim_batch(self.config.pipeline.batch_size)
        if not rows:
            return 0
        try:
            lock = writer_lock(self.repo.root)
            with lock:
                return self._process_rows(rows)
        except Timeout:
            for row_id, _ in rows:                    # give the batch back
                self._mark(row_id, "pending")
            log.warning("writer lock busy; batch of %d requeued", len(rows))
            return 0

    def _process_rows(self, rows: list[tuple[int, str]]) -> int:
        written: list[str] = []
        reviewed = failed = 0
        index_summary = self.index_summary()

        # Batch extraction path (§9.8): pack multiple raw entries into one LLM call
        batch_size = self.config.pipeline.batch_extract_size
        if batch_size > 1 and hasattr(self.extractor, "extract_batch"):
            return self._process_rows_batched(rows, index_summary)

        for row_id, raw_path in rows:
            raw = self.repo.read_raw(raw_path)
            if raw is None:
                self._mark(row_id, "failed", f"raw entry missing: {raw_path}")
                failed += 1
                continue
            try:
                ops = self.extractor.extract(raw, index_summary, self.schema)
            except Exception as e:
                self.repo.write_failed(
                    json.dumps({"raw_path": raw_path, "error": str(e),
                                "extractor": self.extractor.name, "failed_at": utcnow_iso()},
                               indent=2),
                    reason=type(e).__name__,
                )
                self._mark(row_id, "failed", str(e))
                failed += 1
                log.error("extraction failed for %s: %s", raw_path, e)
                continue
            for op in ops:
                if op.user_id is None:
                    op.user_id = raw.user_id
                decision = gate(op, self.config)
                if decision.apply:
                    result = apply_op(self.repo, self.schema, op, raw_path, raw.pii,
                                      agent_id=raw.agent_id, run_id=raw.run_id)
                    written.append(result.page_id)
                else:
                    self.review_queue.add(op, raw_path, decision.reasons)
                    reviewed += 1
            self._mark(row_id, "done")
        if written:
            unique_pages = sorted(set(written))
            self.on_pages_written(unique_pages)       # derived index + index.md, pre-commit
            self.git.commit_all(
                f"markmem: compile {len(rows) - failed} raw entr"
                f"{'y' if len(rows) - failed == 1 else 'ies'} -> {len(unique_pages)} page(s)"
                + (f", {reviewed} queued for review" if reviewed else "")
            )
        elif reviewed:
            log.info("%d op(s) queued for review, nothing written", reviewed)
        return len(rows)

    def _process_rows_batched(self, rows: list[tuple[int, str]], index_summary: str) -> int:
        """Batch extraction: one LLM call for all rows in this cycle (§9.8)."""
        written: list[str] = []
        reviewed = failed = 0

        # Load raw entries, track which failed to load
        valid: list[tuple[int, str, object]] = []
        for row_id, raw_path in rows:
            raw = self.repo.read_raw(raw_path)
            if raw is None:
                self._mark(row_id, "failed", f"raw entry missing: {raw_path}")
                failed += 1
            else:
                valid.append((row_id, raw_path, raw))

        if not valid:
            return len(rows)

        entries = [r for _, _, r in valid]
        all_ops = self.extractor.extract_batch(entries, index_summary, self.schema)

        for (row_id, raw_path, raw), ops in zip(valid, all_ops):
            if not ops:
                # Batch produced nothing for this entry — fall back to single call
                try:
                    ops = self.extractor.extract(raw, index_summary, self.schema)
                except Exception as e:
                    self.repo.write_failed(
                        json.dumps({"raw_path": raw_path, "error": str(e),
                                    "extractor": self.extractor.name,
                                    "failed_at": utcnow_iso()}, indent=2),
                        reason=type(e).__name__,
                    )
                    self._mark(row_id, "failed", str(e))
                    failed += 1
                    continue
            for op in ops:
                if op.user_id is None:
                    op.user_id = raw.user_id
                decision = gate(op, self.config)
                if decision.apply:
                    result = apply_op(self.repo, self.schema, op, raw_path, raw.pii,
                                      agent_id=raw.agent_id, run_id=raw.run_id)
                    written.append(result.page_id)
                else:
                    self.review_queue.add(op, raw_path, decision.reasons)
                    reviewed += 1
            self._mark(row_id, "done")

        if written:
            unique_pages = sorted(set(written))
            self.on_pages_written(unique_pages)
            self.git.commit_all(
                f"markmem: batch-compile {len(valid)} entr"
                f"{'y' if len(valid) == 1 else 'ies'} -> {len(unique_pages)} page(s)"
                + (f", {reviewed} queued for review" if reviewed else "")
            )
        elif reviewed:
            log.info("%d op(s) queued for review (batch mode), nothing written", reviewed)
        return len(rows)

    def flush(self, timeout_s: float = 300.0) -> int:
        """Synchronously drain the queue (forces compilation — for tests and
        explicit turn boundaries). Returns entries processed."""
        import time
        total, deadline = 0, time.monotonic() + timeout_s
        while self.pending_count() > 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"flush timed out with {self.pending_count()} entries pending")
            n = self.process_batch()
            if n == 0:                                # another thread holds the batch
                time.sleep(0.05)
            total += n
        return total

    # ---------------- worker thread ----------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="markmem-writer", daemon=True)
        self._worker.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout_s)
            self._worker = None

    def close(self) -> None:
        """Stop the worker and release all queue-db handles (Windows needs this
        before queue.db can be deleted)."""
        self.stop()
        self._pool.close_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                processed = self.process_batch()
            except Exception:
                log.exception("write worker cycle failed")
                processed = 0
            if processed == 0:
                self._stop.wait(self.config.pipeline.interval_s)

    # ---------------- review decisions ----------------

    def review_accept(self, item_id: str) -> Optional[str]:
        """Apply a queued op; returns the written page id."""
        item = self.review_queue.get(item_id)
        if item is None:
            return None
        op = self.review_queue.pop(item_id, "accept")
        with writer_lock(self.repo.root):
            result = apply_op(self.repo, self.schema, op, item.get("raw_path", ""))
            self.on_pages_written([result.page_id])
            self.git.commit_all(f"markmem: review-accept {item_id} -> {result.page_id}")
        return result.page_id

    def review_reject(self, item_id: str) -> bool:
        return self.review_queue.pop(item_id, "reject") is not None
