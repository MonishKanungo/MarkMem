"""Per-thread SQLite connection cache.

Opening a connection costs ~3ms and closing ~3.5ms on NTFS — dwarfing the
sub-ms statements themselves on the add() hot path (§9). Each thread gets one
long-lived connection per database; ``close_all()`` releases every handle so
files can be deleted (reset(), tests) — mandatory on Windows.

Connections are created with check_same_thread=False solely so close_all()
may close them from another thread; *use* stays thread-confined via
threading.local.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional


class ConnectionPool:
    def __init__(self, db_path: Path, row_factory: Optional[Callable] = None):
        self.db_path = Path(db_path)
        self.row_factory = row_factory
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        if self.row_factory is not None:
            conn.row_factory = self.row_factory
        # WAL is set persistently at schema init; NORMAL skips the per-commit
        # fsync (~10ms on NTFS) — safe: both DBs are rebuildable derived state.
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def get(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = self._connect()
        self._local.conn = conn
        with self._lock:
            self._all.append(conn)
        return conn

    @contextmanager
    def tx(self):
        """Yield the thread's connection; commit on success, roll back on error
        (persistent connections must never leak an open transaction)."""
        conn = self.get()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close_all(self) -> None:
        with self._lock:
            for conn in self._all:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all.clear()
        self._local = threading.local()   # stale thread-local refs are dropped
