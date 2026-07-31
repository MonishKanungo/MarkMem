"""Optional vector search plugins (L2 tier) — install with strata-memory[vector].

Protocols per §6.1; defaults per §8: model2vec static embeddings (query
embedding in microseconds on CPU) + sqlite-vec (brute-force, embedded, fine to
~1M vectors). The core never imports these — ``get_vector_index`` returns None
when the extras are absent and search silently stays lexical-only (L1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Protocol

from ..obs import log

DEFAULT_MODEL = "minishlab/potion-base-8M"


class Embedder(Protocol):
    dim: int
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Model2VecEmbedder:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(model_id)
        self.model_id = model_id
        self.dim = int(self.model.dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.encode(texts)]


class VectorIndex:
    """sqlite-vec table living inside index.db (still one derived cache file).
    A model_id mismatch (config change) drops and rebuilds the table."""

    def __init__(self, db_path: Path, embedder: Embedder):
        import sqlite_vec
        self._load_ext = sqlite_vec.loadable_path()
        self.db_path = db_path
        self.embedder = embedder
        self._ensure_table()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.enable_load_extension(True)
        conn.load_extension(self._load_ext)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='embedder_model'").fetchone() \
                if conn.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone() else None
            if row and row[0] != self.embedder.model_id:
                log.info("embedder changed (%s -> %s): rebuilding vectors", row[0], self.embedder.model_id)
                conn.execute("DROP TABLE IF EXISTS page_vecs")
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS page_vecs USING vec0("
                f"page_id TEXT PRIMARY KEY, embedding float[{self.embedder.dim}])")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedder_model', ?)",
                         (self.embedder.model_id,))

    @staticmethod
    def _doc_text(title: str, summary: str, body: str) -> str:
        return f"{title}\n{summary}\n{body}"[:2000]

    def upsert(self, items: list[tuple[str, str]]) -> None:
        """items: (page_id, text)."""
        if not items:
            return
        import struct
        vecs = self.embedder.embed([t for _, t in items])
        with self._conn() as conn:
            for (pid, _), vec in zip(items, vecs):
                blob = struct.pack(f"{len(vec)}f", *vec)
                conn.execute("DELETE FROM page_vecs WHERE page_id=?", (pid,))
                conn.execute("INSERT INTO page_vecs (page_id, embedding) VALUES (?,?)", (pid, blob))

    def remove(self, page_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM page_vecs WHERE page_id=?", (page_id,))

    def query(self, text: str, k: int = 20) -> list[tuple[str, float]]:
        import struct
        vec = self.embedder.embed([text])[0]
        blob = struct.pack(f"{len(vec)}f", *vec)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT page_id, distance FROM page_vecs WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT ?", (blob, k)).fetchall()
        return [(r[0], r[1]) for r in rows]


def get_vector_index(db_path: Path, model_id: str = DEFAULT_MODEL) -> Optional[VectorIndex]:
    try:
        embedder = Model2VecEmbedder(model_id)
        return VectorIndex(db_path, embedder)
    except ImportError:
        return None
    except Exception as e:                            # model download failure etc.
        log.warning("vector search unavailable: %s", e)
        return None
