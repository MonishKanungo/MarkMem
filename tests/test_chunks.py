"""Chunk-level retrieval (chunks_fts), stopword-filtered queries, evidence
packing, and the index schema migration — the R@5 fix set."""
import sqlite3

from markmem import Memory
from markmem.models import Page
from markmem.read.fts import (INDEX_SCHEMA_VERSION, chunk_body, fts_query,
                             _CHUNK_TARGET_CHARS)
from markmem.util import utcnow_iso

from conftest import add_and_flush


# ---------------- chunk_body ----------------

def test_chunk_body_empty_and_tiny():
    assert chunk_body("") == []
    assert chunk_body("   \n  \n") == []
    assert chunk_body("one line") == ["one line"]


def test_chunk_body_splits_and_preserves_lines():
    lines = [f"speaker: message number {i} with some padding text here" for i in range(40)]
    chunks = chunk_body("\n".join(lines))
    assert len(chunks) > 1
    # no line is ever split mid-way in normal input
    reassembled = [ln for c in chunks for ln in c.splitlines()]
    assert reassembled == lines


def test_chunk_body_carries_date_line_into_every_chunk():
    body = "(conversation session on 1:56 pm on 8 May, 2023)\n" + "\n".join(
        f"alice: filler message {i} " + "x" * 60 for i in range(30))
    chunks = chunk_body(body)
    assert len(chunks) > 1
    for c in chunks:
        assert c.startswith("(conversation session on 1:56 pm on 8 May, 2023)")


def test_chunk_body_handles_giant_single_line():
    body = "word " * 2000                      # one huge line, no newlines
    chunks = chunk_body(body)
    assert len(chunks) > 1
    assert all(len(c) <= _CHUNK_TARGET_CHARS * 2 + 10 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks).replace(" ", "") == ("word" * 2000)


def test_chunk_body_unicode():
    chunks = chunk_body("中文测试 émoji 🌊\n" * 50)
    assert chunks and all("中文测试" in c for c in chunks)


# ---------------- fts_query stopwords ----------------

def test_fts_query_drops_stopwords():
    q = fts_query("What did Bob do in June 2023?")
    assert '"what"' not in q and '"did"' not in q and '"in"' not in q
    assert '"bob"' in q and '"june"' in q and '"2023"' in q


def test_fts_query_all_stopwords_falls_back():
    q = fts_query("what is this")                  # nothing left after filtering
    assert q is not None and '"what"' in q         # falls back to all tokens
    assert fts_query("") is None


# ---------------- retrieval behavior ----------------

def _session_page(mem, uid, slug, lines, date="(chat session on 8 May, 2023)"):
    now = utcnow_iso()
    page = Page(type="session", id=f"u/{uid}/session/{slug}", title=slug,
                user_id=uid, created=now, updated=now, summary=f"session {slug}")
    mem.repo.write_page(page, date + "\n" + "\n".join(lines))
    mem.indexer.index_page(*mem.repo.read_page(page.id))
    return page.id


def test_chunk_match_beats_page_dilution(mem):
    """The page whose single relevant turn matches must outrank pages with
    diffuse weak matches — the exact long-episodic-page failure mode."""
    uid = "diluter"
    # 6 noise pages that mention 'garden' passingly among lots of text
    for i in range(6):
        _session_page(mem, uid, f"noise-{i}",
                      [f"a: we talked about the garden briefly today, item {i}"]
                      + [f"b: unrelated filler chatter number {j} about many topics" for j in range(25)])
    # 1 page where the answer turn is buried deep
    target = _session_page(mem, uid, "target",
                           [f"a: filler opener line {j} about the weather" for j in range(20)]
                           + ["b: I finally planted the heirloom tomato seedlings in the garden bed"]
                           + [f"a: filler closer line {j}" for j in range(10)])
    hits = mem.searcher.search("heirloom tomato seedlings garden", user_id=uid, top_k=5)
    assert hits and hits[0].page_id == target
    # the evidence chunk (not a 12-token snip) is carried on the hit
    assert "heirloom tomato seedlings" in hits[0].snippet


def test_packed_context_contains_evidence_chunk(mem):
    uid = "packer"
    _session_page(mem, uid, "trip",
                  ["a: how was the trip?",
                   "b: We hiked the Azores volcano rim at sunrise, unforgettable"]
                  + [f"a: filler {j}" for j in range(20)])
    ctx = mem.search("volcano rim hike", user_id=uid, format="context")
    assert "Azores volcano rim" in ctx              # answer text, not just summary
    assert "> " in ctx                              # quoted evidence lines


def test_chunks_removed_with_page_and_user(mem):
    uid = "gone"
    pid = _session_page(mem, uid, "solo", ["a: the xylophone concert was moving"])
    assert mem.searcher.search("xylophone concert", user_id=uid, top_k=5)
    mem.indexer.remove_page(pid)
    with sqlite3.connect(mem.indexer.db_path) as conn:
        left = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE page_id=?", (pid,)).fetchone()[0]
    assert left == 0
    assert not mem.searcher.search("xylophone concert", user_id=uid, top_k=5)


def test_page_update_replaces_chunks(mem):
    uid = "upd"
    pid = _session_page(mem, uid, "v", ["a: original walrus content"])
    page, _ = mem.repo.read_page(pid)
    mem.repo.write_page(page, "a: replaced narwhal content")
    mem.indexer.index_page(*mem.repo.read_page(pid))
    assert not mem.searcher.search("walrus", user_id=uid, top_k=5)
    hits = mem.searcher.search("narwhal", user_id=uid, top_k=5)
    assert hits and "narwhal" in hits[0].snippet


def test_reindex_rebuilds_chunks_invariant(tmp_path):
    """Delete .markmem entirely -> chunks come back from markdown (the invariant)."""
    import gc
    import shutil
    m = Memory(repo_path=tmp_path / "m", start_worker=False)
    add_and_flush(m, "user: the quokka sanctuary visit was on a rainy tuesday", user_id="q")
    assert m.searcher.search("quokka sanctuary", user_id="q", top_k=5)
    m.close()
    gc.collect()                      # release lingering sqlite handles on Windows
    import time; time.sleep(0.1)      # give Windows a moment to release file locks
    try:
        shutil.rmtree(tmp_path / "m" / ".markmem")
    except PermissionError:
        # Windows WAL lock race — retry once after a brief wait
        time.sleep(0.5)
        shutil.rmtree(tmp_path / "m" / ".markmem")
    m2 = Memory(repo_path=tmp_path / "m", start_worker=False)
    m2.reindex()
    hits = m2.searcher.search("quokka sanctuary", user_id="q", top_k=5)
    assert hits and "quokka" in (hits[0].snippet + hits[0].summary).lower()
    m2.close()


def test_schema_migration_auto_reindexes(tmp_path):
    """An index.db from the pre-chunk schema triggers exactly one rebuild."""
    m = Memory(repo_path=tmp_path / "m", start_worker=False)
    add_and_flush(m, "user: the capybara cafe opened downtown", user_id="mig")
    m.close()
    # simulate an old index: wipe chunks and stamp an older schema version
    db = tmp_path / "m" / ".markmem" / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('index_schema', '1')")
        conn.commit()
    m2 = Memory(repo_path=tmp_path / "m", start_worker=False)   # migration runs here
    assert m2.indexer.schema_current()
    hits = m2.searcher.search("capybara cafe", user_id="mig", top_k=5)
    assert hits and "capybara" in (hits[0].snippet + hits[0].summary).lower()
    with sqlite3.connect(db) as conn:
        v = conn.execute("SELECT value FROM meta WHERE key='index_schema'").fetchone()[0]
        n = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert v == INDEX_SCHEMA_VERSION and n > 0
    m2.close()


def test_supersession_and_temporal_unaffected(mem):
    """Regression guard: the R@5 changes must not disturb ledger semantics."""
    add_and_flush(mem, "I prefer window seats", user_id="alice")
    add_and_flush(mem, "I prefer aisle seats now.", user_id="alice")
    page, _ = mem.repo.read_page("u/alice/user/profile")
    active = [c.text for c in page.active_claims()]
    assert any("aisle" in t for t in active) and not any("window" in t for t in active)
    ctx = mem.search("seat preference", user_id="alice", format="context")
    profile_block = next(b for b in ctx.split("\n\n") if "user/profile" in b)
    assert "aisle" in profile_block and "window" not in profile_block
