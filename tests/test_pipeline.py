import sqlite3
import time

from strata import Memory
from strata.models import PageOp, RawEntry
from strata.write.extractors.base import ExtractionError


class FailingExtractor:
    name = "failing"

    def extract(self, raw, index_summary, schema):
        raise ExtractionError("boom")


def test_add_is_fast_and_eventual(mem):
    t0 = time.perf_counter()
    result = mem.add("I work at a property investment firm", user_id="alice")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result["status"] == "queued"
    assert elapsed_ms < 500                        # hot path: no LLM, no git
    assert mem.pipeline.pending_count() == 1
    assert mem.get_all(user_id="alice") == []      # compiled pages are eventual
    mem.flush()
    assert mem.pipeline.pending_count() == 0
    assert len(mem.get_all(user_id="alice")) >= 1


def test_raw_searchable_before_compile(mem):
    mem.add("the zanzibar project needs terraform modules", user_id="bob")
    hits = mem.searcher.search("zanzibar terraform", user_id="bob", include_raw=True)
    raw_hits = [h for h in hits if h.tier == "raw"]
    assert raw_hits and "zanzibar" in raw_hits[0].snippet.lower()


def test_batch_makes_single_commit(mem):
    before = len(mem.git.history(limit=100))
    for i in range(3):
        mem.add(f"note number {i} about kubernetes", user_id="carol")
    mem.flush()
    after = len(mem.git.history(limit=100))
    assert after == before + 1                     # one commit per batch, not per add


def test_extraction_failure_dead_letters(mem):
    mem.pipeline.extractor = FailingExtractor()
    mem.add("this will fail", user_id="dave")
    mem.flush()
    stats = mem.pipeline.stats()
    assert stats.get("failed") == 1
    dead = list((mem.repo.raw_dir / "failed").glob("*.md"))
    assert len(dead) == 1
    assert "boom" in dead[0].read_text(encoding="utf-8")


def test_crash_recovery_requeues_processing_rows(tmp_path):
    m1 = Memory(repo_path=tmp_path / "m", start_worker=False)
    m1.add("crash test entry", user_id="eve")
    # simulate a crash mid-processing
    with sqlite3.connect(m1.pipeline.db_path) as conn:
        conn.execute("UPDATE adds SET status='processing'")
        conn.commit()
    m1.close()
    m2 = Memory(repo_path=tmp_path / "m", start_worker=False)
    assert m2.pipeline.pending_count() == 1
    m2.flush()
    assert len(m2.get_all(user_id="eve")) >= 1
    m2.close()


def test_background_worker_compiles(tmp_path):
    m = Memory(repo_path=tmp_path / "w", start_worker=True)
    try:
        m.config.pipeline.interval_s = 0.1
        m.add("background compile of the mars rover project", user_id="frank")
        deadline = time.monotonic() + 15
        while m.pipeline.pending_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        # allow the in-flight batch to finish writing
        deadline = time.monotonic() + 5
        while not m.get_all(user_id="frank") and time.monotonic() < deadline:
            time.sleep(0.1)
        assert m.get_all(user_id="frank")
    finally:
        m.close()


def test_empty_add_skipped(mem):
    assert mem.add("   ")["status"] == "skipped"
    assert mem.pipeline.pending_count() == 0


def test_missing_raw_marks_failed(mem):
    entry = RawEntry(path="raw/g/conversation/ghost.md", text="x")
    mem.pipeline.enqueue(entry)
    mem.flush()
    assert mem.pipeline.stats().get("failed") == 1


def test_flush_from_two_threads_safe(mem):
    import threading
    for i in range(6):
        mem.add(f"parallel entry {i} about databases", user_id="gina")
    errors = []

    def run():
        try:
            mem.flush()
        except Exception as e:                     # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    assert mem.pipeline.pending_count() == 0
    assert mem.pipeline.stats().get("done") == 6
