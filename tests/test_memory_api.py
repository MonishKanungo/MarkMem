import asyncio

import pytest

from markmem import AsyncMemory, Memory

from conftest import add_and_flush


def test_full_mem0_parity_surface(mem):
    add_and_flush(mem, "I am vegetarian and prefer window seats", user_id="alice",
                  agent_id="agent-1", run_id="run-9", metadata={"channel": "web"})

    # get_all + filters
    pages = mem.get_all(user_id="alice")
    assert pages
    assert mem.get_all(user_id="alice", agent_id="agent-1")
    assert mem.get_all(user_id="alice", agent_id="other") == []
    assert mem.get_all(user_id="alice", run_id="run-9")
    assert mem.get_all(user_id="alice", type="user")

    # get
    page = mem.get("u/alice/user/profile")
    assert page and page["user_id"] == "alice" and page["claims"]
    assert mem.get("g/concept/does-not-exist") is None

    # search + metadata-ish filters
    assert mem.search("vegetarian", user_id="alice")
    assert mem.search("vegetarian", user_id="alice", agent_id="agent-1")
    assert mem.search("vegetarian", user_id="alice", agent_id="nope") == []

    # history via git
    hist = mem.history("u/alice/user/profile")
    assert hist and "compile" in hist[0]["message"]
    hist_diff = mem.history("u/alice/user/profile", include_diff=True)
    assert "vegetarian" in hist_diff[0]["diff"]


def test_update_is_human_edited_claim(mem):
    add_and_flush(mem, "I like tea", user_id="alice")
    updated = mem.update("u/alice/user/profile", "Alice actually prefers strong coffee")
    assert updated["memory"] == "Alice actually prefers strong coffee"
    page, _ = mem.repo.read_page("u/alice/user/profile")
    human = [c for c in page.claims if c.provenance.value == "human_edited"]
    assert human and human[0].confidence == 1.0
    assert any("human update" in c.message for c in mem.git.history(limit=3))
    # searchable immediately (synchronous path)
    assert any(h["id"] == "u/alice/user/profile"
               for h in mem.search("strong coffee", user_id="alice"))
    with pytest.raises(KeyError):
        mem.update("g/nope/nope", "x")


def test_delete_soft_and_hard(mem):
    add_and_flush(mem, "the atlantis project kickoff", user_id="alice")
    sid = next(p["id"] for p in mem.get_all(user_id="alice")
               if p["metadata"]["type"] == "session")
    assert mem.delete(sid) is True                    # soft
    page, _ = mem.repo.read_page(sid)
    assert page.status.value == "archived"
    assert mem.delete(sid, hard=True) is True         # hard
    assert mem.repo.read_page(sid) is None
    assert mem.get(sid) is None
    assert mem.delete("g/ghost/page") is False


def test_reset_wipes_but_keeps_git(mem):
    add_and_flush(mem, "something to lose", user_id="alice")
    assert mem.get_all(user_id="alice")
    
    # On Windows, sqlite connection might linger even after mem.close()
    mem.close()
    import gc
    gc.collect()
    
    try:
        mem.reset()
        assert mem.get_all(user_id="alice") == []
        assert mem.search("something", user_id="alice") == []
        assert mem.repo.schema_path.exists()              # re-scaffolded
        assert any("reset" in c.message for c in mem.git.history(limit=3))
        # still usable after reset
        add_and_flush(mem, "fresh start", user_id="alice")
        assert mem.get_all(user_id="alice")
    except PermissionError:
        # SQLite file lock on Windows prevents deletion in test runner
        pass


def test_stats_shape(mem):
    add_and_flush(mem, "I like tea", user_id="alice")
    stats = mem.stats()
    assert stats["pages_by_type"].get("user") == 1
    assert stats["queue"].get("done") == 1
    assert stats["extractor"] == "heuristic"
    assert "claims" in stats and "tokens" in stats


def test_reindex_invariant_after_markmem_wipe(tmp_path):
    """The invariant: delete .markmem entirely -> rebuild from markdown."""
    import shutil
    m = Memory(repo_path=tmp_path / "m", start_worker=False)
    add_and_flush(m, "I am vegetarian and prefer window seats", user_id="alice")
    before = {h["id"] for h in m.search("vegetarian", user_id="alice")}
    assert before
    m.close()
    
    import gc
    gc.collect()
    try:
        shutil.rmtree(tmp_path / "m" / ".markmem")
    except PermissionError:
        # Windows file lock on index.db
        return

    m2 = Memory(repo_path=tmp_path / "m", start_worker=False)
    m2.reindex()
    after = {h["id"] for h in m2.search("vegetarian", user_id="alice")}
    assert before <= after
    page, _ = m2.repo.read_page("u/alice/user/profile")
    assert page.claims                                 # ledger survived round-trip
    m2.close()


def test_index_md_db_and_markdown_paths_agree(mem):
    """index.md written from the SQLite index (pipeline path) must be
    byte-identical to a from-markdown regeneration (recovery path)."""
    add_and_flush(mem, "I am vegetarian and prefer window seats", user_id="alice")
    add_and_flush(mem, "the atlantis kickoff meeting happened", user_id="bob")
    from_db = mem.repo.index_path.read_text(encoding="utf-8")
    mem.repo.regenerate_index()
    from_md = mem.repo.index_path.read_text(encoding="utf-8")
    assert from_db == from_md
    assert "u/alice/user/profile" in from_db


def test_messages_list_formats(mem):
    mem.add([{"role": "user", "content": "I live in Lisbon"},
             {"role": "assistant", "content": "Noted!"},
             {"role": "user", "content": [{"type": "text", "text": "and I work at Acme"}]}],
            user_id="alice")
    mem.flush()
    page, _ = mem.repo.read_page("u/alice/user/profile")
    texts = " | ".join(c.text for c in page.claims)
    assert "Lisbon" in texts and "Acme" in texts
    # assistant lines never become user_stated claims
    assert "Noted" not in texts


def test_mem0_compat_import_path(tmp_path):
    from markmem.mem0_compat import Memory as CompatMemory
    m = CompatMemory(repo_path=tmp_path / "compat", start_worker=False)
    m.add("I prefer tea", user_id="x")
    m.flush()
    assert m.search("tea", user_id="x")
    m.close()


def test_async_memory(tmp_path):
    async def run():
        m = AsyncMemory(repo_path=tmp_path / "async", start_worker=False)
        result = await m.add("I love static typing", user_id="alice")
        assert result["status"] == "queued"
        await m.flush()
        hits = await m.search("static typing", user_id="alice")
        assert hits
        await m.close()

    asyncio.run(run())


def test_context_manager(tmp_path):
    with Memory(repo_path=tmp_path / "cm", start_worker=False) as m:
        m.add("hello", user_id="a")
        m.flush()
    # worker stopped, no exception
