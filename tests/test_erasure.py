import json

from conftest import add_and_flush


def test_forget_scrub_removes_everything_and_tombstones(mem):
    add_and_flush(mem, "I am vegetarian and prefer window seats", user_id="alice")
    add_and_flush(mem, "bob is unrelated", user_id="bob")
    assert mem.get_all(user_id="alice")

    tombstone = mem.forget("alice")
    assert tombstone["pages_erased"] >= 1 and tombstone["mode"] == "scrub"

    # working tree: both path prefixes gone
    assert not (mem.repo.root / "wiki" / "u" / "alice").exists()
    assert not (mem.repo.root / "raw" / "u" / "alice").exists()
    # index: nothing searchable, including raw stopgap
    assert mem.get_all(user_id="alice") == []
    assert mem.search("vegetarian window seats", user_id="alice") == []
    assert mem.indexer.search_raw("vegetarian", user_id="alice") == []
    # bob untouched
    assert mem.get_all(user_id="bob")
    # audit: erasure commit + ops-ledger tombstone
    assert any("forget user alice" in c.message for c in mem.git.history(limit=5))
    events = [op for op in mem.ledgers.ops() if op["event"] == "erasure"]
    assert events and events[-1]["user_id"] == "alice"


def test_forget_scrub_keeps_history_honestly(mem):
    """scrub mode documents that git history still contains the data."""
    add_and_flush(mem, "I love secret waffles", user_id="carol")
    mem.forget("carol")
    log_all = mem.git._run("log", "--all", "--stat").stdout
    assert "u/carol" in log_all                       # history remains until rewrite mode


def test_forget_unknown_user_is_safe(mem):
    tombstone = mem.forget("never-existed")
    assert tombstone["pages_erased"] == 0


def test_delete_all_soft_archives(mem):
    add_and_flush(mem, "I like tea", user_id="dave")
    n = mem.delete_all("dave")
    assert n >= 1
    assert mem.get_all(user_id="dave") == []
    assert mem.get_all(user_id="dave", include_archived=True)   # files still there
    assert (mem.repo.root / "wiki" / "u" / "dave").exists()


def test_delete_all_hard_erases(mem):
    add_and_flush(mem, "I like tea", user_id="eve")
    n = mem.delete_all("eve", hard=True)
    assert n >= 1
    assert not (mem.repo.root / "wiki" / "u" / "eve").exists()


def test_merge_users_moves_pages_and_records_alias(mem):
    add_and_flush(mem, "I prefer window seats", user_id="anon-123")
    add_and_flush(mem, "I am vegetarian", user_id="alice")
    moved = mem.merge_users("anon-123", "alice")
    assert moved >= 1
    assert mem.get_all(user_id="anon-123") == []
    assert not (mem.repo.root / "wiki" / "u" / "anon-123").exists()
    assert not (mem.repo.root / "raw" / "u" / "anon-123").exists()
    page, _ = mem.repo.read_page("u/alice/user/profile")
    assert "anon-123" in page.aliases
    # merged user's pages now searchable under alice
    hits = mem.search("window seats", user_id="alice")
    assert hits


def test_merge_users_collision_keeps_both(mem):
    add_and_flush(mem, "I prefer window seats", user_id="a1")   # both create user/profile
    add_and_flush(mem, "I prefer aisle seats", user_id="a2")
    mem.merge_users("a1", "a2")
    pages = mem.repo.list_pages(user_id="a2")
    profile_ids = [p.id for p, _ in pages if p.type == "user"]
    assert len(profile_ids) == 2                       # collision disambiguated, nothing lost


def test_unicode_user_erasure(mem):
    add_and_flush(mem, "I like tea", user_id="Ünïcode Üser")
    assert mem.get_all(user_id="Ünïcode Üser")
    mem.forget("Ünïcode Üser")
    assert mem.get_all(user_id="Ünïcode Üser") == []
