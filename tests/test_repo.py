import pytest

from markmem.models import Claim, Page, Provenance
from markmem.storage.repo import Repo, RepoError
from markmem.util import slugify, utcnow_iso


@pytest.fixture()
def repo(tmp_path):
    return Repo.scaffold(tmp_path / "r")


def make_page(page_id="g/concept/test", **over):
    now = utcnow_iso()
    base = dict(type="concept", id=page_id, title="Test", created=now, updated=now,
                summary="A test page")
    base.update(over)
    return Page(**base)


def test_scaffold_layout(repo):
    assert repo.schema_path.exists()
    assert repo.config_path.exists()
    assert repo.index_path.exists()
    assert (repo.root / ".gitignore").read_text().strip() == ".markmem/"
    assert (repo.raw_dir / "failed").is_dir()


def test_page_roundtrip_with_claims_and_unicode(repo):
    claim = Claim(id="c-1", text="Prefers Café “Zürich” — naïve emoji 🌊", subject="preference:cafe",
                  valid_from="2026-05-01", recorded_at="2026-05-01T10:00:00+00:00",
                  confidence=0.9, provenance=Provenance.user_stated, sources=["raw/x.md"])
    page = make_page(claims=[claim], tags=["ünïcode", "café"], metadata={"k": [1, 2]})
    body = "Line one.\n\n---\n\nA body with a --- rule and 中文 text."
    repo.write_page(page, body)
    got_page, got_body = repo.read_page("g/concept/test")
    assert got_page.title == "Test"
    assert got_page.claims[0].text == claim.text
    assert got_page.claims[0].valid_from == "2026-05-01"       # date object coerced back to str
    assert got_page.claims[0].provenance == Provenance.user_stated
    assert got_page.metadata == {"k": [1, 2]}
    assert "中文" in got_body and "--- rule" in got_body


def test_path_is_authoritative_for_id(repo):
    page = make_page(id="g/concept/test")
    repo.write_page(page, "body")
    # hand-move the file: the id follows the path, not stale frontmatter
    src = repo.page_path("g/concept/test")
    dst = repo.wiki_dir / "g" / "concept" / "moved.md"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src.unlink()
    got, _ = repo.read_page("g/concept/moved")
    assert got.id == "g/concept/moved"


def test_page_id_traversal_blocked(repo):
    with pytest.raises(RepoError):
        repo.page_path("../../etc/passwd")


def test_user_vs_global_ids():
    assert Repo.make_page_id("concept", "My Idea!") == "g/concept/my-idea"
    uid = Repo.make_page_id("user", "profile", user_id="Alice Smith")
    assert uid == "u/alice-smith/user/profile"


def test_unicode_user_id_slug_is_stable_and_unique():
    a = Repo.user_prefix("Ünïcode Üser")
    b = Repo.user_prefix("Unicode User")
    assert a != b                       # lossy transliteration disambiguated by hash
    assert a == Repo.user_prefix("Ünïcode Üser")


def test_slugify_edge_cases():
    assert slugify("") == "page"
    assert slugify("!!!") .startswith("page-")
    assert slugify("Hello,   World! ") == "hello-world"
    assert len(slugify("x" * 500)) <= 60


def test_raw_append_immutable_and_unique(repo):
    e1 = repo.append_raw("first", user_id="alice", origin="s1")
    e2 = repo.append_raw("second", user_id="alice", origin="s1")
    assert e1.path != e2.path
    assert repo.read_raw(e1.path).text == "first"
    assert repo.read_raw(e2.path).text == "second"
    assert e1.path.startswith("raw/u/alice/")
    e3 = repo.append_raw("global", origin="g")
    assert e3.path.startswith("raw/g/")


def test_raw_roundtrip_metadata(repo):
    e = repo.append_raw("text", user_id="bob", agent_id="a1", run_id="r1",
                        metadata={"turn": 3}, pii=["EMAIL"])
    got = repo.read_raw(e.path)
    assert got.user_id == "bob" and got.agent_id == "a1" and got.run_id == "r1"
    assert got.metadata == {"turn": 3}
    assert got.pii == ["EMAIL"]
    assert got.created            # survives yaml datetime coercion


def test_index_regeneration(repo):
    repo.write_page(make_page(summary="Concept about AWS"), "body")
    repo.regenerate_index()
    index = repo.index_path.read_text(encoding="utf-8")
    assert "g/concept/test" in index and "Concept about AWS" in index


def test_list_pages_filters(repo):
    repo.write_page(make_page(id="g/concept/one"), "b")
    repo.write_page(make_page(id="u/alice/user/profile", type="user", user_id="alice"), "b")
    assert len(repo.list_pages()) == 2
    assert len(repo.list_pages(type="user")) == 1
    assert len(repo.list_pages(user_id="alice")) == 1
    assert repo.list_pages(user_id="alice")[0][0].id == "u/alice/user/profile"


def test_malformed_page_skipped_not_crash(repo):
    bad = repo.wiki_dir / "g" / "concept" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\ntype: [unclosed\n---\nbody", encoding="utf-8")
    repo.write_page(make_page(), "good")
    pages = repo.list_pages()
    assert [p.id for p, _ in pages] == ["g/concept/test"]


def test_index_summary_excludes_inactive(repo):
    page = make_page()
    repo.write_page(page, "b")
    assert "g/concept/test" in repo.index_summary()
    page.status = "archived"
    repo.write_page(page, "b")
    assert "g/concept/test" not in repo.index_summary()
