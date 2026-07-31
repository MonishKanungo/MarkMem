from strata.models import Claim, Page, Provenance, SearchHit
from strata.read.pack import pack
from strata.read.search import effective_confidence, rrf
from strata.util import est_tokens, utcnow_iso

from conftest import add_and_flush


def test_rrf_rank_only_fusion():
    scores = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["c"] == 1 / 63
    assert rrf([], k=60) == {}


def test_effective_confidence_decays():
    now = utcnow_iso()
    assert abs(effective_confidence(0.8, now, 90) - 0.8) < 0.01
    old = "2020-01-01T00:00:00+00:00"
    assert effective_confidence(0.8, old, 90) < 0.01
    assert effective_confidence(0.8, "garbage", 90) == 0.8   # unparseable -> no decay


def test_search_scoping_and_stemming(mem):
    add_and_flush(mem, "I prefer terraform over cloudformation for infra", user_id="alice")
    add_and_flush(mem, "bob only ever talks about databases", user_id="bob")
    hits = mem.search("terraforming infrastructure", user_id="alice")   # porter stems
    assert hits and all(h["user_id"] in (None, "alice") for h in hits)
    assert not [h for h in mem.search("terraform", user_id="bob") if h["user_id"] == "alice"]


def test_search_excludes_archived_by_default(mem):
    add_and_flush(mem, "the kubernetes migration plan", user_id="alice")
    page_id = next(h["id"] for h in mem.search("kubernetes", user_id="alice"))
    mem.delete(page_id)                            # soft archive
    assert not [h for h in mem.search("kubernetes", user_id="alice") if h["id"] == page_id]
    hits = mem.search("kubernetes", user_id="alice", include_superseded=True)
    assert [h for h in hits if h["id"] == page_id]


def test_as_of_temporal_query(mem):
    add_and_flush(mem, "I prefer window seats", user_id="alice")
    # close + supersede via a later statement
    add_and_flush(mem, "I prefer aisle seats now.", user_id="alice")
    page, _ = mem.repo.read_page("u/alice/user/profile")
    old = next(c for c in page.claims if "window" in c.text)
    new = next(c for c in page.claims if "aisle" in c.text)
    # make the historical window explicit: old was true in May
    old.valid_from, old.valid_until = "2026-05-01", "2026-07-01"
    new.valid_from = "2026-07-01"
    mem.repo.write_page(page, _)
    mem.reindex()

    current = mem.searcher.search("seats preference", user_id="alice")
    assert any("aisle" in c.text for h in current for c in h.claims)
    assert not any("window" in c.text for h in current for c in h.claims)

    back_then = mem.searcher.search("seats preference", user_id="alice", as_of="2026-06-01")
    texts = [c.text for h in back_then for c in h.claims]
    assert any("window" in t for t in texts)
    assert not any("aisle" in t for t in texts)


def test_provenance_weighting_orders_user_stated_first(mem):
    from strata.models import ClaimDraft, PageOp
    from strata.write.resolve import apply_op
    apply_op(mem.repo, mem.schema, PageOp(
        op="update", type="user", user_id="u1", title="u1", summary="Profile of u1",
        claims=[ClaimDraft(text="enjoys alpine climbing trips", subject="taste:climbing",
                           provenance=Provenance.user_stated, confidence=0.9)]), "raw/a.md")
    apply_op(mem.repo, mem.schema, PageOp(
        op="update", type="user", user_id="u2", title="u2", summary="Profile of u2",
        claims=[ClaimDraft(text="enjoys alpine climbing trips", subject="taste:climbing",
                           provenance=Provenance.imported, confidence=0.9)]), "import:x")
    mem.reindex()
    hits = mem.searcher.search("alpine climbing", top_k=5)
    provs = [h.provenance for h in hits if h.provenance]
    assert provs and provs[0] == "user_stated"


def test_empty_and_hostile_queries(mem):
    add_and_flush(mem, "regular content here", user_id="alice")
    assert mem.search("", user_id="alice") == []
    assert mem.search("   !!! ---", user_id="alice") == []
    # FTS5 syntax characters must not crash
    assert isinstance(mem.search('"unclosed AND (paren OR', user_id="alice"), list)


def _mk_page(i, nclaims=3):
    now = utcnow_iso()
    claims = [Claim(id=f"c-{i}-{j}", text=f"fact {j} of page {i} " + "x" * 30,
                    recorded_at=now, confidence=0.9 - j * 0.1,
                    provenance=Provenance.user_stated, sources=["raw/a.md"])
              for j in range(nclaims)]
    return Page(type="user", id=f"u/u{i}/user/profile", title=f"u{i}", created=now,
                updated=now, summary=f"Summary of page {i}", claims=claims)


def test_pack_budget_and_ordering():
    standing = [(_mk_page(0), "body")]
    hits = [SearchHit(page_id=f"g/concept/h{i}", score=1.0 - i * 0.1, title=f"Hit {i}",
                      summary="s " * 20, type="concept", updated=utcnow_iso())
            for i in range(10)]
    packed = pack(standing, hits, token_budget=200)
    assert est_tokens(packed) <= 200
    # standing context comes first
    first_block = packed.split("\n\n")[1]
    assert first_block.startswith("[u/u0/user/profile")
    # dedupe: standing page never repeats even if it's also a hit
    packed2 = pack(standing, [SearchHit(page_id="u/u0/user/profile", score=9.9,
                                        updated=utcnow_iso())], token_budget=500)
    assert packed2.count("u/u0/user/profile") == 1


def test_pack_truncates_at_claim_boundary():
    page = _mk_page(1, nclaims=50)
    packed = pack([(page, "")], [], token_budget=120)
    lines = [ln for ln in packed.splitlines() if ln.startswith("- ")]
    assert lines                                     # some claims made it
    assert all(ln.endswith(("x", ")")) or len(ln) > 10 for ln in lines)  # whole lines only
    assert est_tokens(packed) <= 120


def test_pack_attribution_present():
    page = _mk_page(2, nclaims=1)
    packed = pack([(page, "")], [], token_budget=500)
    assert "(user_stated, 0.90)" in packed
    assert "[u/u2/user/profile | user | conf" in packed


def test_pack_empty_returns_empty():
    assert pack([], [], token_budget=100) == ""


def test_standing_context_includes_pinned(mem):
    add_and_flush(mem, "I am into rock gardens", user_id="alice")
    page, body = mem.repo.read_page("u/alice/user/profile")
    ctx = mem.search("anything at all", user_id="alice", format="context")
    assert "u/alice/user/profile" in ctx             # L0 even when search misses
    # pin a global page; it should join the standing context
    from strata.models import Page as P
    now = utcnow_iso()
    pinned = P(type="concept", id="g/concept/house-rules", title="House Rules",
               created=now, updated=now, summary="Always be kind.", pinned=True)
    mem.repo.write_page(pinned, "Always be kind.")
    mem.reindex()
    ctx2 = mem.search("anything at all", user_id="alice", format="context")
    assert "g/concept/house-rules" in ctx2
