from markmem.models import ClaimDraft, PageOp, Provenance
from markmem.schema import Schema
from markmem.storage.repo import Repo
from markmem.write.resolve import apply_op, canonical_page_id


def _schema():
    return Schema()


def _repo(tmp_path):
    return Repo.scaffold(tmp_path / "r")


def _op(claims, user_id="alice", **over):
    base = dict(op="update", type="user", title=user_id, user_id=user_id,
                summary=f"Profile of {user_id}", confidence=0.8)
    base.update(over)
    return PageOp(**base, claims=claims)


def draft(text, subject=None, provenance=Provenance.user_stated, confidence=0.85,
          valid_from=None):
    return ClaimDraft(text=text, subject=subject, provenance=provenance,
                      confidence=confidence, valid_from=valid_from)


def test_user_ops_converge_on_profile_page(tmp_path):
    repo = _repo(tmp_path)
    op = _op([draft("I like tea", "taste:tea")])
    assert canonical_page_id(op, repo) == "u/alice/user/profile"


def test_same_subject_supersedes(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("I prefer window seats", "preference:seat",
                                      valid_from="2026-05-01")]), "raw/a.md")
    result = apply_op(repo, schema, _op([draft("I prefer aisle seats", "preference:seat",
                                               valid_from="2026-07-06")]), "raw/b.md")
    assert len(result.supersessions) == 1
    page, _ = repo.read_page("u/alice/user/profile")
    old = next(c for c in page.claims if "window" in c.text)
    new = next(c for c in page.claims if "aisle" in c.text)
    assert old.valid_until == "2026-07-06"          # closed at new claim's valid_from
    assert new.supersedes == old.id
    assert new.valid_until is None
    assert [c.text for c in page.active_claims()] == ["I prefer aisle seats"]


def test_corroboration_bumps_confidence_capped(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("I prefer aisle seats", "preference:seat",
                                      confidence=0.97)]), "raw/a.md")
    result = apply_op(repo, schema, _op([draft("i prefer AISLE seats!", "preference:seat",
                                               confidence=0.9)]), "raw/b.md")
    assert result.claims_corroborated == 1 and result.claims_added == 0
    page, _ = repo.read_page("u/alice/user/profile")
    assert len(page.claims) == 1
    c = page.claims[0]
    assert c.confidence == 1.0                      # capped at user_stated ceiling
    assert set(c.sources) == {"raw/a.md", "raw/b.md"}


def test_trust_ceiling_caps_imported(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("loves skydiving", "taste:skydiving",
                                      provenance=Provenance.imported, confidence=0.99)]),
             "import:mem0:1")
    page, _ = repo.read_page("u/alice/user/profile")
    assert page.claims[0].confidence == 0.6         # imported ceiling


def test_different_subjects_coexist(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("I am vegetarian", "identity:vegetarian"),
                                draft("I prefer window seats", "preference:seat")]), "raw/a.md")
    page, _ = repo.read_page("u/alice/user/profile")
    assert len(page.active_claims()) == 2


def test_no_subject_never_supersedes(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("likes hiking", None)]), "raw/a.md")
    apply_op(repo, schema, _op([draft("hates hiking", None)]), "raw/b.md")
    page, _ = repo.read_page("u/alice/user/profile")
    assert len(page.active_claims()) == 2           # no subject key -> append only


def test_update_appends_details_section_and_merges_tags(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    op1 = PageOp(op="create", type="concept", title="AWS Strategy", tags=["aws"],
                 summary="s1", body="First body.")
    r1 = apply_op(repo, schema, op1, "raw/a.md")
    op2 = PageOp(op="update", page_id=r1.page_id, type="concept", title="AWS Strategy",
                 tags=["cost"], summary="s2", body="Second body.")
    apply_op(repo, schema, op2, "raw/b.md")
    page, body = repo.read_page(r1.page_id)
    assert page.tags == ["aws", "cost"]
    assert page.summary == "s2"
    assert page.sources == ["raw/a.md", "raw/b.md"]
    assert "First body." in body and "Second body." in body
    assert "## Details (updated" in body


def test_page_confidence_is_mean_of_active_claims(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    apply_op(repo, schema, _op([draft("a", "s:a", confidence=0.8),
                                draft("b", "s:b", confidence=0.6)]), "raw/a.md")
    page, _ = repo.read_page("u/alice/user/profile")
    assert abs(page.confidence - 0.7) < 1e-6


def test_fresh_evidence_revives_archived_page(tmp_path):
    repo, schema = _repo(tmp_path), _schema()
    r = apply_op(repo, schema, _op([draft("likes tea", "taste:tea")]), "raw/a.md")
    page, body = repo.read_page(r.page_id)
    page.status = "archived"
    repo.write_page(page, body)
    apply_op(repo, schema, _op([draft("likes green tea", "taste:green-tea")]), "raw/b.md")
    page, _ = repo.read_page(r.page_id)
    assert page.status.value == "active"
