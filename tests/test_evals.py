from strata.evals import generate_domain_eval, run_domain_eval

from conftest import add_and_flush


def test_no_supersessions_no_cases(mem):
    add_and_flush(mem, "I like tea", user_id="alice")
    assert generate_domain_eval(mem) == []
    result = run_domain_eval(mem, save=False)
    assert result.cases == 0


def test_domain_eval_from_supersession_chain(mem):
    add_and_flush(mem, "I prefer window seats", user_id="alice")
    add_and_flush(mem, "I prefer aisle seats now.", user_id="alice")
    cases = generate_domain_eval(mem)
    assert len(cases) == 1
    case = cases[0]
    assert "current" in case.question and "seat" in case.question
    assert "aisle" in case.expected_text and "window" in case.superseded_text
    assert case.page_id == "u/alice/user/profile"

    result = run_domain_eval(mem, k=5, save=False)
    assert result.cases == 1
    assert result.hit_at_k == 1.0                     # profile page surfaces
    assert result.current_fact_rate == 1.0            # packed context has aisle claim
    assert result.stale_leak_rate == 0.0              # window never presented as current
    assert result.failures == []


def test_eval_results_saved_and_committed(mem):
    add_and_flush(mem, "I live in Lisbon", user_id="alice")
    add_and_flush(mem, "I live in Porto now.", user_id="alice")
    result = run_domain_eval(mem, save=True)
    saved = list((mem.repo.root / "evals").glob("domain-*.json"))
    assert saved
    assert any("eval run" in c.message for c in mem.git.history(limit=3))
    assert result.cases >= 1
