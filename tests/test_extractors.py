from types import SimpleNamespace

import pytest

from markmem.config import Config
from markmem.models import RawEntry
from markmem.schema import Schema
from markmem.write.extractors.anthropic_llm import TOOL_NAME, AnthropicExtractor
from markmem.write.extractors.base import ExtractionError, get_extractor
from markmem.write.extractors.heuristic import HeuristicExtractor, extract_fact_claims


def _raw(text, user_id="alice"):
    return RawEntry(path="raw/u/x/conversation/t.md", text=text, user_id=user_id,
                    created="2026-07-07T00:00:00+00:00")


# ---------------- heuristic ----------------

def test_compound_sentence_yields_multiple_claims():
    claims = extract_fact_claims("I am vegetarian and prefer window seats.")
    subjects = {c.subject for c in claims}
    assert "identity:vegetarian" in subjects
    assert "preference:seat" in subjects


def test_subject_stability_across_phrasings():
    a = extract_fact_claims("I prefer window seats")[0]
    b = extract_fact_claims("I now prefer an aisle seat")[0]
    assert a.subject == b.subject == "preference:seat"


def test_attribute_employer_location_patterns():
    claims = extract_fact_claims(
        "My favorite color is blue. I work at Acme Corp. I live in Lisbon.")
    subjects = {c.subject for c in claims}
    assert "attribute:favorite-color" in subjects
    assert "employer" in subjects and "location" in subjects


def test_transcript_only_user_lines_become_claims():
    text = "user: I love sailing\nassistant: I am a large language model and I love that!"
    ext = HeuristicExtractor()
    ops = ext.extract(_raw(text), "", Schema())
    profile_ops = [op for op in ops if op.type == "user"]
    assert profile_ops
    texts = " | ".join(c.text for c in profile_ops[0].claims)
    assert "sailing" in texts
    assert "large language model" not in texts


def test_all_claims_user_stated_and_deduped():
    claims = extract_fact_claims("I like tea. I like tea. I like tea!")
    assert len(claims) == 1
    assert claims[0].provenance.value == "user_stated"


def test_empty_text_no_ops():
    assert HeuristicExtractor().extract(_raw("   "), "", Schema()) == []


def test_session_page_always_created():
    ops = HeuristicExtractor().extract(_raw("Discussed quarterly cloud budget."), "", Schema())
    assert ops[0].type == "session"
    assert ops[0].tags                                  # keyword tags extracted
    assert ops[0].user_id == "alice"


# ---------------- anthropic (fake client, no network) ----------------

def _fake_response(tool_input, usage=(100, 50)):
    block = SimpleNamespace(type="tool_use", name=TOOL_NAME, input=tool_input)
    return SimpleNamespace(content=[block],
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_llm_extractor_valid_output(tmp_path):
    from markmem.obs import Ledgers
    ledgers = Ledgers(tmp_path)
    good = {"ops": [{"op": "update", "type": "user", "user_id": "alice", "title": "alice",
                     "summary": "Profile of alice",
                     "claims": [{"text": "Prefers aisle seats", "subject": "preference:seat",
                                 "provenance": "user_stated", "confidence": 0.9}]}]}
    client = FakeClient([_fake_response(good)])
    ext = AnthropicExtractor(Config(), ledgers, client=client)
    ops = ext.extract(_raw("I prefer aisle seats"), "(none)", Schema())
    assert len(ops) == 1 and ops[0].claims[0].subject == "preference:seat"
    # forced tool use + token ledger recorded
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert ledgers.token_totals() == {"input_tokens": 100, "output_tokens": 50, "calls": 1}


def test_llm_extractor_retries_once_then_succeeds():
    bad = {"ops": [{"op": "definitely-not-valid", "type": "user"}]}
    good = {"ops": []}
    client = FakeClient([_fake_response(bad), _fake_response(good)])
    ext = AnthropicExtractor(Config(), client=client)
    assert ext.extract(_raw("x"), "", Schema()) == []
    assert len(client.calls) == 2
    # the retry conversation carries the validation error back
    retry_msgs = client.calls[1]["messages"]
    assert any("failed validation" in str(m) for m in retry_msgs)


def test_llm_extractor_fails_after_two_bad_outputs():
    bad = {"ops": [{"op": "nope", "type": "user"}]}
    client = FakeClient([_fake_response(bad), _fake_response(bad)])
    ext = AnthropicExtractor(Config(), client=client)
    with pytest.raises(ExtractionError):
        ext.extract(_raw("x"), "", Schema())


def test_llm_extractor_no_tool_block():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="chatty")], usage=None)
    client = FakeClient([resp])
    ext = AnthropicExtractor(Config(), client=client)
    with pytest.raises(ExtractionError):
        ext.extract(_raw("x"), "", Schema())


def test_get_extractor_defaults_to_heuristic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_extractor(Config()).name == "heuristic"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert get_extractor(Config(), force_heuristic=True).name == "heuristic"
