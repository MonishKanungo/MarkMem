from pathlib import Path

import pytest

from markmem.schema import SchemaError, load_schema, parse_schema

TEMPLATE = Path(__file__).parent.parent / "markmem" / "templates" / "schema.md"


def test_template_parses():
    schema = load_schema(TEMPLATE)
    assert set(schema.page_types) == {"user", "session", "concept", "entity", "project"}
    assert schema.decay_for("user").half_life_days == 365
    assert schema.decay_for("user").archive_below_confidence is None
    assert schema.decay_for("session").half_life_days == 14
    assert schema.retain_days_for("session") == 365
    assert schema.retain_days_for("user") is None
    assert schema.ceiling_for("imported") == 0.6
    assert schema.ceiling_for("user_stated") == 1.0


def test_unknown_type_gets_defaults():
    schema = parse_schema("x\n```yaml\npage_types:\n  note: {decay: weird}\n```\n")
    # unknown decay class falls back to DecayRule defaults
    assert schema.decay_for("note").half_life_days == 90
    assert schema.decay_for("never-declared").half_life_days == 90


def test_missing_fence_raises():
    with pytest.raises(SchemaError):
        parse_schema("# Schema\nno yaml here\n")


def test_invalid_yaml_raises():
    with pytest.raises(SchemaError):
        parse_schema("```yaml\npage_types: [unclosed\n```")


def test_non_mapping_raises():
    with pytest.raises(SchemaError):
        parse_schema("```yaml\n- just\n- a list\n```")


def test_last_fence_wins():
    text = "```yaml\npage_types: {a: {decay: fast}}\n```\nmore\n" \
           "```yaml\npage_types: {b: {decay: slow}}\ndecay_rules:\n  slow: {half_life_days: 100}\n```"
    schema = parse_schema(text)
    assert "b" in schema.page_types and "a" not in schema.page_types
    assert schema.decay_for("b").half_life_days == 100
