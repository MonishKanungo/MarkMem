import json

from strata import Memory
from strata.interop import (export_jsonl, export_mem0, export_memory_md,
                            import_jsonl, import_mem0)

from conftest import add_and_flush


def test_jsonl_roundtrip_lossless(mem, tmp_path):
    add_and_flush(mem, "I am vegetarian and prefer window seats", user_id="alice")
    add_and_flush(mem, "I prefer aisle seats now.", user_id="alice")
    out = tmp_path / "export.jsonl"
    n = export_jsonl(mem.repo, out)
    assert n == len(list(mem.repo.iter_pages()))

    m2 = Memory(repo_path=tmp_path / "fresh", start_worker=False)
    try:
        written = import_jsonl(m2.repo, out)
        assert len(written) == n
        page, _ = m2.repo.read_page("u/alice/user/profile")
        orig, _ = mem.repo.read_page("u/alice/user/profile")
        assert [c.model_dump() for c in page.claims] == [c.model_dump() for c in orig.claims]
        m2.reindex()
        assert m2.search("aisle seats", user_id="alice")
    finally:
        m2.close()


def test_mem0_import_caps_trust(mem, tmp_path):
    mem0_export = {"results": [
        {"id": "m1", "memory": "Loves skydiving", "user_id": "alice",
         "created_at": "2026-03-01T10:00:00Z", "metadata": {}},
        {"id": "m2", "memory": "Prefers dark mode", "user_id": "bob"},
        {"id": "m3", "memory": "", "user_id": "bob"},                 # empty: skipped
    ]}
    path = tmp_path / "mem0.json"
    path.write_text(json.dumps(mem0_export), encoding="utf-8")
    written = import_mem0(mem.repo, mem.schema, path)
    assert len(written) == 2
    page, _ = mem.repo.read_page("u/alice/user/profile")
    c = page.claims[0]
    assert c.provenance.value == "imported"
    assert c.confidence == 0.6                          # trust ceiling (§4.5)
    assert c.valid_from == "2026-03-01"
    assert c.sources == ["import:mem0:m1"]


def test_mem0_export_flattens_claims(mem, tmp_path):
    add_and_flush(mem, "I am vegetarian", user_id="alice")
    out = tmp_path / "mem0-out.json"
    n = export_mem0(mem.repo, out)
    data = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert n == len(data) and n >= 1
    rec = next(r for r in data if "vegetarian" in r["memory"] and "provenance" in r["metadata"])
    assert rec["user_id"] == "alice"
    assert rec["metadata"]["provenance"] == "user_stated"


def test_memory_md_export(mem, tmp_path):
    add_and_flush(mem, "I am vegetarian and I work at Acme", user_id="alice")
    out = tmp_path / "claude-mem"
    n = export_memory_md(mem.repo, out, user_id="alice")
    assert n >= 1
    index = (out / "MEMORY.md").read_text(encoding="utf-8")
    assert index.startswith("# Memory Index")
    files = list(out.glob("user-*.md"))
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert "vegetarian" in content and content.startswith("---")
