import os

import pytest

from markmem import Memory


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Tests always run offline against the heuristic extractor with default
    config. This must clear every provider/override variable: chatbot.llm's
    load_env() pulls the project .env (which may hold real Azure/NVIDIA keys
    and MARKMEM_* overrides) into os.environ mid-session, and without this
    guard later tests silently flip to live LLM extraction."""
    for var in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for var in [v for v in os.environ if v.startswith("MARKMEM_")]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def mem(tmp_path):
    m = Memory(repo_path=tmp_path / "mem", start_worker=False)
    yield m
    m.close()


@pytest.fixture()
def mem_factory(tmp_path):
    """Create Memory instances on demand (same or different repos)."""
    created = []

    def factory(name="mem", **kwargs):
        m = Memory(repo_path=tmp_path / name, start_worker=kwargs.pop("start_worker", False),
                   **kwargs)
        created.append(m)
        return m

    yield factory
    for m in created:
        m.close()


def add_and_flush(m: Memory, text: str, **kwargs) -> dict:
    result = m.add(text, **kwargs)
    m.flush()
    return result
