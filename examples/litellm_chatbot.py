"""Use ANY LLM for memory extraction, via LiteLLM.

Strata's extraction step (raw conversation -> structured claims) can run on any
of LiteLLM's 100+ providers. Retrieval, the claim ledger, and storage are
unchanged — only the compile step swaps out.

    pip install "strata-memory[litellm]"

Pick a provider by setting two env vars, then run this file:

    # OpenAI
    set OPENAI_API_KEY=sk-...
    set STRATA_LLM_COMPILE_MODEL=gpt-4o-mini

    # Groq — fastest, generous free tier
    set GROQ_API_KEY=gsk_...
    set STRATA_LLM_COMPILE_MODEL=groq/llama-3.1-8b-instant

    # Google Gemini
    set GEMINI_API_KEY=...
    set STRATA_LLM_COMPILE_MODEL=gemini/gemini-1.5-flash

    # Ollama — fully local, no API key
    ollama pull llama3.1
    set STRATA_LLM_COMPILE_MODEL=ollama/llama3.1
    set STRATA_LLM_LITELLM_API_BASE=http://localhost:11434

    python examples/litellm_chatbot.py
"""
from __future__ import annotations

import os

from strata import Memory
from strata.config import Config, LLMConfig


def build_memory(repo: str = "./chat-memory-litellm") -> Memory:
    """A Memory whose extractor is LiteLLM, configured from the environment."""
    cfg = Config(llm=LLMConfig(
        provider="litellm",
        compile_model=os.environ.get("STRATA_LLM_COMPILE_MODEL", "gpt-4o-mini"),
        litellm_api_base=os.environ.get("STRATA_LLM_LITELLM_API_BASE"),
    ))
    return Memory(repo_path=repo, config=cfg, start_worker=False)


def main() -> None:
    memory = build_memory()
    print(f"extractor: {memory.pipeline.extractor.name}")
    print(f"model    : {os.environ.get('STRATA_LLM_COMPILE_MODEL', 'gpt-4o-mini')}\n")

    if memory.pipeline.extractor.name != "litellm":
        print("LiteLLM extractor did not activate. Check:")
        print('  pip install "strata-memory[litellm]"')
        print("  and that your provider API key is set.")
        memory.close()
        return

    # Two statements where the second CONTRADICTS the first — this is what the
    # claim ledger exists for. A good extractor reuses the subject key so the
    # old fact is superseded rather than duplicated.
    memory.add("I'm Alice. I work at NVIDIA and I prefer window seats.",
               user_id="alice")
    memory.flush()
    memory.add("Update: I moved to Google, and I prefer aisle seats now.",
               user_id="alice")
    memory.flush()

    profile = memory.get("u/alice/user/profile")
    if profile:
        print("Claim ledger:")
        for c in profile["claims"]:
            state = "current" if not c.get("valid_until") else f"closed {c['valid_until']}"
            print(f"  [{state:>14}] {c.get('subject') or '-':<20} {c['text']}")

    print("\nPacked context (what your prompt would receive):")
    print(memory.search("where does alice work and which seat",
                        user_id="alice", format="context") or "(empty)")
    memory.close()


if __name__ == "__main__":
    main()
