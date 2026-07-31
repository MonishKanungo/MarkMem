"""Nemotron client for the demo chatbot.

NVIDIA serves Nemotron through an OpenAI-compatible endpoint, so the standard
`openai` SDK is the whole integration. All config comes from .env / the
environment (see .env at the project root):

    NVIDIA_API_KEY      — required for real replies
    NEMOTRON_MODEL      — default nvidia/llama-3.1-nemotron-70b-instruct
    NEMOTRON_BASE_URL   — default https://integrate.api.nvidia.com/v1

No key → `available` is False and the app runs in a clearly-labelled no-LLM
demo mode (memory still works end-to-end; only reply generation is stubbed).
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the project-root .env (falls back to dotenv's own search)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()


class NemotronClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None):
        load_env()
        self.api_key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
        self.model = model or os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("NEMOTRON_BASE_URL", DEFAULT_BASE_URL)
        
        # Fall back to Azure OpenAI if NVIDIA key is missing but Azure OpenAI key is present
        if not self.api_key and os.environ.get("AZURE_OPENAI_API_KEY"):
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            self.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            self.model = model or os.environ.get("STRATA_LLM_COMPILE_MODEL", "gpt-4o-mini")
            
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, system: str, messages: list[dict], temperature: float = 0.6,
             max_tokens: int = 1024) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        if not self.available:
            raise RuntimeError("NVIDIA_API_KEY is not set — fill it in .env")
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        # Nemotron reasoning variants can emit a <think>…</think> block first;
        # strip it so only the answer reaches the chat / the graders.
        if "<think>" in text:
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
