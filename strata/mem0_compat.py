"""Backwards-compatible import path from spec v2.0 — the class moved to
``strata.Memory``; this alias keeps `from strata.mem0_compat import Memory` working."""
from .memory import AsyncMemory, Memory

__all__ = ["Memory", "AsyncMemory"]
