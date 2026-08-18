"""MarkMem — git-native memory layer for any chatbot.

Mem0-shaped API over plain markdown + git: the files ARE the memory.
"""
from .memory import AsyncMemory, Memory
from .models import FORMAT_VERSION

__version__ = "0.4.0"
__all__ = ["Memory", "AsyncMemory", "FORMAT_VERSION", "__version__"]
