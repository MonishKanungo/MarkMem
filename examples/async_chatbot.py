"""Strata AsyncMemory — for async frameworks (FastAPI, aiohttp, etc.)

Install:
    pip install strata-memory

Run:
    python examples/async_chatbot.py
"""
import asyncio
from strata import AsyncMemory


async def main():
    async with AsyncMemory(repo_path="./chat-memory-async") as memory:
        # add() is non-blocking — returns in milliseconds
        await memory.add("I prefer dark roast coffee", user_id="carol")
        await memory.add("I work remotely from Lisbon", user_id="carol")

        # flush() compiles everything synchronously
        await memory.flush()

        # search returns packed context string or list of hits
        context = await memory.search(
            "carol preferences", user_id="carol", format="context"
        )
        print("Packed context for system prompt:")
        print(context)

        # Point-in-time query
        hits = await memory.search(
            "carol location", user_id="carol",
            as_of="2025-01-01"  # what did we know on this date?
        )
        print(f"\nHistorical query returned {len(hits)} hits")

        # Stats
        stats = await memory.stats()
        print(f"\nPages: {stats['pages_by_type']}")
        print(f"Vector search: {stats['vector_search']}")


if __name__ == "__main__":
    asyncio.run(main())
