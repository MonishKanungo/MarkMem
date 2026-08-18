"""MarkMem + FastAPI — production memory-augmented REST chatbot.

Install:
    pip install markmem fastapi uvicorn openai

Run:
    uvicorn examples.fastapi_app:app --reload

Endpoints:
    POST /chat          — send a message, get a reply with memory
    GET  /memory/{uid}  — inspect a user's memory
    DELETE /memory/{uid} — GDPR erasure
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from markmem import Memory

# Shared memory instance — created once at startup
_memory: Optional[Memory] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory
    _memory = Memory(repo_path="./chat-memory", start_worker=True)
    yield
    _memory.close()


app = FastAPI(title="Memory-augmented chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    memory_injected: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    import asyncio

    # Retrieve relevant memory
    context = await asyncio.to_thread(
        _memory.search, req.message,
        user_id=req.user_id, format="context"
    )

    # Call your LLM here — example with a stub
    reply = f"[stub reply with memory: {bool(context)}] " \
            f"I remember you said: {context[:100] if context else 'nothing yet'}"

    # Store the turn asynchronously (background worker compiles it)
    await asyncio.to_thread(
        _memory.add,
        f"user: {req.message}\nassistant: {reply}",
        user_id=req.user_id
    )

    return ChatResponse(reply=reply, memory_injected=context or "")


@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    import asyncio
    pages = await asyncio.to_thread(_memory.get_all, user_id=user_id)
    return {"user_id": user_id, "pages": len(pages), "data": pages}


@app.delete("/memory/{user_id}")
async def forget_user(user_id: str):
    import asyncio
    tombstone = await asyncio.to_thread(_memory.forget, user_id, mode="scrub")
    return tombstone


@app.get("/stats")
async def stats():
    import asyncio
    return await asyncio.to_thread(_memory.stats)
