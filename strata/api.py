"""FastAPI REST layer — Phase 2 HTTP interface for Strata (§12).

Exposes the full Memory API as a JSON REST service. Designed for:
- Multi-user deployments where the memory repo is on a server
- Language-agnostic clients (not just Python)
- Webhook / microservice integration patterns

Run:
    pip install strata-memory[api]
    strata api --repo ./my-memory --host 0.0.0.0 --port 8000

Or programmatically:
    from strata.api import build_app
    app = build_app("./my-memory")

All endpoints mirror the Memory Python API exactly so the REST docs
are the Python docs. OpenAPI schema auto-generated at /docs.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Request / response models (Pydantic, separate from core models so the API
# layer has no obligation to expose internal storage details)
# ---------------------------------------------------------------------------

def _pydantic():
    from pydantic import BaseModel, Field
    return BaseModel, Field


try:
    from pydantic import BaseModel, Field as PydanticField

    class AddRequest(BaseModel):
        messages: str | list[dict]
        user_id: Optional[str] = None
        agent_id: Optional[str] = None
        run_id: Optional[str] = None
        metadata: Optional[dict] = None
        source_type: str = "conversation"
        origin: str = ""

    class SearchRequest(BaseModel):
        query: str
        user_id: Optional[str] = None
        agent_id: Optional[str] = None
        run_id: Optional[str] = None
        top_k: int = 5
        type: Optional[str] = None
        as_of: Optional[str] = None
        include_superseded: bool = False
        format: Optional[str] = None

    class UpdateRequest(BaseModel):
        data: str

    class ForgetRequest(BaseModel):
        mode: str = "scrub"

    class MergeUsersRequest(BaseModel):
        into_user: str

    class MaintenanceRequest(BaseModel):
        decay: bool = True
        consolidate: bool = True
        retention: bool = True

except ImportError:
    pass  # FastAPI import will also fail; build_app() handles it cleanly


def build_app(repo_path: str | Path, start_worker: bool = True):
    """Build and return the FastAPI application.

    Args:
        repo_path: Path to the Strata memory repo (created if absent).
        start_worker: Start the background write worker (default True).

    Returns:
        FastAPI application instance.

    Raises:
        ImportError: If fastapi or uvicorn are not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError(
            "FastAPI REST layer needs the [api] extra: "
            "pip install strata-memory[api]"
        ) from exc

    from .memory import Memory

    # Shared Memory instance — created once at startup, shared across requests.
    # FastAPI's lifespan context manager handles clean teardown.
    _state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _state["memory"] = Memory(
            repo_path=repo_path, start_worker=start_worker
        )
        yield
        _state["memory"].close()

    app = FastAPI(
        title="Strata Memory API",
        description=(
            "Git-native memory layer REST interface. "
            "Mem0-shaped API over plain markdown + git with a bi-temporal claim ledger."
        ),
        version="0.3.1",
        lifespan=lifespan,
    )

    def mem() -> Memory:
        return _state["memory"]

    # ------------------------------------------------------------------
    # Write endpoints
    # ------------------------------------------------------------------

    @app.post("/memories", summary="Add / ingest content")
    async def add(req: AddRequest) -> dict:
        """Append raw content and enqueue for compilation.
        Returns in milliseconds. Use POST /flush to force synchronous compile."""
        return mem().add(
            req.messages,
            user_id=req.user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            metadata=req.metadata,
            source_type=req.source_type,
            origin=req.origin,
        )

    @app.post("/memories/flush", summary="Flush the write queue")
    async def flush(timeout_s: float = Query(default=300.0)) -> dict:
        """Synchronously compile everything in the queue.
        Use after add() when you need read-your-writes consistency."""
        n = mem().flush(timeout_s=timeout_s)
        return {"compiled": n}

    @app.put("/memories/{memory_id}", summary="Human-correct a memory")
    async def update(memory_id: str, req: UpdateRequest) -> dict:
        """Record a human_edited correction. Adds a new claim at full confidence
        and updates the page summary. Committed immediately."""
        try:
            return mem().update(memory_id, req.data)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such memory: {memory_id}")

    @app.delete("/memories/{memory_id}", summary="Delete / archive a memory")
    async def delete(
        memory_id: str,
        hard: bool = Query(default=False, description="True = hard delete from filesystem"),
    ) -> dict:
        """Soft delete (default): archives the page. Hard delete: removes file and index entry."""
        ok = mem().delete(memory_id, hard=hard)
        if not ok:
            raise HTTPException(status_code=404, detail=f"no such memory: {memory_id}")
        return {"deleted": memory_id, "hard": hard}

    @app.delete("/users/{user_id}/memories", summary="Delete all memories for a user")
    async def delete_all(
        user_id: str,
        hard: bool = Query(default=False),
    ) -> dict:
        n = mem().delete_all(user_id=user_id, hard=hard)
        return {"user_id": user_id, "deleted": n, "hard": hard}

    # ------------------------------------------------------------------
    # Read endpoints
    # ------------------------------------------------------------------

    @app.post("/memories/search", summary="Search memory")
    async def search(req: SearchRequest) -> list[dict]:
        """Tiered search: L0 standing context → L1 FTS5 BM25 → L2 vectors (if installed).
        Pass format='context' to get a packed string ready for a system prompt."""
        return mem().search(
            req.query,
            user_id=req.user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            top_k=req.top_k,
            type=req.type,
            as_of=req.as_of,
            include_superseded=req.include_superseded,
            format=req.format,
        )

    @app.get("/memories/{memory_id:path}", summary="Read one page")
    async def get(memory_id: str) -> dict:
        """Return the full page dict: frontmatter, claim ledger, and body.
        memory_id may contain slashes (e.g. u/alice/user/profile)."""
        page = mem().get(memory_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"no such memory: {memory_id}")
        return page

    @app.get("/memories", summary="List all pages")
    async def get_all(
        user_id: Optional[str] = Query(default=None),
        type: Optional[str] = Query(default=None),
        agent_id: Optional[str] = Query(default=None),
        run_id: Optional[str] = Query(default=None),
        include_archived: bool = Query(default=False),
    ) -> list[dict]:
        """List all compiled pages with optional filters."""
        return mem().get_all(
            user_id=user_id, type=type, agent_id=agent_id,
            run_id=run_id, include_archived=include_archived,
        )

    @app.get("/memories/{memory_id}/history", summary="Git history of a page")
    async def history(
        memory_id: str,
        include_diff: bool = Query(default=False),
    ) -> list[dict]:
        """Return git log --follow for the page. Full audit trail."""
        return mem().history(memory_id, include_diff=include_diff)

    # ------------------------------------------------------------------
    # Governance endpoints
    # ------------------------------------------------------------------

    @app.post("/users/{user_id}/forget", summary="Compliance erasure")
    async def forget(user_id: str, req: ForgetRequest) -> dict:
        """Erase all memory for a user.
        mode=scrub: deletes from working tree (git history kept for audit).
        mode=rewrite: additionally purges all git history (needs git-filter-repo)."""
        return mem().forget(user_id, mode=req.mode)

    @app.post("/users/{user_id}/merge", summary="Merge user identity")
    async def merge_users(user_id: str, req: MergeUsersRequest) -> dict:
        """Move all pages from user_id into into_user. Records alias."""
        moved = mem().merge_users(user_id, req.into_user)
        return {"from_user": user_id, "into_user": req.into_user, "pages_moved": moved}

    @app.post("/review/{item_id}/accept", summary="Accept a quarantined write")
    async def review_accept(item_id: str) -> dict:
        page_id = mem().pipeline.review_accept(item_id)
        if page_id is None:
            raise HTTPException(status_code=404, detail=f"no such review item: {item_id}")
        return {"accepted": item_id, "page_id": page_id}

    @app.post("/review/{item_id}/reject", summary="Reject a quarantined write")
    async def review_reject(item_id: str) -> dict:
        ok = mem().pipeline.review_reject(item_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"no such review item: {item_id}")
        return {"rejected": item_id}

    @app.get("/review", summary="List quarantined write ops")
    async def review_list() -> list[dict]:
        return mem().pipeline.review_queue.list()

    # ------------------------------------------------------------------
    # Lifecycle & utility
    # ------------------------------------------------------------------

    @app.post("/maintenance", summary="Run lifecycle sweeps")
    async def maintenance(req: MaintenanceRequest) -> dict:
        """Run decay, consolidation, and retention sweeps. Idempotent; one commit."""
        return mem().maintenance(
            decay=req.decay, consolidate=req.consolidate, retention=req.retention
        )

    @app.post("/reindex", summary="Rebuild the SQLite index from markdown")
    async def reindex() -> dict:
        n = mem().reindex()
        return {"reindexed": n}

    @app.post("/reset", summary="Wipe all memory (keep git history)")
    async def reset() -> dict:
        mem().reset()
        return {"status": "reset"}

    @app.get("/stats", summary="Repo statistics")
    async def stats() -> dict:
        return mem().stats()

    @app.get("/lint", summary="Memory hygiene check")
    async def lint() -> list[dict]:
        findings = mem().lint()
        return [{"page_id": f.page_id, "check": f.check, "detail": f.detail}
                for f in findings]

    @app.get("/health", summary="Health check")
    async def health() -> dict:
        return {"status": "ok", "repo": str(mem().repo.root)}

    return app


def main(repo_path: Optional[str] = None, host: str = "0.0.0.0",
         port: int = 8000, reload: bool = False) -> None:
    """Entry point for `strata api` CLI command."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "FastAPI REST layer needs the [api] extra: pip install strata-memory[api]"
        )
    repo = repo_path or os.environ.get("STRATA_REPO", ".")
    app = build_app(repo)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
