"""MCP server — secondary interface for Claude Code/Desktop (§10 of the spec,
extended per Phase 1 with supersede/history/review tools).

Run: `strata serve --repo <path>` or `python -m strata.mcp_server` with
STRATA_REPO set. Requires the [mcp] extra; import stays lazy so the core
library never depends on it.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def build_server(repo_path: str):
    from mcp.server.fastmcp import FastMCP

    from .memory import Memory
    from .util import today_iso

    memory = Memory(repo_path=repo_path, start_worker=True)
    mcp = FastMCP("strata")

    @mcp.tool()
    def wiki_search(query: str, type: Optional[str] = None, user_id: Optional[str] = None,
                    top_k: int = 5, as_of: Optional[str] = None) -> str:
        """Search memory pages (BM25 + claim ledger). as_of=YYYY-MM-DD for point-in-time."""
        hits = memory.search(query, user_id=user_id, type=type, top_k=top_k, as_of=as_of)
        return json.dumps(hits, ensure_ascii=False, indent=2)

    @mcp.tool()
    def wiki_read(id: str) -> str:
        """Read one page: full frontmatter (incl. claim ledger) + body."""
        page = memory.get(id)
        return json.dumps(page, ensure_ascii=False, indent=2) if page else f"no such page: {id}"

    @mcp.tool()
    def wiki_list(type: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """List pages — same content as index.md."""
        return json.dumps(memory.get_all(user_id=user_id, type=type), ensure_ascii=False, indent=2)

    @mcp.tool()
    def wiki_ingest(content: str, source_type: str = "conversation", origin: str = "",
                    user_id: Optional[str] = None, wait: bool = True) -> str:
        """Ingest raw content; wait=True compiles synchronously."""
        result = memory.add(content, user_id=user_id, source_type=source_type, origin=origin)
        if wait and result.get("status") == "queued":
            memory.flush()
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def wiki_supersede(page_id: str, old_claim_id: str, new_text: str) -> str:
        """Close an existing claim and record the replacing fact (human-directed)."""
        from .models import Claim, Provenance
        from .util import new_claim_id, utcnow_iso
        parsed = memory.repo.read_page(page_id)
        if parsed is None:
            return f"no such page: {page_id}"
        page, body = parsed
        old = page.get_claim(old_claim_id)
        if old is None:
            return f"no such claim: {old_claim_id}"
        old.valid_until = today_iso()
        page.claims.append(Claim(
            id=new_claim_id(new_text), text=new_text, subject=old.subject,
            valid_from=today_iso(), recorded_at=utcnow_iso(), confidence=1.0,
            provenance=Provenance.human_edited, sources=["mcp:wiki_supersede"],
            supersedes=old_claim_id))
        page.updated = utcnow_iso()
        memory.repo.write_page(page, body)
        memory.repo.regenerate_index()
        memory.git.commit_all(f"strata: supersede {old_claim_id} on {page_id}")
        memory.indexer.index_pages([page_id], memory.repo)
        return f"superseded {old_claim_id}"

    @mcp.tool()
    def wiki_history(page_id: str, include_diff: bool = False) -> str:
        """Audit trail of one page (git log --follow)."""
        return json.dumps(memory.history(page_id, include_diff=include_diff),
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def wiki_review(item_id: Optional[str] = None, decision: Optional[str] = None) -> str:
        """List quarantined memory ops, or decide one (decision=accept|reject)."""
        if item_id and decision == "accept":
            page_id = memory.pipeline.review_accept(item_id)
            return f"applied -> {page_id}" if page_id else f"no such item: {item_id}"
        if item_id and decision == "reject":
            return "rejected" if memory.pipeline.review_reject(item_id) else f"no such item: {item_id}"
        return json.dumps(memory.pipeline.review_queue.list(), ensure_ascii=False, indent=2)

    return mcp


def main(repo_path: Optional[str] = None) -> None:
    repo_path = repo_path or os.environ.get("STRATA_REPO", ".")
    try:
        server = build_server(repo_path)
    except ImportError as e:
        raise SystemExit(
            f"MCP server needs the [mcp] extra: pip install strata-memory[mcp] ({e})")
    server.run()


if __name__ == "__main__":
    main()
