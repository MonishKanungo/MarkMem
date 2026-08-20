<div align="center">

# 🧠 MarkMem

**A memory layer for chatbots that stores memory as plain markdown in a git repo.**

[![PyPI version](https://badge.fury.io/py/markmem.svg)](https://badge.fury.io/py/markmem)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/markmem.svg)](https://pypi.org/project/markmem/)

[Installation](#installation) • [Quick Start](#quick-start) • [Key Features](#key-features) • [How it Works](#how-it-works) • [API](#api)

</div>

## What is MarkMem?

Chatbots forget. They cannot remember anything past the current session. The usual fix—writing memories into a vector database—trades one problem for four: you can't inspect your own memory, you can't see what changed or why, facts overwrite each other silently, and proving user data erasure is nearly impossible.

**MarkMem** is different. Your chatbot's memory is a folder you can `cat`, `grep`, `git diff`, and delete. SQLite is a rebuildable cache on top; delete it and `markmem reindex` restores it from the markdown.

## Why MarkMem? (The Ecosystem Gap)

The market is crowded with "Markdown + hybrid search" tools (like *memweave* and *basic-memory*), and massive vector/graph databases (like *Mem0* and *Graphiti*). 

MarkMem occupies a strictly unique gap. It is the **only** memory layer that combines these three pillars:

1. **Bi-Temporal Facts in YAML**: Old facts are never deleted or silently overwritten. If a user changes jobs from Google to Microsoft, the Google claim is closed (`valid_until`) and a successor pointer (`supersedes`) is created. This allows perfect temporal reasoning (`as_of` queries) instead of just relying on score decay.
2. **Presidio as a Write Gate**: Native integration with Microsoft Presidio blocks or masks sensitive PII (SSNs, emails, credit cards) *before* it ever hits the disk.
3. **Git as the Compliance Surface**: Your audit log isn't an opaque database table; it's `git log`. When GDPR erasure is required, MarkMem performs provable, path-scoped deletes and rewrites the git history.

*(It also includes everything you expect: BM25+Vector RRF fusion, 100+ LLMs via LiteLLM, injection quarantines, an MCP Server, and a REST API).*

## Installation

```bash
pip install markmem                  # core: markdown + git + BM25. no external DB
pip install "markmem[vector]"        # + semantic search (recommended)
pip install "markmem[all]"           # everything below
```

| Extra | Adds | For |
|---|---|---|
| `vector` | sqlite-vec, model2vec | Hybrid BM25 + semantic search (RRF fusion) |
| `litellm` | litellm | Compile memory with any of 100+ LLMs |
| `llm` | anthropic, openai | Native Claude / OpenAI extractors |
| `pii` | presidio | 30+ PII entity types instead of regex |
| `mcp` | mcp | MCP server for Claude Code / Desktop |
| `api` | fastapi, uvicorn | REST server |
| `crypto` | cryptography | AES-256-GCM crypto-shred erasure |

*Note: Nothing degrades hard. No vector extra → BM25 only. No LLM key → heuristic extractor. The core never imports torch.*

## Quick Start

```python
from markmem import Memory

m = Memory(repo_path="./chat-memory")

# 1. Add facts (compiles asynchronously)
m.add("I'm vegetarian and prefer window seats", user_id="alice")
m.add("Actually I prefer aisle seats now", user_id="alice")
m.flush()

# 2. Retrieve packed memory for your prompt
print(m.search("alice seating", user_id="alice", format="context"))
```

**Output:**
```markdown
### Memory (cite ids when you rely on these)
[u/alice/user/profile | user | conf 0.85 | updated 2026-07-31]
- (user_stated, 0.85) I am vegetarian
- (user_stated, 0.85) I prefer aisle seats
```
*(Notice how the aisle seat supersedes the window seat, while the vegetarian fact remains!)*

## How it works

```
add("I prefer aisle seats now", user_id="alice")
  │
  ├─ PII gate ............ Presidio/regex → tag | mask | block
  ├─ Injection guard ..... instruction-override patterns → quarantine
  ├─ raw/ append ......... immutable, timestamped, never rewritten
  └─ SQLite queue ........ add() returns; compile happens off the hot path
                                     │
        background worker ───────────┘
                │
                ├─ Extractor ......... heuristic (default) | any LLM
                ├─ Claim resolver .... same subject + different value?
                │                      → close old (valid_until), new supersedes
                ├─ Review gate ....... injection / low confidence → review queue
                ├─ Write page ........ markdown + YAML frontmatter
                └─ ONE git commit per batch

search("alice seating", format="context")
  │
  ├─ L0  standing context .. user profile + pinned pages
  ├─ L1  BM25 FTS5 ......... pages, chunks, claims, raw
  ├─ L2  vectors ........... model2vec + sqlite-vec  [optional]
  ├─ RRF fusion ............ rank-only, no score calibration
  ├─ Provenance weight ..... user_stated > tool_derived > agent_inferred
  ├─ Decay adjustment ...... per-type confidence half-life
  └─ Token-budgeted pack ... active claims only, every block cited
```

A user's entire footprint lives under exactly two prefixes — `wiki/u/<id>/` and `raw/u/<id>/` — which is what makes path-scoped erasure provable.

## Use any LLM

Memory extraction runs on any LiteLLM-supported provider. Retrieval, the ledger and storage are unchanged — only the compile step swaps.

```bash
pip install "markmem[litellm]"

MARKMEM_LLM_PROVIDER=litellm
MARKMEM_LLM_COMPILE_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=gsk_...
```

*See the `examples/` directory for OpenAI, Anthropic, LiteLLM, async, and FastAPI examples.*

## API

| Method | Notes |
|---|---|
| `add(messages, user_id, agent_id, run_id, metadata)` | PII-gated, enqueued; compiles off the hot path |
| `flush()` | Force synchronous compilation (tests, turn boundaries) |
| `search(query, user_id, top_k, as_of, format="context")` | Tiered L0→L1→L2; `format="context"` returns a packed prompt string |
| `get(id)` / `get_all(user_id, type)` | Full page dicts including the claim ledger |
| `update(id, text)` | Human correction → `human_edited` claim at full trust |
| `delete(id, hard=False)` | Soft archive, or hard delete |
| `forget(user_id, mode="scrub"\|"rewrite")` | Compliance erasure + tombstone |
| `history(id, include_diff=True)` | Literally `git log --follow` on the page |
| `maintenance()` | Decay, consolidation, retention sweeps |
| `lint()` | Broken links, unsourced claims, injection, ledger/prose drift |

*Note: `AsyncMemory` mirrors the whole surface with `await`.*

## Integrations

**MCP Server** (Claude Code / Desktop) — `pip install "markmem[mcp]"`:
```json
{
  "mcpServers": {
    "markmem": {
      "command": "python",
      "args": ["-m", "markmem.mcp_server"],
      "env": { "MARKMEM_REPO": "./my-memory" }
    }
  }
}
```
*Exposes `wiki_search`, `wiki_read`, `wiki_list`, `wiki_ingest`, `wiki_supersede`, `wiki_history`, `wiki_review`.*

**REST API** — `pip install "markmem[api]"` then run `markmem api --port 8000`.

**CLI** — Run `markmem --help` for 19 powerful memory management commands.

## Compliance & Erasure

Every erasure writes a tombstone to `.markmem/ops.jsonl` and commits to git.

| Mode | What it does | Trade-off |
|---|---|---|
| `forget(user, "scrub")` | Deletes the user's two path prefixes, commits, tombstones | Content remains in git history — audit-friendly |
| `forget(user, "rewrite")` | Also purges all history via `git-filter-repo` | Provable; invalidates existing clones |
| `crypto-shred` | Deletes the per-user AES-256-GCM key | Instant, works even against backups; requires encryption enabled up front |

## Portability

```bash
markmem export --to jsonl   --out memory.jsonl    # lossless round-trip
markmem export --to mem0    --out mem0.json       # migrate to mem0
markmem export --to memory-md --out ./MEMORY/     # Claude Code format
markmem import --from mem0 mem0-export.json       # migrate from mem0
```

## Benchmarks

MarkMem ships **MarkMemBench**, a hand-authored dataset where evidence is labeled for true R@1 / R@5 metrics. We also evaluate on industry-standard massive-scale datasets like LoCoMo and LongMemEval.

All benchmarks run on a single-pass extraction pipeline (no agentic loops).

| Benchmark / Metric | MarkMem (Full Run) | Mem0 (April 2026) | Letta / MemGPT | Khoj |
|---|---|---|---|---|
| **LoCoMo** (R@5 evidence recall) | **83.3%** | 92.5% | 68.5% | 83.2% |
| **LongMemEval (Small)** (R@5 evidence recall) | **100.0%** | 94.4% | — | — |
| **BEAM** (Retrieval Accuracy) | **100.0%** | 64.1% | — | — |
| **Search Latency** (p50) | **1.5 ms** | 880.0 ms | — | — |

*Note: A frontier model (GPT-4o or Claude 3.5 Sonnet) achieves perfect recall on extraction. Smaller 8B models drop facts and break supersession. Mem0 latency and tokens are from their April 2026 Memory Algorithm release.*

### Security & Edge Cases

MarkMem goes beyond standard retrieval benchmarks to explicitly test security, compliance, and edge cases that vector databases struggle with.

| Metric | MarkMem | Ideal |
|---|---|---|
| **Multi-User Isolation** (cross-contamination) | **100% isolated** (0 leaks) | 100% |
| **Temporal Reasoning** (supersession accuracy) | **100.0%** | 100% |
| **GDPR Erasure** (crypto-shred + tombstone) | **230.5 ms** | < 1s |
| **Context Packing Latency** (p50) | **2.0 ms** | < 10ms |

## Limitations

1. **~50–100K pages per repo.** Many small files is git's and NTFS's worst case.
2. **Read-your-writes is eventual** for compiled pages. Raw text is searchable immediately; `flush()` forces compilation.
3. **FTS5 stemming is English-biased.** Multilingual needs the vector extra.

## Development

```bash
git clone <repo> && cd markmem
pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

## License

Apache-2.0
