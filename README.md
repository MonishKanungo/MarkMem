# Strata

**A memory layer for chatbots that stores memory as plain markdown in a git repo.**

Your chatbot's memory is a folder you can `cat`, `grep`, `git diff`, and delete —
not rows in a vector database you can't inspect. SQLite is a rebuildable cache
on top; delete it and `strata reindex` restores it from the markdown.

```python
from strata import Memory

m = Memory(repo_path="./chat-memory")

m.add("I'm vegetarian and prefer window seats", user_id="alice")
m.add("Actually I prefer aisle seats now", user_id="alice")
m.flush()

print(m.search("alice seating", user_id="alice", format="context"))
# ### Memory (cite ids when you rely on these)
# [u/alice/user/profile | user | conf 0.85 | updated 2026-07-31]
# - (user_stated, 0.85) I am vegetarian
# - (user_stated, 0.85) I prefer aisle seats
```

The window-seat fact is not deleted — it is closed with `valid_until` and kept in
the ledger, so `search(..., as_of="2026-06-01")` still returns it. It just never
reaches the prompt as a current fact.

---

## The problem

Chatbots forget. The usual fix — write memories into a vector store — trades one
problem for four:

| | Typical vector-store memory | Strata |
|---|---|---|
| Can you read your own memory? | No, it's embeddings in a DB | It's markdown on disk |
| Can you see what changed and why? | No | `git log` / `git diff` |
| What happens when a fact changes? | Overwritten, or both versions retrieved | Old claim closed, new one supersedes it |
| Can you prove a user was erased? | Delete rows and hope | Path-scoped delete, history rewrite, or crypto-shred |
| Where do inferred facts rank vs stated ones? | Same | Provenance-weighted; `user_stated` outranks `agent_inferred` |
| Can a poisoned message enter memory silently? | Usually yes | Injection-shaped writes are quarantined for review |

## Install

```bash
pip install strata-memory                  # core: markdown + git + BM25. no external DB
pip install "strata-memory[vector]"        # + semantic search (recommended)
pip install "strata-memory[all]"           # everything below
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

Nothing degrades hard: no vector extra → lexical BM25 only; no LLM key → the
deterministic heuristic extractor. The core never imports torch.

---

## Benchmarks

Full detail, caveats and reproduction commands are in the benchmarks section below.

Strata ships **StrataBench**, its own hand-authored suite. It exists because
public benchmarks label the gold *answer string* but not the gold *evidence
page* — which makes true R@k uncomputable and forces everyone to report
substring "answer presence" proxies. StrataBench labels the evidence, so R@1 /
R@5 / MRR are real retrieval metrics.

10 sessions · 3 users · 18 questions · 10 capability categories · deterministic offline

| Metric | Strata | Reference |
|---|---|---|
| **R@1** | **0.875** | — |
| **R@5** | **1.000** | Mem0 0.952 · Khoj 0.832 · Letta 0.685 |
| **MRR** | **0.938** | — |
| Precision@5 | 0.487 | ceiling ≈0.4 at `top_k=5` |
| answer-in-context | 1.000 | — |
| stale-leak (asserted current) | **0.000** | — |
| abstention rate | 1.000 | — |
| EM / F1 (LLM-graded) | 0.778 / 0.843 | ±0.17 judge variance |
| retrieval latency p50 | 14.4 ms | — |

> **R@5 = 1.000 is on a 10-session corpus and is not comparable to Mem0's 0.952
> on LoCoMo.** Theirs is a much larger, harder dataset. What this shows is that
> the retrieval stack has no systematic failure across the ten capabilities
> tested — not that Strata out-retrieves Mem0. It will fall as corpora grow.

Other suites (`python run_benchmarks.py --pages 100 --with-llm`):

| | Strata | Reference / ideal |
|---|---|---|
| `search()` p50 | **1.6 ms** | Mem0 ~8 ms |
| `search(format="context")` p50 | **2.6 ms** | Mem0 ~11 ms |
| Temporal hit@5 (supersession) | **1.000** | Hippo 0.944 |
| Temporal current-fact / stale-leak | **1.000 / 0.000** | not measured by others |
| Multi-user isolation | **100%**, 0 contamination | — |
| GDPR erasure + crypto-shred | complete, 265 ms | — |
| PII detection (Presidio) | 60% | — |
| Injection quarantine | 2/4 caught | 4/4 ideal |

### Three results we are not going to dress up

**1. A small LLM extractor is *worse* than the deterministic heuristic.** We
expected the opposite:

| | Heuristic | LLM (llama-3.1-8b) |
|---|---|---|
| R@5 | **1.000** | 0.750 |
| F1 | **0.843** | 0.583 |
| stale-leak | **0.000** | 0.500 |
| failing cases | **0/18** | 8/18 |

The 8B model drops facts entirely, mis-routes claims onto session pages, and
mints a fresh `subject` key for a contradicting fact instead of reusing the
existing one — which breaks supersession, so both the old and new value stay
active. Supersession depends on stable subject keys. Use the heuristic default,
or a GPT-4o / Claude Sonnet class model. **We have not measured a frontier model
on this suite**, so no claim is made for that configuration.

**2. `add()` latency is not published.** The benchmark harness calls `flush()`
inline, so its reading measures enqueue *plus* synchronous extraction and a git
commit. That is a harness bug, not a latency figure. Search and pack numbers are
measured cleanly. Fixing the harness is tracked work.

**3. HaluMem scores stale-leak 1.000 where StrataBench scores 0.000.** They
disagree by design: HaluMem counts the stale string anywhere in context,
StrataBench excludes dated episodic blocks. Against HaluMem's stricter reading
Strata fails, because the session page retains the original wording. Both are
reported rather than picking the flattering one.

Also note: the token "compression ratio" the runner prints (0.43×) means packed
context is *larger* than 20 raw one-line facts — per-block provenance metadata
dominates at that size. The token *budget* is the real guarantee, not
compression. No 70× claim is made.

---

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
                │                      → claims with a normalised subject key
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

On disk:

```
chat-memory/
├── schema.md              page types, decay classes, retention  (you edit this)
├── config.yaml            search / PII / review knobs
├── index.md               auto-generated contents
├── wiki/                  ← THE SOURCE OF TRUTH
│   ├── g/concept/*.md         shared knowledge
│   └── u/alice/
│       ├── user/profile.md    the claim ledger
│       └── session/*.md       episodic records
├── raw/u/alice/...        immutable original inputs
└── .strata/               ← DERIVED. delete it, run `strata reindex`, it's back
    ├── index.db               FTS5 + claims + vectors
    ├── queue.db               durable write queue
    ├── review/                quarantined writes awaiting decision
    └── ops.jsonl              erasure tombstones, review decisions
```

A user's entire footprint lives under exactly two prefixes — `wiki/u/<id>/` and
`raw/u/<id>/` — which is what makes path-scoped erasure provable rather than
best-effort.

### The claim ledger

Every fact is a claim, not a row to overwrite:

```yaml
claims:
  - id: c-2026-07-31-a1b2
    text: I prefer aisle seats
    subject: preference:seat        # ← stable key; how contradictions are matched
    valid_from: 2026-07-31          # event time: when it became true
    valid_until: null               # still current
    recorded_at: 2026-07-31T14:30:00  # record time: when we learned it
    provenance: user_stated
    confidence: 0.95
    supersedes: c-2026-06-01-9f3e   # the claim this replaced
```

Two timelines (event time and record time) are what make `as_of` queries and
audit both work. Contradictions close the old claim rather than deleting it.

---

## Use any LLM

Memory extraction runs on any LiteLLM-supported provider. Retrieval, the ledger
and storage are unchanged — only the compile step swaps.

```bash
pip install "strata-memory[litellm]"
```

```yaml
# config.yaml in your memory repo
llm:
  provider: litellm
  compile_model: gpt-4o-mini
  litellm_api_base: http://localhost:11434   # only for Ollama / proxies
```

Or entirely by environment:

```bash
STRATA_LLM_PROVIDER=litellm
STRATA_LLM_COMPILE_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=gsk_...
```

| Provider | `compile_model` | Key |
|---|---|---|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Groq | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| Gemini | `gemini/gemini-1.5-flash` | `GEMINI_API_KEY` |
| Ollama (local) | `ollama/llama3.1` | none |
| NVIDIA NIM | `nvidia_nim/meta/llama-3.1-8b-instruct` | `NVIDIA_NIM_API_KEY` |
| Azure | `azure/<deployment>` | `AZURE_API_KEY` + `AZURE_API_BASE` |
| Bedrock | `bedrock/anthropic.claude-3-haiku-...` | AWS credentials |

Models with function-calling get forced tool-use; others fall back to JSON mode.
Either way the output is validated against a pydantic schema, retried once, then
dead-lettered to `raw/failed/` rather than silently dropped.

Working example: [`examples/litellm_chatbot.py`](examples/litellm_chatbot.py)

> Read the benchmark note above before choosing a small model — an 8B extractor
> measured *worse* than the offline heuristic on our suite.

## API

```python
from strata import Memory, AsyncMemory
```

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
| `stats()` / `reset()` / `reindex()` | Plumbing |

`AsyncMemory` mirrors the whole surface with `await`.

Integration is three lines per turn:

```python
context = memory.search(user_msg, user_id=uid, format="context")
reply = your_llm(system_prompt + context, user_msg)
memory.add(f"user: {user_msg}\nassistant: {reply}", user_id=uid)
```

More: [`examples/`](examples/) — OpenAI, Anthropic, LiteLLM, async, FastAPI.

## Other interfaces

**CLI** — 19 commands:

```bash
strata init ./my-memory
strata ingest "I prefer aisle seats" --user alice
strata search "seating" --user alice --context
strata claim u/alice/user/profile              # inspect the ledger
strata review                                  # quarantined writes
strata forget alice --rewrite                  # provable erasure
strata sweep                                   # decay + consolidate + retention
strata lint                                    # memory hygiene
strata eval                                    # eval from your own supersessions
strata doctor                                  # environment check
strata serve                                   # MCP server
strata api --port 8000                         # REST server
```

**MCP** (Claude Code / Desktop) — `pip install "strata-memory[mcp]"`:

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": ["-m", "strata.mcp_server"],
      "env": { "STRATA_REPO": "./my-memory" }
    }
  }
}
```

Exposes `wiki_search`, `wiki_read`, `wiki_list`, `wiki_ingest`, `wiki_supersede`,
`wiki_history`, `wiki_review`.

**REST** — `pip install "strata-memory[api]"`, then `strata api`. OpenAPI docs at
`/docs`.

## Compliance

| Mode | What it does | Trade-off |
|---|---|---|
| `forget(user, "scrub")` | Deletes the user's two path prefixes, commits, tombstones | Content remains in git history — audit-friendly |
| `forget(user, "rewrite")` | Also purges all history via `git-filter-repo` | Provable; invalidates existing clones |
| crypto-shred | Deletes the per-user AES-256-GCM key | Instant, works even against backups; requires encryption enabled up front |

Every erasure writes a tombstone to `.strata/ops.jsonl`. The erasure commit
itself is the in-repo audit record. The GDPR-vs-audit tension is real, so both
modes are explicit rather than one being silently chosen for you.

## Portability

```bash
strata export --to jsonl   --out memory.jsonl    # lossless round-trip
strata export --to mem0    --out mem0.json
strata export --to memory-md --out ./MEMORY/     # Claude Code format
strata import --from mem0 mem0-export.json       # migrate in
```

Imported facts are capped at `confidence 0.6` by provenance ceiling — they never
masquerade as things the user told you.

## Limitations

1. **~50–100K pages per repo.** Many small files is git's and NTFS's worst case.
   Beyond that this is the wrong tool.
2. **Read-your-writes is eventual** for compiled pages. Raw text is searchable
   immediately; `flush()` forces compilation.
3. **The heuristic extractor is a floor.** First-person pattern matching. It
   scored well on our suite, but its coverage is narrow by construction.
4. **A small LLM extractor can be worse than the heuristic** — measured, see above.
5. **FTS5 stemming is English-biased.** Multilingual needs the vector extra.
6. **No graph search** (Mem0 has one) and no hosted service. Both are roadmap.
7. **`add()` latency is unmeasured** — the harness conflates enqueue with compile.

## Development

```bash
git clone <repo> && cd strata
pip install -e ".[all,dev]"
python -m pytest tests/ -q
python -m benchmarks.memory_evals.stratabench
```


## License

Apache-2.0
