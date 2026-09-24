# MCP Server

HLM exposed over the Model Context Protocol, for agents that are not Hermes.

Three documents used to cover this — `MCP-QUICK-START.md`,
`docs/mcp-architecture.md` and `docs/mcp-web-search.md` — split by no principle
a reader could state, so answering most questions meant visiting all three. They
were merged into this one in `44e30dc`: how to run it, how it works, and the
one optional tool. If you need what they said —
`git show 44e30dc^:MCP-QUICK-START.md`,
`git show 44e30dc^:docs/mcp-architecture.md`,
`git show 44e30dc^:docs/mcp-web-search.md`.

- [Quick start](#quick-start) — install and run
- [Architecture](#architecture) — components, storage, threading, lifecycle
- [Web search](#web-search) — the optional SearXNG tool

---

## Quick start

Run HLM's memory store as an MCP server for any agent (Claude, Codex, custom scripts) — no Hermes required.

### Architecture

Single shared HTTP endpoint with profile-based isolation. One process, multiple profiles, shared vector index.

**Quick model:**
```
Agent A (Pi)      ──→  ┌──────────────┐  ──→ ~/.hermes/.../pi.db
Agent B (Hermes)  ──→  │ mcp_server.py │  ──→ ~/.hermes/.../profile-a.db
Agent C (Shared)  ──→  │ port 3801     │  ──→ ~/.hermes/.../shared.db
                        └──────────────┘
```

**See also:** [Architecture](#architecture) for full design, lifecycle, and security considerations.

### Prerequisites

- Python 3.11+
- Qdrant running (or use SQLite-only mode)
- Embedding backend (Ollama or FastEmbed)

### 1. Install

```bash
git clone https://github.com/alexeytz/hermes-layered-memory.git
cd hermes-layered-memory
pip install mcp qdrant-client
```

### 2. Start Qdrant

```bash
docker compose up -d
sleep 3
curl -s http://localhost:6333/   # verify: returns version info
```

### 3. Set up the environment

Create an `.env.mcp` inside a directory named local/ that you make yourself.
It is gitignored, so neither the directory nor the file is in a fresh clone.
Copy `.env.example` and edit for your setup:

```bash
# Base directory for all agent DBs
export HLM_MCP_PATH="/opt/memory-dbs"

# Embedding backend
export HLM_EMBED_URL="http://localhost:11434/api/embed"
export HLM_EMBED_MODEL="qwen3-embedding:8b"

# Qdrant (optional if running locally)
export HLM_QDRANT_URL="http://localhost:6333"
```

Then source it:

```bash
source local/.env.mcp
```

#### Alternative: Single-DB override

If you only need one DB (not per-agent), use `HLM_DB_PATH` instead:

```bash
export HLM_DB_PATH="/path/to/my-memory.db"
```

This overrides `HLM_MCP_PATH` entirely.

### 4. Connect Your Agent

#### Stdio transport (recommended — each agent spawns its own process)

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hermes-memory": {
      "command": "python3",
      "args": ["/path/to/hermes-layered-memory/mcp_server.py"],
      "env": {
        "HLM_MCP_PROFILE": "claude",
        "HLM_MCP_PATH": "/opt/memory-dbs",
        "HLM_EMBED_URL": "http://localhost:11434/api/embed",
        "HLM_EMBED_MODEL": "qwen3-embedding:8b",
        "HLM_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Other agents** — same pattern, just change `HLM_MCP_PROFILE`:

```json
{
  "env": {
    "HLM_MCP_PROFILE": "codex"
  }
}
```

#### Streamable HTTP transport (shared server on the network)

Start the server:

```bash
source local/.env.mcp
python3 mcp_server.py --transport streamable-http --port 3801
```

Connect from your agent:

```json
{
  "mcpServers": {
    "hermes-memory": {
      "url": "http://host:3801/mcp"
    }
  }
}
```

**Multi-agent profile targeting:** The shared HTTP server doesn't lock you to one profile.
Use the `profile` parameter on any tool to target a specific profile:

- `memory_query(query="...", profile="profile-a")` — search profile-a profile
- `memory_write(action="add", content="...", profile="pi")` — write to pi profile
- `memory_query(query="...", cross_profile=true)` — search all profiles
- `memory_list_profiles()` — discover available profiles

The HTTP server binds to `127.0.0.1` by default (localhost-only). Use `--host 0.0.0.0` to expose it on the network.

**Security note:** The server is unauthenticated. Secure access via SSH tunnel, VPN, or firewall rules — it's your responsibility.

#### Sharing memory between agents

Agents with the same `HLM_MCP_PROFILE` value share the same DB file:

```json
{ "env": { "HLM_MCP_PROFILE": "shared" } }
```

Both agents will read/write `{MCP_PATH}/shared.db`.

### 5. Available Tools

All tools accept an optional `profile` parameter to target a specific profile (defaults to the server's `HLM_MCP_PROFILE`).

The server exposes the same 43 actions as the Hermes plugin, grouped into 8 tools. `tests/test_mcp.py` (T400) reads the plugin's dispatch table and fails if any action is unreachable here, so the two surfaces cannot drift apart silently.

#### `memory_list_profiles` — Discover profiles

Scans the DB directory and returns all profile names, record counts, and status.

Each entry carries `db_file`, **not** `db_path` — the basename of the resolved
database file, not its absolute path. This server has no authentication, and
`_safe_error` exists so local filesystem paths never reach a caller; a success
path handing over the directory layout undid that. `get_status` had the same
thing removed on 2026-08-22; `list_profiles` kept full paths on the reasoning
that symlinked profile aliases are detected by them, and the 2026-09-15 review
(round 2, bundle04 F3) called that in.

The resolution still happens — it is just done before the basename is taken,
which is the part that matters and the part the obvious fix gets wrong. A
symlinked alias and its target both report the file they share, so a client can
still tell that two profile names are one database; `basename(db_path)` without
resolving would have made them look like two. `T424`.

#### `memory_query` — Search memories

Returns `{"results": [...]}`, plus `conflict_alert` and/or `low_relevance` when
present — matching the plugin's `_do_retrieve` shape, which hoists both out of
`layer3_flags` for the same reason: a signal the caller never sees is
indistinguishable from no signal. Before v0.7.57 this returned a bare JSON
array with neither hoisted; a conflict detected at the default `max_layer=2`
was written into a record's `layer3_flags` and never surfaced to an MCP
client. Supports filters: `scope`, `data_type`, `data_id`, `session_name`.
Pass `cross_profile=true` to search across all profiles.

#### `memory_write` — Unified write tool

Pass `action="add"` with `content`, `data_type`, `data_id`, etc. Also supports `action="update"` (requires `uuid`), `action="delete"` (requires `uuid`), `action="list"`, `action="status"`, `action="maintenance"`.

Auto-enriches metadata. Creates the profile DB on first write. Pass `ttl` (ISO timestamp) to auto-expire the memory.

**`add` returns two different shapes, and neither is `{"uuid", "status"}`.**
This paragraph claimed `{"uuid": "...", "status": "added"}` until 0.8.86; that
string appears nowhere in the codebase and never has. What actually comes back:

| outcome | over MCP | plugin, for comparison |
|---|---|---|
| stored | `"d064912c..."` — a bare JSON string | `{"uuid": "...", "tag": N}` |
| collision | `{"uuid", "status", "similarity", "existing_content", "note"}` | the same dict |

`backend.add()` returns the uuid on success and a verdict dict on a
dedup/contradiction collision, and this door serialises whichever it gets. The
plugin normalises because it has something to add — `tag` is the session
handle its seen-UUID gate spends, and an MCP call has no session to register
one in.

So a client must parse the result and branch on its type: a string means
stored, an object means the write was refused or questioned. Same class as
[`peek` returns a bare array](#peek-returns-a-bare-array-over-mcp) — a
documented difference, pinned by `MCP-T683`, rather than drift to be fixed
silently. It is a shape worth disliking; it is not a shape to change without
deciding what existing clients do about it.

#### `memory_maintenance` — Database maintenance

`action=` `sync_check` | `rebuild` | `purge` | `sleep` | `decay` | `resolve_conflicts` | `test_cleanup` | `review`.

`sleep`, `decay`, `resolve_conflicts`, and `review` are dry-run until `execute=true`.

**This is stricter than the Hermes plugin, deliberately.** On the plugin,
`layered_maintenance(action="sleep")` and `decay` execute immediately: the same
functions run as *automatic housekeeping* at teardown (`__init__.py`'s
maintenance block calls `be.sleep(...)` on an insert threshold), so gating them
would disable periodic maintenance. MCP has no such housekeeping role and is
the unauthenticated surface, so every mutating maintenance action there is
opt-in. `compact`, `review` and `resolve_conflicts` are dry-run on **both**
front ends. If you are comparing the two doors, this row is a difference by
design, not a parity bug.
`purge`, `test_cleanup`, and `rebuild` run immediately — there is no dry-run path
for them. `review` additionally needs `HLM_MCP_ADMIN` — it runs an LLM pass over
the table (unbounded spend) and soft-deletes on `execute=true`. Unlike the
plugin's version it is synchronous, because an MCP call has no session to report
back into later.

#### `memory_summaries` — Summary CRUD

`action=` `summarize` | `get` | `search` | `list` | `list_expiring` | `update` | `delete` | `batch_delete` | `sync`.

Backed by the same `digests.db` the plugin uses, so summaries written through
either front end are visible to both. Rows are scoped by profile; pass
`scope="all"` on delete/sync to cross that boundary deliberately.

**`scope` is a two-word vocabulary — `"own"` or `"all"` — and this door has
checked it since 2026-08-22.** Since 0.8.74 so does the backend: `_profile_sql`
and `_row_profile_ok` refuse an unknown word against one `PROFILE_SCOPES`
definition, where eight of the nine doors previously read anything that is not
the literal `"all"` as if the caller had said `"own"`. That mattered less over
MCP, which refuses first, than over the plugin, which had no check at all
(`T663`).

`batch_delete` returns four counts — `deleted` / `not_found` / `refused` /
`invalid`. `refused` is a row owned by another profile; `invalid` is an element
that is not a uuid string, which until 0.8.74 was reported as `not_found` or,
for a value SQLite could not bind, aborted the whole batch with a
`ProgrammingError` (`T664`).

#### `memory_advanced` — Advanced and debug

`action=` `peek` | `traces` | `stats` | `graph_health` | `discover` | `feedback` | `enrich` | `reenrich` | `compact`.

`graph_health` and `discover` are the plugin's `layered_advanced` and
`layered_memory` actions respectively; grouping differs between the two front
ends, coverage does not (T400).

`peek` inspects one pipeline layer's raw output (`layer` 0-4) and is the tool for
working out *why* a retrieval returned what it did. `compact` is dry-run until
`execute=true`.

#### `memory_io` — Import / export / backup — **requires `HLM_MCP_ADMIN=true`**

`action=` `export` | `import` | `backup` | `obsidian_ingest`.

Every action here reads or writes the local filesystem or bulk-extracts the
store, and the server has no authentication — `import` accepts a *file path* as
its payload, which is an arbitrary local file read. Enable only behind an
authenticating proxy or on localhost. `export` output is deliberately **not**
fenced so it can round-trip through `import`; treat it as untrusted data.

#### `memory_config` — Runtime configuration

`action=` `get` | `get_taxonomy` (open) | `set` | `delete` | `register_taxonomy` | `unregister_taxonomy` (**require `HLM_MCP_ADMIN=true`**).

Writes persist to the config file and change behaviour for the Hermes plugin
reading the same profile, which is why they are gated.

#### `web_search` — Web search via SearXNG

Only available when `HLM_SEARXNG_URL` is set. Searches the web and returns structured results. See [Web search](#web-search) for setup.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HLM_MCP_PATH` | No | `~/.hermes/...` | Base dir for all profile DBs |
| `HLM_DB_PATH` | No | — | Single DB override (overrides MCP_PATH) |
| `HLM_MCP_PROFILE` | No | `default` | Profile name → determines DB file |
| `HLM_MCP_MAX_LAYER` | No | `2` | Retrieval depth (2=fast, 3=LLM, 4=thorough) |
| `HLM_MCP_LIMIT` | No | `5` | Default result limit |
| `HLM_MCP_AGENT_ID` | No | `mcp-client` | Agent ID in history logs |
| `HLM_EMBED_URL` | No | — | Embedding endpoint URL |
| `HLM_EMBED_MODEL` | No | — | Embedding model name |
| `HLM_QDRANT_URL` | No | `http://localhost:6333` | Qdrant URL |
| `HLM_QDRANT_ENABLED` | No | `true` | Set `false` for SQLite-only mode |
| `HLM_SEARXNG_URL` | No | — | SearXNG URL (enables `web_search`) |
| `HLM_MCP_LLM_BUDGET` | No | `120` | Max LLM-backed requests per rolling window. Charged by `memory_query`/`peek` at `max_layer>=3` (1 — depth 3 and depth 4 cost the same, since depth 4 skips the reranker rather than adding to it), `review` (5), and `compact` (5, only when `execute=true` — a dry run never reaches the merge call and is free). `enrich` and `reenrich` are **not** flat-rate: each scales with the caller-supplied bound rather than the one request that reached the server, because both fan out to one LLM call per record internally. `enrich` charges `ceil(2 * max_items / 40)` (`enrich_existing` runs two selects, each `LIMIT max_items`, chunked at 40 records per classification call); `reenrich` charges `limit` directly (`re_enrich` issues one call per record). Both refuse a non-positive bound outright rather than charging 1 for what the backend treats as "every record" — an unbounded LLM pass metered as a single request. Retrieval below depth 3 is never charged, so exhausting the budget degrades depth rather than taking the server down. `0` disables it. |
| `HLM_MCP_LLM_BUDGET_WINDOW` | No | `3600` | Window in seconds for the above. |
| `HLM_MCP_ADMIN` | No | `false` | Enables the actions that reach outside the database — `memory_io.*` (filesystem read/write, bulk export) and `memory_config` writes, plus `memory_maintenance.review`. Off by default because the server has no authentication. |
| `HLM_SUMMARIES_DB` | No | `~/.hermes/hermes-layered-memory-dbs/digests.db` | Summaries store used by `memory_summaries` (shared across profiles, rows scoped by profile). |
| `HLM_SUMMARIES_DIR` | No | `~/Documents/hlm-summaries/` | Directory for summary `.md` files. |

### CLI flags

```bash
python3 mcp_server.py                              # streamable-http mode (default, localhost:3801)
python3 mcp_server.py --transport stdio             # stdio mode (for direct agent use)
python3 mcp_server.py --transport streamable-http   # HTTP mode (explicit, localhost)
python3 mcp_server.py --transport streamable-http --host 0.0.0.0  # HTTP mode (network)
python3 mcp_server.py --port 9999                   # custom port
```

### Troubleshooting

**Qdrant dimension error** — Your Qdrant collection was created with a different embedding model. Use the same backend consistently.

**`qdrant_client not installed`** — Install it: `pip install qdrant-client`. Server works without it (SQLite-only mode), but vector search is disabled. To explicitly disable Qdrant, set `HLM_QDRANT_ENABLED=false`.

**Embedding import error (`huggingface-hub`)** — The `sentence_transformers` fallback has a dependency conflict on some systems. Use Ollama or FastEmbed instead.

**DB auto-creates** — First connection with a new profile creates the SQLite DB automatically (`CREATE TABLE IF NOT EXISTS`). No manual provisioning needed.

### History tracking

All mutations are logged to `{db_dir}/memories-history.jsonl` alongside each DB. Format: one JSON line per event with timestamp, action, UUID, profile, agent_id, old/new fields. Rotates at 10MB (keeps last 3).

---

## Architecture

### Overview

HLM's MCP server is a **single shared HTTP endpoint** that serves multiple agents via profile-based isolation. One process, multiple profiles, shared vector index.

### Components

```
┌─────────────────────────────────────────────────┐
│              User (orchestrator)                 │
│  "Start one MCP server, agents connect to it"   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         mcp_server.py (single process)          │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │           ProfileRegistry                 │  │
│  │                                           │  │
│  │  profile: "pi"      → LayeredBackend #1  │  │
│  │  profile: "profile-a" → LayeredBackend #2│  │
│  │  profile: "shared"  → LayeredBackend #3  │  │
│  │  ...lazy creation on first access         │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Rate limiter: asyncio.Semaphore(5)             │
│  Per-profile: asyncio.Lock (write safety)       │
│  All sync calls: asyncio.to_thread()             │
│                                                 │
│  8 MCP tools (+ web_search when configured):    │
│  memory_query, memory_write,                    │
│  memory_list_profiles, memory_maintenance,      │
│  memory_summaries, memory_advanced,             │
│  memory_io*, memory_config*   (* HLM_MCP_ADMIN) │
└──────────────┬──────────────────┬───────────────┘
               │ streamable-http  │
               ▼                  ▼
┌──────────────────────┐ ┌──────────────────────┐
│ Agent: Pi            │ │ Agent: Hermes        │
│ (curl / MCP client)  │ │ (hermes-layered-mem) │
│                      │ │                      │
│ calls memory_write(action=add) │ │ calls memory_query   │
│ profile="pi"                   │ │ profile="profile-a"  │
└──────────────────────┘ └──────────────────────┘
```

### Storage Model

#### Per-Profile SQLite (isolated writes)

Each profile gets its own SQLite database file. This is the single source of truth for memory records.

| Profile | DB Path | Purpose |
|---------|---------|---------|
| `pi` | `~/.hermes/hermes-layered-memory-dbs/pi.db` | Pi's private memories |
| `profile-a` | `~/.hermes/hermes-layered-memory-dbs/profile-a.db` | Hermes's private memories |
| `shared` | `~/.hermes/hermes-layered-memory-dbs/shared.db` | Cross-agent shared memories |

- WAL mode enables concurrent reads across profiles
- `HLM_MCP_PATH` sets the base directory
- `HLM_DB_PATH` overrides for a single DB

#### Shared Qdrant (cross-profile vector index)

All profiles share one Qdrant vector database. Points are tagged with `profile_name` in their payload:

- **Point ID:** UUID converted to Qdrant format (32-char hex → dashed UUID via `_to_qdrant_id()`)
- **Vector:** embedding of memory content
- **Payload:** `{data_type, data_id, session_name, profile_name}`
- Queries filter by `profile_name` when targeting a specific profile
- Cross-profile search queries all profiles' points

### Data Flow

#### Query Flow

```
memory_query(query="coding conventions", profile="pi")
  │
  ├─ ProfileRegistry.get("pi") → LayeredBackend (creates if needed)
  │
  ├─ Layer 0: Qdrant ANN search
  │   ├─ Embed query → vector
  │   ├─ Filter: profile_name="pi"
  │   └─ Return top-k candidates [(uuid, distance)]
  │
  ├─ Layer 1: SQLite join + filters
  │   ├─ Open pi.db
  │   ├─ SELECT * FROM memories WHERE uuid IN (...)
  │   ├─ Apply: data_type, data_id, session_name, TTL, temporal
  │   └─ BM25 scoring via FTS5
  │
  ├─ Layer 2: Fusion scoring
  │   └─ RRF + BM25 + trust + recency + topic/keyword boosts
  │
  ├─ Layer 3 (optional): LLM reranker
  │   └─ Skip if BM25 scores are sufficient (smart-skip)
  │
  ├─ Layer 4 (optional): Gap detection
  │   └─ LLM checks if results fully answer the query
  │
  └─ Return top N results
```

#### Cross-Profile Query Flow

```
memory_query(query="coding conventions", cross_profile=true)
  │
  ├─ Layer 0: Qdrant ANN (NO profile filter, top_k × 3 for coverage)
  │   └─ Returns candidates from ALL profiles
  │
  ├─ FTS5 fallback: if Qdrant returns too few, search all profile DBs
  │
  ├─ Layer 1: Multi-DB join
  │   ├─ _discover_profile_dbs() → read config.yaml per profile, check .env for DB path
  │   ├─ For each DB:
  │   │   ├─ Open connection
  │   │   ├─ SELECT * WHERE uuid IN (candidates)
  │   │   ├─ BM25 scoring
  │   │   └─ Close connection
  │   └─ Merge results with profile_name tag
  │
  └─ Layers 2-4: same as single-profile query
```

#### Write Flow

```
memory_write(action="add", content="...", profile="pi")
  │
  ├─ ProfileRegistry.get("pi") → LayeredBackend
  ├─ ProfileRegistry.lock_for("pi") → asyncio.Lock
  │
  ├─ Embed content → vector
  ├─ Heuristic classification → data_type, data_id
  ├─ Semantic dedup (cosine >= 0.97 → return existing)
  ├─ INSERT INTO memories → pi.db
  ├─ Upsert to Qdrant (payload includes profile_name="pi")
  ├─ Write history archive entry (JSONL sidecar)
  ├─ Background LLM enrichment (non-blocking, daemon thread)
  │
  └─ Release lock
```

### Thread Safety

| Operation | Lock | Notes |
|-----------|------|-------|
| `memory_query` | None | WAL mode handles concurrent reads |
| `memory_list_profiles` | None | Scan directory + read-only queries |
| `memory_write` (all actions) | Per-profile | The `asyncio.Lock` wraps the whole action dispatch, so read-only `list`/`status` serialize behind writes to the same profile too |

Global rate limiter: `asyncio.Semaphore(5)` caps concurrent requests across all profiles.

### Tool Reference

| Tool | Params | Default Profile | Cross-Profile |
|------|--------|----------------|---------------|
| `memory_list_profiles` | none | N/A | Lists all |
| `memory_query` | query, max_layer, limit, scope, data_type, data_id, session_name, rerank, **profile**, **cross_profile** | `HLM_MCP_PROFILE` | `cross_profile=true` |
| `memory_write` | action=add/update/delete/list/status/maintenance, **profile**, plus action-specific params | `HLM_MCP_PROFILE` | N/A (writes are scoped — see Profile Isolation) |
| `web_search` | query, max_results, categories | N/A | N/A (requires `HLM_SEARXNG_URL`) |

`memory_write(action="update")` applies `content`, `summary`, `trust_score`, `topic`, `priority`, `ttl`, `sensitivity` and `protected` — `protected` is now passed through to `store.update()` (it was previously accepted by the tool and silently dropped). Fields left unset are ignored; at least one is required. `metadata` **is** among them: the parameter is declared on the tool and forwarded to `store.update()`, matching the plugin's `layered_memory(action="update")`, which gained `metadata` support in v0.7.55. This paragraph previously said the opposite — it described the state before the tool grew the parameter and was never revisited.

### Lifecycle

#### Startup

1. Parse CLI args: `--transport`, `--host`, `--port`
2. Read env vars: `HLM_MCP_PROFILE`, `HLM_*`
3. Create `ProfileRegistry` with default profile
4. Build MCP instructions (list available profiles)
5. Start FastMCP server (stdio or streamable-http)

#### Profile Creation

Profiles are created lazily on first access:
1. `ProfileRegistry.get("pi")` called
2. If no backend exists for "pi":
   - Validate the name and check the allowlist (inside `_get_db_path()`)
   - Resolve DB path: `{MCP_PATH}/pi.db`
   - Create directory if needed
   - If the cache already holds `HLM_MCP_MAX_PROFILES` backends (default 50), evict the least-recently-used one and close it on a worker thread
   - Create `LayeredBackend` (SQLite + Qdrant)
   - Create `asyncio.Lock` for writes
3. Return backend (and mark it most-recently-used)

The cache is an LRU `OrderedDict` rather than an unbounded dict: a caller can name an arbitrary number of distinct (regex-valid) profiles, and each cached backend holds a SQLite connection plus a Qdrant client. The cap bounds file descriptors and memory.

#### Shutdown

- `ProfileRegistry.shutdown()` is registered with `atexit` — every cached backend is closed on process exit
- SQLite WAL files persist (cleaned on next open)
- Qdrant connections close on process exit
- History archive files persist (rotated at 10MB)

### Security

#### Network Access

- **Default:** `127.0.0.1` (localhost only)
- **Network:** `--host 0.0.0.0` (requires firewall/VPN/SSH tunnel)
- **No authentication:** Server does not authenticate clients
- **No encryption:** HTTP is unencrypted (use SSH tunnel for transport security)

#### Profile Isolation

**Enforced by the server:**

- **Profile name validation.** Every profile name must match `^[A-Za-z0-9_-]+$` (`_validate_profile()`), checked in `_get_db_path()` — before any path construction — and again at the top of `memory_query` and `memory_write`. An invalid name returns `{"error": "Invalid profile name: ..."}` instead of being interpolated into a DB path, so `profile="../../../tmp/evil"` cannot create a file outside the DB directory.
- **Write scoping.** `memory_write` actions `add`/`update`/`delete`/`maintenance` go through `_check_write_profile_allowed()`: the target must be the server's default profile (`HLM_MCP_PROFILE`) or be listed in `HLM_MCP_ALLOWED_PROFILES`. Anything else returns a permission error and performs no write. Without this, a client could write to any profile on the host purely by naming it.
- **Read policy (looser, deliberately).** `memory_query` and the read-only `list`/`status` actions go through `_check_profile_allowed()` only: while `HLM_MCP_ALLOWED_PROFILES` is empty, every profile is readable, and `cross_profile=true` reads are intentional. Setting the allowlist tightens *targeted* reads too — only listed profiles can be named, so include the default profile in the list or backend creation for it will fail. It also disables cross-profile reads outright (see below).
- Each profile has an isolated SQLite DB, so `memory_write(action="delete")` only deletes from the profile that was resolved.

**Not enforced — operator responsibility:**

- There is still **no authentication**. Write scoping bounds *which* profile an anonymous caller can write to; it does not establish *who* is calling. Anyone who can reach the port can still write to the default profile. Treat this as a blast-radius control, not authn/authz — network-level controls (localhost bind, firewall, SSH tunnel) remain the actual access control.
- **With no allowlist set, cross-profile reads are unrestricted by design.** `cross_profile=true` resolves DBs through the backend's `_discover_profile_dbs()` (scans `~/.hermes/profiles/`), which does not consult `HLM_MCP_ALLOWED_PROFILES` — so any profile on the host with HLM enabled is readable that way. Do not keep content on this host that the connecting clients must not read.

  **With an allowlist set, they are refused, not filtered.** Because that scan cannot be restricted to the allowlist, both cross-profile entry points return a permission error instead of over-serving: `memory_query(cross_profile=true)` since `5006884` (shipped in 0.3.0), and `memory_advanced(action="discover")` — which calls `retrieve(cross_profile=True)` internally — since 0.7.63. Query the allowed profiles individually instead. This paragraph read "cross-profile reads **bypass the allowlist**" until 0.7.65, which was written the day *after* the first refusal shipped and stayed wrong for eleven days while `docs/security.md` described the same control correctly — understating the protection an operator actually has.

#### `peek` returns a bare array over MCP

`memory_advanced(action="peek")` returns the layer's results as a plain JSON
array; the plugin's `layered_memory(action="peek")` wraps the same data as
`{"layer": N, "results": [...]}`. The shapes differ and the MCP suite pins the
array form (MCP-T54/T55), so this is a documented difference rather than drift
to be fixed silently.

What it is *not* is a lost signal. `peek` deliberately carries no
`conflict_alert` — its own schema says so: "raw means raw: unlike retrieve it
does not attach a conflict_alert, so divergent records show up as ordinary
results". A 2026-08-20 review read the missing envelope as "conflict signals
can't be hoisted over MCP"; there are none to hoist, by design. Use
`memory_query` when you want the alert.

#### A missing summary is an error over MCP and a status on the plugin

`memory_summaries` `get` / `update` / `delete` against a uuid that is not there
return `{"error": "summary <8 chars> not found"}`. The plugin's twin returns
`{"status": "not_found", "uuid": "..."}` — a normal outcome with no `error`
key. Same call, same absent record, two classifications.

`delete`'s message is `not found **or not owned**`, and that conflation is
deliberate: distinguishing the two would let an unauthenticated client probe
for the existence of another profile's summaries. The *classification* as an
error is the part that is merely inherited — `SummariesBackend` returns a
falsy `ok` and this door has nowhere else to put it.

Worth knowing because `batch_delete` counts `not_found` as **data** on both
doors, so "absent is not an error" is the established vocabulary everywhere
except here. A client that switches doors must string-match the error text to
recover what `{"status": "not_found"}` says outright.

Documented rather than changed, on the same reasoning as the two shape notes
above: altering it would change what existing clients receive, and the choice
between the two classifications is a product decision, not a defect to be
quietly patched. Pinned by `T684`.

#### Error Responses

Tool errors are sanitized before they leave the process. `_safe_error(context, e)` logs the full exception with traceback server-side and returns only `"<context> failed: <ExceptionType>"` to the client (e.g. `"query failed: OperationalError"`). Raw exception text — which can carry local filesystem paths or SQL — is never handed to an unauthenticated caller. It covers backend creation and query in `memory_query`, every `memory_write` action, each individual `maintenance` operation, and per-profile status in `memory_list_profiles`.

Validation and permission failures are the deliberate exception: `Invalid profile name: ...` and `Writes to profile ... are not allowed ...` are returned verbatim, because the caller needs them to correct the call.

#### Observability

All mutations logged to history archive:
- Path: `{db_dir}/memories-history.jsonl`
- Format: JSON lines with timestamp, action, UUID, profile, agent_id
- Rotation: 10MB → `.1`, `.2`, `.3` (keep 3)

### Configuration

#### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HLM_MCP_PROFILE` | Server's default profile | `default` |
| `HLM_MCP_MAX_LAYER` | Default retrieval depth | `2` |
| `HLM_MCP_LIMIT` | Default result limit | `5` |
| `HLM_MCP_AGENT_ID` | Agent ID for history tracking | `mcp-client` |
| `HLM_MCP_ALLOWED_PROFILES` | Comma-separated profile allowlist. Empty/unset = all profiles readable and only the default profile writable; listing a profile makes it writable too (see Profile Isolation) | — (empty) |
| `HLM_MCP_MAX_PROFILES` | Max cached per-profile backends (each holds a SQLite connection + Qdrant client); least-recently-used backends are evicted and closed beyond the cap | `50` |
| `HLM_MCP_PATH` | Base directory for profile DBs | `~/.hermes/...` |
| `HLM_DB_PATH` | Override for single DB | — |
| `HLM_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `HLM_QDRANT_ENABLED` | Disable Qdrant (SQLite-only) | `true` |
| `HLM_EMBED_URL` | Remote embedding endpoint | — |
| `HLM_EMBED_MODEL` | Embedding model name | — |

#### CLI Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--transport` | Transport mode | `streamable-http` |
| `--host` | Bind address | `127.0.0.1` |
| `--port` | Bind port | `3801` |

#### Transport Modes

| Mode | Use Case | Connection |
|------|----------|------------|
| `streamable-http` | Shared network access (**default**) | Clients connect via URL |
| `stdio` | Direct agent use (Claude, Codex) | Agent spawns server process |

---

## Web search

The `web_search` tool provides internet search via [SearXNG](https://searxng.org), a privacy-respecting metasearch engine that aggregates results from Google, Bing, DuckDuckGo, Wikipedia, and 70+ other sources.

### Prerequisites

#### 1. Run a SearXNG Instance

Install SearXNG locally or on your network. Docker is the simplest method:

```bash
docker run -d -p 8888:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8888/ \
  searxng/searxng
```

#### 2. Enable JSON Format in SearXNG

By default, SearXNG only serves HTML results. You must enable JSON format in `settings.yml`:

**Docker:** Mount a custom `settings.yml` or set the environment variable:
```bash
docker run -d -p 8888:8080 \
  -e SEARXNG_SETTINGS_SEARCH_FORMATS='["html", "json"]' \
  -v ./searxng-settings:/etc/searxng \
  searxng/searxng
```

**Custom settings.yml:** Add JSON to the formats list under `search:`:
```yaml
search:
  formats:
    - html
    - json      # ← Add this line
```

Then restart the container:
```bash
docker restart <container_id>
```

**Verify JSON is enabled:**
```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -m json.tool | head -5
# Should return JSON, not HTML
```

#### 3. Configure the MCP Server

Set the `HLM_SEARXNG_URL` environment variable when starting the MCP server:

```bash
HLM_SEARXNG_URL=http://searxng:8888 \
  python3 mcp_server.py --transport streamable-http --port 3801
```

Or add it to your `~/.pi/agent/mcp.json`:
```json
{
  "mcpServers": {
    "hermes-layered-memory": {
      "url": "http://127.0.0.1:3801/mcp",
      "env": {
        "HLM_MCP_PROFILE": "pi",
        "HLM_SEARXNG_URL": "http://searxng:8888"
      }
    }
  }
}
```

### Tool Description

#### `web_search` — Search the internet

Searches the web using your SearXNG instance and returns structured results.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *(required)* | Search query string |
| `max_results` | integer | `5` | Maximum number of results to return (1–20) |
| `categories` | string | `"general"` | Search category: `general`, `images`, `videos`, `news`, `files`, `it`, `science`, `files`, `music` |

**Returns:** JSON array of result objects:
```json
[
  {
    "title": "Page title",
    "url": "https://example.com/page",
    "content": "Snippet of the page content...",
    "engines": ["google", "duckduckgo"],
    "score": 7.5,
    "publishedDate": "2026-07-14",
    "author": "Author name"
  }
]
```

**Example:**
```python
# Via MCP tool call
mcp({tool: "web_search", args: '{"query": "NVIDIA driver release notes", "max_results": 3}'})

# Returns results from Google, Bing, DuckDuckGo, etc.
```

### Behavior

- **Conditional registration:** The `web_search` tool only appears in the MCP tool list if `HLM_SEARXNG_URL` is set. Users without SearXNG see no change.
- **Timeout:** 10 seconds per request. Returns an error if SearXNG is unreachable.
- **Rate limiting:** Subject to your SearXNG instance's configured limits and the upstream engines' rate limits.
- **Error handling:** If SearXNG is down or misconfigured, the tool returns a JSON error object with a `hint` field.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `web_search` tool not appearing | Check `HLM_SEARXNG_URL` is set and server restarted |
| Returns HTML instead of JSON | Enable JSON format in SearXNG `settings.yml` (see above) |
| `SearXNG search failed: HTTP error` | Verify SearXNG is running and accessible from the MCP server host |
| Empty results | Check SearXNG engine status at `http://searxng:8888/stats` |
| Slow responses | Increase timeout in code or optimize SearXNG engine selection |

### Security Notes

- SearXNG proxies all search queries, so your upstream IP is hidden from search engines
- No API keys required for SearXNG itself
- If exposing SearXNG on a network, consider adding authentication via reverse proxy (nginx/Traefik)
- The MCP server connects to SearXNG over HTTP — use HTTPS if SearXNG is on a different host
