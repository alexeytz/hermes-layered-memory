# Layered Memory (HLM) Plugin for Hermes Agent

Multi-layer memory retrieval with precision filtering. SQLite (FTS5, metadata) + Qdrant (HNSW vectors) hybrid backend with user-controlled depth, circuit breaker fallback, cross-profile search, Obsidian vault ingestion, semantic dedup, trust score feedback, and explicit rerank control.

**Original:** <https://github.com/alexeytz/hermes-layered-memory>

**6 meta-tools:** layered_memory, layered_maintenance, layered_summaries, layered_advanced, layered_io, layered_config. All operations dispatch through an `action` parameter. Full generated reference: [docs/tools.md](docs/tools.md).

## Features

- **5-layer retrieval pipeline** — Qdrant ANN → SQLite filters → fusion scoring → LLM reranker → gap detection
- **Hard heuristics** — layer escalation gates prevent over-use of expensive LLM layers
- **Cross-profile search** — per-profile SQLite, shared Qdrant, BM25 per-DB
- **Obsidian ingestion** — frontmatter parsing, wikilinks, tags extraction. `vault_path` must resolve under an allowed root (home by default — see `HLM_OBSIDIAN_VAULT_ROOTS`)
- **Prompt injection defense** — untrusted content fenced on the documented paths to the model: retrieve/peek/list (content, summary, keywords, backlinks, metadata), prefetch, summary read/search/browse, and the L3/L4/classify/merge LLM prompts. Authoritative path list: [docs/security.md](docs/security.md)
- **Conflict detection** — configurable thresholds, cosine + Jaccard + LLM annotation
- **Compaction source preservation** — merged records inherit obsidian source (security-first OR)
- **Auto-priority** — ENV-DATA/SYSTEM default to `priority=1` (half decay). Pinned (priority=3) immune to decay.
- **Smart-skip** — Layer 3 skips LLM when BM25 scores are sufficient (~37s saved)
- **Retry** — Layer 3/4 LLM calls retry 3x with exponential backoff; non-retryable 4xx responses (bad key, wrong model) fail fast instead of burning the budget
- **Embedding stripped** — not in tool responses (saves 280K+ chars per call)
- **Circuit breaker** — 5 Qdrant failures → 120s cooldown, then a single half-open trial request. While degraded (disabled, cooldown, or init failure) retrieval keeps real semantic ranking via brute-force cosine over the locally-stored vectors, falling back to FTS5/BM25 only when that is unavailable
- **Supersession** — `add(..., supersedes=<uuid>)` records that a fact changed
- **Prefetch gating** — turns carrying no retrievable intent inject nothing; `sensitivity > 0` never auto-injected
- **Extraction ledger** — every auto-extraction recorded in history sidecar, surfaced via `layered_advanced(action="stats")`
- **Retrieval evaluation** — `tests/eval_retrieval.py` measures recall@5, MRR and latency per layer

**Security:** [docs/security.md](docs/security.md) — trust boundaries and threat model.

## Quick Start

**Not sure what any of this should look like for your setup?** Answer a few
questions and get the whole plan printed — the commands, the exact `.env`, and
what to verify:

```bash
python3 scripts/plan-install.py
```

It changes nothing. It asks for a profile name, whether embeddings come from
the built-in 384-dim model or an endpoint, whether Qdrant is starting here or
already running or off, and whether there is an LLM for the deeper layers —
then prints the steps below, filled in. The rest of this section is the same
thing, generic.

### 1. Symlink Plugin

```bash
mkdir -p ~/.hermes/profiles/<profile>/plugins
ln -sf /path/to/hermes-layered-memory ~/.hermes/profiles/<profile>/plugins/hermes-layered-memory
```

### 2. Start Qdrant

```bash
cd /path/to/hermes-layered-memory
docker compose up -d
sleep 3
curl -s http://localhost:6333/   # verify: returns version info
```

Data persists in `./qdrant-data/` (bind mount). Ports: 6333 (HTTP), 6334 (gRPC).

**No Docker on this host?** Skip this step and run SQLite-only: put
`HLM_QDRANT_ENABLED=false` in the `.env` you create in step 4 (it is already
there, commented out). HLM works without Qdrant — retrieval falls back to a
brute-force cosine scan over the embeddings SQLite already holds, so semantic
ranking survives; latency grows with corpus size, and semantic dedup at write
time is bypassed, so run `layered_advanced(action="compact")` if you later
enable Qdrant. This escape used to be documented only further down the page,
which left an operator without Docker stuck at a step with no exit.

### 3. Enable the Plugin

**Option A — CLI (recommended):**

Use `-p <profile>` to target any profile without switching:

```bash
hermes -p <profile> plugins enable hermes-layered-memory
hermes -p <profile> config set memory.memory_enabled false
hermes -p <profile> config set memory.user_profile_enabled false
hermes -p <profile> config set memory.write_approval false
hermes -p <profile> config set memory.provider hermes-layered-memory
```

**Option B — config.yaml (manual):**

Edit `~/.hermes/profiles/<profile>/config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-layered-memory

memory:
  memory_enabled: false
  user_profile_enabled: false
  write_approval: false
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: hermes-layered-memory
```

Restart the Hermes session. The plugin auto-initializes on the first tool call (lazy init).

**Two things that surprise a first-time installer:**

- **A profile needs its own `model:` block.** Profiles do not inherit the root
  config's, so on a newly created profile the first `chat` exits with
  *"No inference provider configured"* before HLM is ever reached. Set one (in
  the same `config.yaml`, or with `hermes -p <profile> config set`) pointing at
  whichever provider you use.
- **`plugins enable` may report that it could not grant a tool override.** That
  happens when the command runs without a TTY and is harmless here: HLM
  registers a memory provider and does not override any existing tool, so
  `allow_tool_override: false` costs it nothing. It initializes normally.

### 4. Configure

Copy the minimal JSON config and `.env.example`:
```bash
cp hermes-layered-memory.example.json ~/.hermes/profiles/<profile>/hermes-layered-memory.json
cp .env.example ~/.hermes/profiles/<profile>/.env
```

Edit `.env` only — the JSON config (collections, scoring) rarely needs changes:
```bash
# ~/.hermes/profiles/<profile>/.env
HLM_DB_PATH=~/.hermes/hermes-layered-memory-dbs/<profile>.db
HLM_QDRANT_URL=http://localhost:6333
HLM_EMBED_URL=http://localhost:11434/api/embed
HLM_EMBED_MODEL=qwen3-embedding:8b
HLM_MAX_LAYER=2
HLM_LAYER3_MODEL=QWEN3.6-27B
HLM_LAYER3_BASE_URL=http://localhost:8000/v1
HLM_LAYER3_API_KEY=your-api-key
HLM_REASONING_EFFORT=none
HLM_DEDUP_THRESHOLD=0.97
HLM_LOG=DEBUG
```

**DB path:** Defaults to `~/.hermes/hermes-layered-memory-dbs/<profile>.db` (auto-created). Set `HLM_DB_PATH` in .env to override. See `.env.example` for all available env vars.

**Filesystem access:** `layered_io(action="obsidian_ingest")` and `layered_io(action="import")` only read paths that resolve under an allowed root. Both default to the real home directory. Extend with `HLM_OBSIDIAN_VAULT_ROOTS` / `HLM_IMPORT_ALLOWED_ROOTS` (colon-separated absolute paths) — **a vault or import file outside home is now rejected**, so existing callers pointing at, e.g., `/mnt/vaults/notes` must set the corresponding var.

**Qdrant timeout:** `HLM_QDRANT_TIMEOUT` sets the Qdrant HTTP client timeout in seconds (default 30) instead of inheriting the client library's implicit default.

### 5. Verify

```bash
python3 scripts/check-environment.py
```

Two verdicts, deliberately separate: the repo's own invariants (the same answer
on every machine) and this machine's readiness (embedder reachable, Qdrant
answering, the layer-3 endpoint if you configured one). **Export
`HLM_EMBED_URL` and `HLM_EMBED_MODEL` in the shell first if you set them** —
the check reads the environment, not the profile's `.env`, and an unset pair is
the one failure that otherwise hides: every vector call silently uses the
384-dim fallback while everything reports success.

Then start a session; the plugin initialises lazily on the first tool call:

```bash
hermes -p <profile> chat -q "remember that the deploy host is host-01"
```

### Disable the Plugin

```bash
hermes -p <profile> plugins disable hermes-layered-memory
```

**Remember:** If you re-enable, also re-set the `memory:` config keys (see step 3) or Hermes will fall back to the built-in MEMORY.md/USER.md system.

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/reference.md](docs/reference.md) | Full API reference: tools, config, env vars, scoring, cleanup |
| [docs/architecture.md](docs/architecture.md) | Pipeline, data partitioning, lifecycle hooks, conflict detection |
| [docs/troubleshooting.md](docs/troubleshooting.md) | FAQ, common errors, dependencies, testing |
| [docs/security.md](docs/security.md) | Trust boundaries and threat model |
| [docs/profile-isolation.md](docs/profile-isolation.md) | Profile isolation, .env loading, secret scope |
| [docs/mcp.md](docs/mcp.md) | MCP server design |
| [docs/mcp-profile-setup.md](docs/mcp-profile-setup.md) | Standing up an MCP-only profile and its server fleet |

## Testing

```bash
# Run full regression suite (~7-20 min — background it)
python3 tests/run-regression.py

# Run dispatch-layer tests (37 tests, ~15s)
python3 tests/test_dispatch.py
```

```bash
# Is this clone and this machine actually ready?
python3 scripts/check-environment.py
```

The suite is not part of this distribution; see the development repository.

## Dependencies

HLM's Python dependencies are **not auto-installed** by Hermes. After a Hermes upgrade:

```bash
# torch first, from the CPU index. `sentence-transformers` depends on torch, and
# the default PyPI torch wheel is the CUDA build — it pulls the nvidia-* runtime
# libraries and triton with it. Measured on a CPU-only host: 2.7G of nvidia-*,
# 1.1G of torch and 690M of triton, to run a library that is itself 4.8M.
#
# None of it is ever used. HLM constructs the model as
# `SentenceTransformer(model, device="cpu")` (backend/core.py) — hardcoded — so
# it would not touch a GPU even on a machine that has one. On a GPU-less host
# `torch.cuda.is_available()` is False and the CUDA libraries are never loaded.
# The CPU wheel is the right choice on every host, which is why this is not a
# trade-off worth thinking about.
~/.hermes/hermes-agent/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.hermes/hermes-agent/venv/bin/pip install qdrant-client numpy sentence-transformers
```

If you already installed the CUDA build, `pip uninstall torch` and re-run the
first command; the `nvidia-*` and `triton` packages can be uninstalled too.

Skip torch entirely if you are not using the default local embedder: set
`HLM_EMBED_URL` (any Ollama-compatible endpoint) or `HLM_LOCAL_EMBED_MODEL`
(FastEmbed, ONNX, no torch), and `sentence-transformers` is never imported.

| Package | Required? | Purpose |
|---------|-----------|---------|
| qdrant-client >= 1.18 | Yes | Qdrant vector search client |
| numpy | Yes | Brute-force semantic search fallback |
| sentence-transformers >= 5.5 | Only for the default local embedder | 384-dim MiniLM fallback; install torch from the CPU index first (above). Unused when `HLM_EMBED_URL` or `HLM_LOCAL_EMBED_MODEL` is set |
| fastembed | No | Local multilingual embedding |
| Docker + Docker Compose | No | Qdrant container |

> **SQLite-only mode:** Set `HLM_QDRANT_ENABLED=false` to disable Qdrant
> entirely. Retrieval falls back to a brute-force cosine scan over the
> embeddings SQLite already holds, so semantic ranking survives; latency grows
> with corpus size. FTS5/BM25 lexical ranking is only the *second* fallback,
> used if numpy is missing or the embedder cannot produce a query vector.
> Write-time semantic dedup is bypassed — run
> `layered_advanced(action="compact")` if you later enable Qdrant. Useful when
> Docker is unavailable or for lightweight setups.

## MCP Server (External Agent Access)

HLM ships with an MCP server for external agents that don't run Hermes.

```bash
# HTTP mode (default)
HLM_MCP_PROFILE="profile-a" python3 mcp_server.py

# Stdio mode (for direct agent use)
HLM_MCP_PROFILE="profile-a" python3 mcp_server.py --transport stdio
```

See [docs/mcp.md](docs/mcp.md) for standalone usage and the full multi-agent design.

## Contributing

**Commit conventions:** `feat:`, `fix:`, `docs:`, `test:`

**Branch policy:** `main` is production-ready. Feature branches prefixed `feature/` or `fix/`.

**Mandatory:** code changes require corresponding test updates. Run `python3 tests/run-regression.py && python3 tests/test_dispatch.py` before committing.

`docs/architecture.md` covers the pipeline and data model, and `docs/reference.md` the full tool and config surface.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Alexey T.
