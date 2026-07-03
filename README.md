# Layered Memory (HLM) Plugin for Hermes Agent

Multi-layer memory retrieval with precision filtering. SQLite (FTS5, metadata) + Qdrant (HNSW vectors) hybrid backend with user-controlled depth, circuit breaker fallback, cross-profile search, Obsidian vault ingestion, semantic dedup, trust score feedback, and explicit rerank control.

**Original:** <https://github.com/alexeytz/hermes-layered-memory>

**28 tools:** retrieve, peek, add, update, delete, rebuild, purge, sync_check, list, sleep, enrich, obsidian_ingest, feedback, review, compact, export, import, decay, backup, summarize, list_summaries, get_summary, search_summaries, delete_summary, update_summary, delete_multiple_summaries, list_expiring_summaries, sync_summaries.

## Features

- **5-layer retrieval pipeline** — Qdrant ANN → SQLite filters → fusion scoring → LLM reranker → gap detection
- **Hard heuristics** — layer escalation gates prevent over-use of expensive LLM layers
- **Cross-profile search** — per-profile SQLite, shared Qdrant, BM25 per-DB
- **Obsidian ingestion** — frontmatter parsing, wikilinks, tags extraction
- **Smart-skip** — Layer 3 skips LLM when BM25 scores are sufficient (~37s saved)
- **Retry** — Layer 3/4 LLM calls retry 3x with exponential backoff
- **Embedding stripped** — not in tool responses (saves 280K+ chars per call)
- **Circuit breaker** — Qdrant offline → SQLite fallback, 120s cooldown

## Quick Start

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

For the default profile (no `-p` needed):
```bash
hermes plugins enable hermes-layered-memory
hermes config set memory.memory_enabled false
hermes config set memory.user_profile_enabled false
hermes config set memory.write_approval false
hermes config set memory.provider hermes-layered-memory
```

**Option B — config.yaml (manual):**

Edit `~/.hermes/profiles/<profile>/config.yaml` (or `~/.hermes/config.yaml` for the default profile):

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
The `provider: hermes-layered-memory` line tells Hermes to use the layered memory provider.

Restart the Hermes session. The plugin auto-initializes on the first tool call (lazy init).

### 4. Configure

Copy the minimal JSON config and `.env.example`:
```bash
cp hermes-layered-memory.example.json ~/.hermes/profiles/<profile>/hermes-layered-memory.json
cp .env.example ~/.hermes/profiles/<profile>/.env
```

Edit `.env` only — the JSON config (collections, scoring) rarely needs changes:
```bash
# ~/.hermes/profiles/<profile>/.env
HERMES_LAYERED_DB_PATH=~/.hermes/hermes-layered-memory-dbs/<profile>.db
HERMES_LAYERED_QDRANT_URL=http://localhost:6333
HERMES_LAYERED_EMBED_URL=http://localhost:11434/api/embed
HERMES_LAYERED_EMBED_MODEL=qwen3-embedding:8b
HERMES_LAYERED_MAX_LAYER=2
HERMES_LAYERED_LAYER3_MODEL=QWEN3.6-27B
HERMES_LAYERED_LAYER3_BASE_URL=http://localhost:8000/v1
HERMES_LAYERED_LAYER3_API_KEY=your-api-key
HERMES_LAYERED_REASONING_EFFORT=none
HERMES_LAYERED_LOG=DEBUG
```

**DB path:** Defaults to `~/.hermes/hermes-layered-memory-dbs/<profile>.db` (auto-created). Set `HERMES_LAYERED_DB_PATH` in .env to override. No `db_path` needed in JSON config or config.yaml.

See `.env.example` for all available env vars and defaults.

**Cross-profile search** — each profile has its own SQLite DB, but all share one Qdrant index. The `profile_name` payload in Qdrant enables:
- Default: search current profile only
- `cross_profile=true`: search all profiles
- `profile_name="other-profile"`: search specific profile
- General session (no profile): sees everything

### 4. Disable

To disable the plugin:
```bash
hermes -p <profile> plugins disable hermes-layered-memory
```

For the default profile (no `-p` needed):
```bash
hermes plugins disable hermes-layered-memory
```

Verify it's disabled:
```bash
hermes -p <profile> plugins list | grep hermes-layered-memory   # should show "not enabled"
```

The plugin code and config remain intact — it just won't load on next session. Re-enable with:
```bash
hermes -p <profile> plugins enable hermes-layered-memory
```

**Remember:** If you re-enable, also re-set the `memory:` config keys (see step 3) or Hermes will fall back to the built-in MEMORY.md/USER.md system.

## Architecture

```
Query → [Layer 0: Qdrant ANN] → [Layer 1: SQLite filters] → [Layer 2: Fusion scoring] → [Layer 3: Reranker] → [Layer 4: Gap detect] → Agent
```

| Layer | Operation | Cost | Filter |
|-------|-----------|------|--------|
| 0 | Qdrant vector search (HNSW) | ~5ms | data_type |
| 1 | SQLite join + FTS5 BM25 | ~10ms | data_id, session_name, scope, TTL, temporal |
| 2 | ByteRover compound scoring | ~2ms | BM25 70% + importance 15% + recency 15% |
| 3 | LLM reranker | 10-40s (or instant if skipped) | Requests `limit*3` candidates from Layer 2. LLM decides if reranking is needed — returns `{"rerank_needed": false}` if BM25 already ranks well, saving ~30s. If ambiguous, re-ranks and returns top `limit`. Falls back to pass-through if LLM unavailable. Config: `layer3_mode`, `layer3_model`, `layer3_provider_config`. |
| 4 | Gap detection | 10-40s | Calls LLM to check if results answer the query, identifies missing info. Annotates records with gaps and confidence. Falls back to `gap_checked=true, gaps=[], confidence=1.0`. |

### Data Partitioning

Memories are partitioned for precise retrieval:

| Field | Purpose | Examples |
|-------|---------|----------|
| `data_type` | Structural partition + collection routing | USER-DATA, ENV-DATA, SESSION-DATA, SYSTEM, CUSTOM (user-defined, mapped to collections in `hermes-layered-memory.json`) |
| `data_id` | Sub-partition | HW, SW, preferences, session UUID |
| `session_name` | Human-readable label | "vLLM GPU tuning", "memory plugin review" |

**Multi-collection routing** — `data_type` maps to Qdrant collections via config `collections`:
```json
"collections": {
  "SYSTEM": "memories",
  "USER-DATA": "memories",
  "ENV-DATA": "memories",
  "SESSION-DATA": "sessions",
  "OBSIDIAN": "vault",
  "CUSTOM": "memories"
}
```

- Upserts route to the correct collection by `data_type`
- Queries without `data_type` filter search all configured collections, merge results
- Queries with `data_type` filter target that collection directly
- Each collection is independent: different vector config, backup, deletion
- Default: all types use `memories` collection if `collections` not specified

## Operations

### Logging

Logs written to `<hermes-home>/logs/hermes-layered-memory.log`.
- Default profile: `~/.hermes/logs/hermes-layered-memory.log`
- Named profile: `~/.hermes/profiles/<profile>/logs/hermes-layered-memory.log`

Override via `HERMES_LAYERED_LOG_FILE` (see env vars table below).

```bash
# Follow the log in real-time
tail -f ~/.hermes/logs/hermes-layered-memory.log

# Control verbosity (INFO default, DEBUG for full layer pipeline)
export HERMES_LAYERED_LOG=DEBUG
```

**Layered env vars** — add to `~/.hermes/profiles/<profile>/.env` (profile-specific) or `~/.hermes/.env` (global). Loaded automatically on session start.

| Env var | Default | Purpose |
|---------|---------|---------|
| `HERMES_LAYERED_DB_PATH` | `~/.hermes/hermes-layered-memory-dbs/<profile>.db` | SQLite database path (auto-created). |
| `HERMES_LAYERED_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (REQUIRED). |
| `HERMES_LAYERED_MAX_LAYER` | `2` | Default max layer depth: 0=Qdrant, 1=+SQLite, 2=+fusion, 3=+LLM reranker, 4=+gap detect. |
| `HERMES_LAYERED_EMBED_URL` | _(none)_ | Remote embedding endpoint (Ollama `/api/embed`, etc.) |
| `HERMES_LAYERED_EMBED_MODEL` | _(none)_ | Embedding model name (e.g. `qwen3-embedding:8b`) |
| `HERMES_LAYERED_LAYER3_MODE` | `inline` | Reranker mode: `inline` (single call), `delegate` (sub-agent), `self` (in-context). Only if `max_layer >= 3`. |
| `HERMES_LAYERED_LAYER3_MODEL` | _(none)_ | LLM model for Layer 3 reranker. Only if `max_layer >= 3`. |
| `HERMES_LAYERED_LAYER3_BASE_URL` | _(none)_ | Base URL for Layer 3 LLM endpoint (OpenAI-compatible). |
| `HERMES_LAYERED_LAYER3_API_KEY` | _(none)_ | API key for Layer 3 LLM endpoint. |
| `HERMES_LAYERED_REASONING_EFFORT` | _(none)_ | Disable/reduce model thinking mode for all HLM LLM calls. `"none"` disables thinking (Qwen3, DeepSeek, Gemma4 via vLLM). `"low"`/`"medium"`/`"high"` enable varying depths. Omit to preserve model default behavior. These are JSON-formatting tasks — reasoning is wasteful (~700 tokens, ~5s extra per call). |
| `HERMES_LAYERED_ENRICH_LLM` | `false` | Enable LLM fallback for topic/keywords during enrichment. Default `false` (heuristic-only, instant). Set `true` to run LLM classification when heuristics fill data_type/data_id but leave topic/keywords empty. Costs ~1-2s and ~100 tokens per add with missing metadata. |
| `HERMES_LAYERED_LOW_TRUST_ARCHIVE_DAYS` | `365` | Days before low-trust (<0.3) records are auto-archived by `layered_sleep()`. Records with priority >= 2 are never auto-archived. |
| `enrich_on_add` | `"low_confidence"` | Set in `hermes-layered-memory.json`. Enrichment mode at add-time. `"low_confidence"` (default) runs LLM only when heuristic confidence < 0.6. `"true"` always runs LLM enrichment. `"false"` disables enrichment entirely. |
| `min_bm25_threshold` | `0.0` | Set in `hermes-layered-memory.json`. Drop results below BM25 threshold. Default `0.0` (disabled). Set `0.1` to filter out weak matches (BM25=0 noise). Applied in `_layer1()` after BM25 scoring. |
| `dedup_threshold` | `0.97` | Set in `hermes-layered-memory.json`. Semantic dedup at add-time. Compares new content against existing memories via vector similarity. Default `0.97` (very high bar — avoids false positives on distinct content about same topic). Set `0.0` to disable. On duplicate: returns existing UUID with similarity score, does NOT store. Agent can use `layered_update` to merge content. |
| `query_expand` | `false` | Set in `hermes-layered-memory.json`. BM25 query expansion. When true, expands BM25 query with keywords from top-5 vector matches. Improves cross-vocabulary recall (e.g., "GPU" → picks up "NVIDIA", "CUDA", "video memory"). Set `true` for broader recall. |
| `entity_patterns_path` | _(unset)_ | Set in `hermes-layered-memory.json`. Override entity patterns file path. Default: all `.json` files in `entity-patterns/` folder. Set to a single file path for full replacement. |
| `HERMES_LAYERED_LOG_FILE` | `<hermes-home>/logs/hermes-layered-memory.log` | Log file path |
| `HERMES_LAYERED_LOG` | `INFO` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |
| `HERMES_LAYERED_SUMMARIES_DIR` | `~/Documents/hlm-summaries/` | Summaries .md files directory |
| `HERMES_LAYERED_SUMMARIES_DB` | `~/.hermes/hermes-layered-memory-dbs/digests.db` | Shared summaries SQLite DB (all profiles write to the same file; WAL mode handles concurrency) |

Example log output:
```
22:35:12 INFO hermes-layered-memory.backend retrieve: query='who am I?' max_layer=3 scope=None data_type=None data_id=None session_name=None limit=5
22:35:12 DEBUG hermes-layered-memory.backend layer0: 30 candidates
22:35:12 DEBUG hermes-layered-memory.backend layer1: 17 records after filters
22:35:12 DEBUG hermes-layered-memory.backend layer2: 17 scored
22:35:12 DEBUG hermes-layered-memory.backend layer3: 17 reranked
22:35:12 INFO hermes-layered-memory.backend retrieve: returned 5 results
```

### Tool Reference

All 28 tools expose inline documentation with explicit data_type/data_id mapping. The model sees this in its system prompt — no need to repeat the mapping in every call.

| Tool | Purpose |
|------|---------|
| `layered_retrieve(query, max_layer, scope, data_type, data_id, session_name, profile_name, cross_profile, limit)` | Multi-layer retrieval |
| `layered_peek(query, layer)` | Inspect a specific layer output |
| `layered_add(content, summary, topic, keywords, scope, data_type, data_id, session_name, sensitivity, ttl, priority)` | Store a memory |
| `layered_update(uuid, ...)` | Update existing memory |
| `layered_delete(uuid)` | Soft-delete (marks as deleted) |
| `layered_rebuild(since)` | Rebuild Qdrant index from SQLite |
| `layered_purge(purge_deleted, purge_archived, min_age_hours, vacuum)` | Permanently delete soft-deleted/archived records + Qdrant cleanup |
| `layered_sync_check()` | Check Qdrant ↔ SQLite sync |
| `layered_list(topic, scope, limit, sort)` | List recent memories |
| `layered_sleep(max_items, min_age_hours)` | Consolidation: merge duplicates, archive stale |
| `layered_enrich(since, max_items)` | Batch-enrich existing memories with empty metadata |
| `layered_feedback(uuid, helpful)` | Increments/decrements trust_score (+/-0.1) |
| `obsidian_ingest(vault_path)` | Ingest Obsidian vault notes |
| `layered_review(min_age_hours, force)` | LLM review: classify records as keep/delete, soft-deletes garbage. Fire-and-forget background session. |
| `layered_compact(similarity_threshold, topic, max_groups, dry_run)` | Merge duplicate/overlapping memories via LLM. Corroboration boost for trust score. |
| `layered_export(format, status, data_type, topic, scope, profile_name, cross_profile)` | Export memories as JSON or Markdown. Supports filtering and cross-profile export. |
| `layered_import(data, mode, target_status)` | Import memories from JSON. Modes: skip_existing, overwrite, new_uuid. Triggers Qdrant rebuild. |
| `layered_decay(min_age_days, max_age_days, decay_rate, min_score)` | Confidence decay: reduces trust_score for stale memories. Integrated into session-end maintenance. |
 | `layered_backup(dest_dir, keep_days)` | Atomic SQLite backup of both memories and summaries DBs. WAL-safe. Default: 7-day retention. |

### Summaries (SQLite only — intentionally no Qdrant)

Summaries are stored separately from layered memory. Lookup-based search via FTS5 with Porter stemmer (not semantic similarity). Full summaries saved as portable `.md` files. Deep search indexes full_text in FTS5. Lifecycle metadata (status, coverage_status, token_estimate) + content_hash for freshness detection. URL canonicalization (YouTube, GitHub) + idempotency (already_exists). FTS5 health check at startup + auto-rebuild on corruption. Relative paths for portability. Tag normalization (lowercase, hyphens).

**Storage model (dual copy):**
- **DB (`digests.db`):** Stores `full_text` for FTS5 deep search indexing. Without this, FTS5 can only search highlights/title/tags, missing content buried in the summary body.
- **`.md` files:** Portable export, human-readable, version-control friendly. This is the primary read path — `layered_get_summary()` returns the `.md` file path, not the `full_text` inline. The agent reads `.md` via `read_file(offset, limit)` for context budget control.
- **Why both?** The DB copy enables fast full-text search. The `.md` copy enables portability, offline access, and controlled reading. The redundancy is intentional — search needs DB, reading needs files.

| Tool | Purpose |
|------|---------|
| `layered_summarize(source, source_type, title, highlights, full_text, tags, force)` | Store a summary (idempotent by source_url) |
| `layered_list_summaries(source_type, source_url, tag, limit, offset, sort_by)` | List stored summaries (filtered count) |
| `layered_get_summary(uuid)` | Get summary by UUID (returns highlights + path to .md) |
| `layered_search_summaries(query, limit)` | Search summaries via FTS5 (snippet + rank) |
| `layered_update_summary(uuid, title, highlights, tags, metadata, full_text)` | Update a summary (keeps same UUID) |
| `layered_delete_summary(uuid)` | Delete summary + .md file |
| `layered_delete_multiple_summaries(uuids)` | Batch delete (returns deleted/not_found count) |
| `layered_list_expiring_summaries(max_age_days)` | List old summaries (TTL cleanup) |
| `layered_sync_summaries()` | Sync DB with filesystem (delete orphaned records) |

**Storage:** `digests.db` (SQLite + FTS5, WAL mode) + `.md` files in `~/Documents/hlm-summaries/` (env: `HERMES_LAYERED_SUMMARIES_DIR`).

**Data extraction:** Use `markitdown` package (already installed) to convert PDF/DOCX/PPTX/YouTube/etc. to Markdown before summarizing.

**Default limit:** `layered_retrieve` returns 5 results by default.

**Parameter defaults:** `scope`, `data_type`, `data_id`, `session_name` default to `None` (no filter). When `None`, the agent searches across all partitions. The agent chooses filters based on query context — general questions get unfiltered searches, specific questions get targeted filters (e.g. `data_type=SYSTEM` for rules, `data_id=preferences` for user prefs). You can instruct the agent to use specific filters in prompts.

**Scoring:** All weights are configurable in `hermes-layered-memory.json` under `scoring`. Defaults are baked into the code — config section is optional.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm25_weight` | 0.7 | Lexical match importance (FTS5 BM25). Higher = exact keyword matches dominate. Lower = vector similarity matters more. |
| `importance_weight` | 0.15 | Influence of `trust_score` (0-1). Higher = trusted memories surface more. |
| `recency_weight` | 0.15 | Influence of freshness. Higher = newer memories favored. Pinned items (priority=3) exempt from decay. |
| `topic_boost` | 0.3 | Flat bonus when query words appear in `topic` field. Use to prioritize topic-based retrieval. |
| `keyword_boost` | 0.15 | Flat bonus when query words appear in `keywords` array. Lower than topic — keywords are secondary. |
| `pinned_boost` | 1.5 | Multiplier for priority=3 memories. Reduces to 1.0 to disable pinning. |
| `recency_half_life_days` | 21 | Days until a memory's recency score drops by half. Lower = memories age faster. |
| `rrf_constant` | 60 | RRF (Rank Fusion) constant. Higher = rank position matters less, scores more uniform. Lower = top-ranked memories dominate. |

Tune `bm25_weight` up (0.8-0.9) if exact keyword matches matter most. Tune `recency_weight` up (0.2-0.3) if freshness matters more than importance. Adjust `topic_boost` and `keyword_boost` if your ingestion produces good topic/keyword metadata.

**Data type mapping (built into tool schemas):**

| data_type | data_id | Use for |
|-----------|---------|---------|
| SYSTEM | identity | Agent persona, system identity |
| SYSTEM | rules | Behavioral constraints, system rules |
| USER-DATA | preferences | User prefs, communication style |
| USER-DATA | habits | Working style, coding conventions |
| USER-DATA | conventions | Naming, structure, format standards |
| USER-DATA | style | Tone, language, response preferences |
| ENV-DATA | HW | Hardware specs, GPU, RAM |
| ENV-DATA | SW | Software, tools, versions |
| ENV-DATA | NET | Network, proxies, endpoints |
| ENV-DATA | workflow | Operational patterns, kanban, delegation |
| ENV-DATA | conventions | Project conventions, file structure |
| ENV-DATA | troubleshooting | Known issues, fixes, workarounds |
| SESSION-DATA | <session_id> | Per-session context + session_name |
| CUSTOM | folder-name | Obsidian vault notes (Concepts, Entities) |
| CUSTOM | (omit) | Free-form notes |

**Metadata enrichment** — when `data_type`, `data_id`, `topic`, or `keywords` are omitted, HLM auto-enriches using a hybrid approach: keyword heuristics first (zero cost), LLM fallback for low-confidence matches. Controlled by `enrich_on_add` config (`"low_confidence"` default, `"true"` always, `"false"` disabled). Use `layered_enrich()` to retroactively enrich existing records with empty metadata.

## Testing

```bash
# Run test suite (requires Qdrant running)
cd hermes-layered-memory
docker compose up -d
python3 -c "
import sys; sys.path.insert(0, '~/.hermes/plugins/hermes-layered-memory')
from backend import LayeredBackend
# ... or use the test script if available
"
```

See `DEVELOPMENT.md` for testing instructions.

## Troubleshooting

### Qdrant not responding
```bash
docker compose ps    # check container status
docker compose logs  # check for errors
curl -s http://localhost:6333/  # health check
```

### Circuit breaker triggered (Qdrant offline → SQLite fallback)
```bash
# Check Qdrant
curl -s http://localhost:6333/collections
# Restart if needed
docker compose restart qdrant
```

### PyTorch warning: `KernelPreference` deprecation
```
W0611 ... torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass...
```
From `sentence-transformers`/`torch` when loading the embedding model. Harmless — won't affect functionality. Suppress with `export PYTHONWARNINGS="ignore::DeprecationWarning"`.
```

### Sync drift
```bash
# Check
layered_sync_check()
# Fix
layered_rebuild()
```

### Plugin not discovered
```bash
# Verify symlink
ls -la ~/.hermes/profiles/<profile>/plugins/hermes-layered-memory
# Should point to ~/.hermes/plugins/hermes-layered-memory
# Enable
hermes -p <profile> plugins enable hermes-layered-memory
```

### Qdrant upsert fails (point ID format)
The plugin handles UUID format conversion internally. If you see "invalid point ID" errors, ensure you're running the latest code (post-fix for qdrant-client 1.18 compatibility).

## Dependencies

| Package | Purpose |
|---------|---------|
| qdrant-client >= 1.18 | Qdrant vector search client |
| sentence-transformers >= 5.5 | Embedding model (all-MiniLM-L6-v2, 384-dim) |
| Docker + Docker Compose | Qdrant container |

### Embedding Model

**Local (default):** `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) — computed in Python before upserting to Qdrant. ~2s cold start on first call, then cached for the session.

**Remote (enhanced):** Point to any OpenAI-compatible `/embeddings` or Ollama `/api/embed` endpoint via `HERMES_LAYERED_EMBED_URL` and `HERMES_LAYERED_EMBED_MODEL` env vars (see table above). Supports Ollama, llama.cpp, or any service with the same API format.

**Progress bar:** `sentence-transformers` shows `Batches: 100%|...|` on first call. Suppress with `export TRANSFORMERS_VERBOSITY=error`.

**Note:** Changing embedding dimension (e.g. 384 → 4096) requires recreating Qdrant collections with the new vector size.

## Entity Patterns

Named entity extraction during enrichment pulls entity keywords from content (GPU models, software versions, LLM names, hostnames). Patterns are configurable via JSON files in `entity-patterns/` folder.

**Default patterns** (shipped): `default.json` — GPU/CPU models, software versions, LLM models, service hostnames. Do not edit — use as reference.

**Add custom patterns:** Create `.json` files in `entity-patterns/` folder:
```bash
cp entity-patterns/default.json entity-patterns/hardware.json
# Edit hardware.json to add/modify patterns
```

All `.json` files are loaded at startup, merged by pattern name:
- Same name = override default
- New name = add to defaults
- Files loaded alphabetically

**Config override:** Set `entity_patterns_path` in JSON config for a single custom file:
```json
"entity_patterns_path": "/path/to/your/patterns.json"
```

**Pattern structure:**
```json
{
  "patterns": [
    {
      "name": "my_pattern",
      "regex": "\\b(my\\w+)\\s+(\\d+\\.\\d+)\\b",
      "flags": "IGNORECASE",
      "enabled": true,
      "filter": "exclude_stopwords",
      "stopwords": ["the", "and"]
    }
  ]
}
```

**Auto-restore:** If `entity-patterns/` folder is missing/empty, auto-creates with `default.json`. User files survive — only restores if nothing exists.

## Files

```
./
├── __init__.py                 # MemoryProvider ABC impl, 28 tool schemas, dispatch
├── backend.py                  # LayeredBackend: SQLite + Qdrant + 5-layer pipeline + enrichment + compaction + export/import + decay + backup
├── summaries.py                # SummariesBackend: SQLite + FTS5 + .md files (profile-scoped, WAL mode)
├── plugin.yaml                 # Plugin manifest (v0.2.0)
├── docker-compose.yml          # Qdrant container (bind mount)
├── qdrant-data/                # Qdrant storage (gitignored)
├── entity-patterns/            # Entity extraction patterns (JSON)
├── .env.example                # Environment variable template
├── hermes-layered-memory.example.json  # JSON config template
├── tests/                      # Test suites (custom runners)
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Maintenance

Maintenance runs automatically via the `on_session_end` hook in `plugin.yaml`. No manual scheduling required.

**What runs at session end (in order):**
1. `enrich_existing(max_items=20)` — fills missing metadata on recent records
2. `sleep(max_items=50, min_age_hours=0)` — consolidates duplicates, archives low-trust records
3. `purge(purge_deleted=True, purge_archived=False, min_age_hours=24, vacuum=True)` — permanently removes soft-deleted records + Qdrant cleanup
4. `decay(min_age_days=30, max_age_days=365, decay_rate=0.05, min_score=0.1)` — reduces trust_score for stale memories
5. `summaries.sync()` — syncs summaries DB with filesystem
6. `sync_check()` → `rebuild()` if Qdrant index is stale

**Additional hooks:**
- `on_memory_write` — mirrors MEMORY.md/USER.md writes to HLM (if `memory_enabled` is still on)
- `prefetch` — searches memory on each user turn, returns top results as context
- `sync_turn` — spawns background fact extraction every 5 turns (throttled: 1 per 30s)
- `on_pre_compress` — records session in chain, extracts facts before context compression

**Optional: Scheduled cron job for heavy maintenance:**
```bash
hermes cronjob create --schedule "0 3 * * 0" --name "HLM weekly heavy purge" \
  --prompt "Call layered_purge(purge_deleted=true, purge_archived=true, min_age_hours=168), then layered_sleep(max_items=50, min_age_hours=0). Only run if there are records older than 7 days."
```

## Future Concerns

See `proposed_features.md` in the original repository for the comprehensive tracked list.

**All 8 high-priority items implemented:** purge, score threshold, Qdrant cleanup, FTS5 triggers, semantic dedup, schema cleanup, trust feedback, explicit rerank.

Current medium-priority items: entity linking, query expansion, schema field cleanup, Qdrant keyword indexes.

## Known Limitations

- `data_type:data_id` is the natural grouping for consolidation — `layered_sleep()` archives low-trust duplicates within the same group
- Fact extraction (`sync_turn`, `on_pre_compress`, `on_session_end`) spawns background Hermes sessions; rate-limited to 1 per 30s, every 5 turns
- REST API and MCP server wrapper are not yet implemented
- Qdrant collections must exist for all configured data_types (initialized on backend start)
## Documentation

This README is the primary documentation. Additional development documentation (design docs, test plans, audit reports) is available in the original repository.
