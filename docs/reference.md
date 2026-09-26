# API Reference

Full reference for HLM tools, configuration, and environment variables.

## Tool Reference

All 6 meta-tools expose inline documentation with explicit `data_type`/`data_id` mapping. The model sees this in its system prompt — no need to repeat the mapping in every call. Full generated reference: [docs/tools.md](tools.md).

### layered_memory

| Action | Signature | Purpose |
|--------|-----------|---------|
| `retrieve` | `(query, max_layer, scope, data_type, data_id, session_name, profile_name, cross_profile, limit, rerank, status)` | Multi-layer retrieval. `rerank=true` forces L3 even at `max_layer=2`; `status` filters `active` (default) / `deleted` / `all`, and an unrecognised value narrows to `active` rather than widening (0.7.80). **`rerank` is available on MCP `memory_query`; `status` is not** — this row previously said neither was, which was wrong about `rerank`. |
| `peek` | `(query, layer)` | Inspect a specific layer output |
| `add` | `(content, summary, topic, keywords, scope, data_type, data_id, session_name, sensitivity, ttl, priority, force, supersedes)` | Store a memory |
| `update` | `(uuid, ...)` | Update existing memory |
| `delete` | `(uuid)` | Soft-delete (marks as deleted) |
| `list` | `(topic, scope, limit, sort)` | List recent memories |
| `discover` | `(query, limit, min_score)` | Top-N candidates from *other* profiles as metadata only — profile, topic, data_type, data_id, created_at, trust_score, score. No content, summary or keywords cross the profile boundary; follow with `retrieve(cross_profile=true)` to read one. Own records are excluded by database identity, not by profile name. |
| `list_profiles` | `()` | List all profiles with record **and summary** counts. The summary count read 0 for every profile before v0.5.2 — two independent faults, an `UnboundLocalError` swallowed by a bare `except` and a status filter using the memories vocabulary against the summaries table |

**`add` parameters:**
- `force=true` — stores even when dedup/contradiction would block it
- `supersedes=<uuid>` — records that this replaces an existing fact; old record is kept and linked but stops appearing in retrieval. The target must be the current chain head (`status='active'` and not itself already superseded) — otherwise the link is skipped with a warning and the new record is still stored.
- `priority` — controls decay rate: `0`=full, `1`=half, `2`=slow, `3+`=exempt (pinned). ENV-DATA/SYSTEM default to 1. Override with `priority=N`. `protected=true` for permanent exemption.

**Write-payload limits:** `add` and `update` raise `ValueError` on six FTS5-indexed fields — `content` > 50,000 (`MAX_CONTENT_CHARS`), serialized `metadata` > 20,000 (`MAX_METADATA_CHARS`), and `summary`/`topic`/`keywords` > `MAX_FIELD_CHARS` (50,000) plus `data_id` > `MAX_DATA_ID_CHARS` (500). The latter four are bounded because they are written into the FTS5 index, where an unbounded value bloats it exactly as content does; `keywords` is checked both serialized and per entry. Checked before embedding/dedup — see `_check_fts_field_bounds` in `backend/store.py` and `backend/constants.py`.

Three more scalars are bounded separately by `_check_scalar_field_bounds` because they are not FTS5-indexed but are still stored verbatim and echoed on every read: `session_name` > `MAX_SESSION_NAME_CHARS` (500), `scope` > `MAX_SCOPE_CHARS` (200), `source_url` > `MAX_SOURCE_URL_CHARS` (2048). `add()` checks all three; `update()` only checks `session_name` — `scope` and `source_url` are not in `update()`'s field allowlist, so a caller cannot set either through it and there is nothing there to bound.

**Limit clamping:** `retrieve` clamps `limit` to 1..200, `list` clamps `limit` to 1..500. Out-of-range or non-numeric values are silently bounded (non-numeric falls back to the default), not honored or rejected.

### layered_maintenance

| Action | Signature | Purpose |
|--------|-----------|---------|
| `rebuild` | `(since)` | Rebuild Qdrant index from SQLite |
| `purge` | `(purge_deleted, purge_archived, min_age_hours, min_age_hours_deleted, min_age_hours_archived, vacuum)` | Permanently delete soft-deleted/archived records + Qdrant cleanup. Defaults: `purge_deleted=true`, `purge_archived=true`, `vacuum=true`. All three accept a boolean or the string forms a provider sends (`true`/`false`/`1`/`0`/`yes`/`no`); **anything else is refused** rather than read as false, because a purge that deletes nothing and reports success is worse than an error (0.8.71, T645). |
| `sync_check` | `()` | Check Qdrant ↔ SQLite sync |
| `sleep` | `(max_items, min_age_hours, archive_age_days)` | Archive low-trust duplicates and enforce TTL. `min_age_hours` (default 24) is a grace period — a duplicate is archived only if its `updated_at` is older than that. **`protected` records are never archived by any of `sleep()`'s three branches**, TTL expiry included (0.7.76), and `decay()` skips them too. It guards *automatic* maintenance, not deliberate calls: an explicit `update(status="archived")` still archives a protected record, because the caller has said so directly. Archiving now retires the record's Qdrant point and releases its supersessions, so `sync_check` stays truthful (0.7.77); the count is reported as `retired_points`. Lightweight — no LLM calls. Content merging is `compact()`. |
| `review` | `(min_age_hours, force, execute)` | LLM review: classify records as keep/delete. **Dry-run by default** — nothing is soft-deleted without `execute=true`. Runs as a direct in-process LLM call; it has not spawned a background session since the autonomous-spawn removal. |
| `decay` | `(min_age_days, max_age_days, decay_rate, min_score)` | Confidence decay: reduces trust_score for stale memories. Integrated into session-end maintenance. |
| `test_cleanup` | `()` | Delete test records |
| `resolve_conflicts` | `(execute)` | Resolve conflicting records: keeps highest trust_score (newest on a tie), soft-deletes the rest. **Dry-run by default** — nothing is soft-deleted without `execute=true`. |

**`purge` age thresholds:** deleted and archived records have independent thresholds, defaulting to `cleanup.purge_deleted_hours` (24h) and `cleanup.purge_archived_hours` (168h/7d). `min_age_hours_deleted` / `min_age_hours_archived` override one category each and take precedence. The legacy shared `min_age_hours` sets BOTH — passing it alone collapses the 7-day archived safety margin down to that value, so prefer the category-specific parameters when tuning only one.

### layered_summaries

| Action | Signature | Purpose |
|--------|-----------|---------|
| `summarize` | `(source, source_type, title, highlights, full_text, tags, force)` | Store a summary (idempotent by source_url) |
| `list` | `(source_type, source_url, tag, limit, offset, sort_by)` | List stored summaries (filtered count) |
| `get` | `(uuid)` | Get summary by UUID (returns highlights + path to .md) |
| `search` | `(query, limit)` | Search summaries via FTS5 (snippet + rank) |
| `update` | `(uuid, title, highlights, tags, metadata, full_text)` | Update a summary (keeps same UUID) |
| `delete` | `(uuid)` | Delete summary + .md file |
| `batch_delete` | `(uuids)` | Batch delete. Returns `deleted` / `not_found` / `refused` / `invalid` counts — four outcomes, because they mean four different things to whoever is auditing a batch. `refused` is a row owned by another profile; `invalid` is an element that is not a uuid string, which until 0.8.74 was reported as `not_found` (an int) or raised `sqlite3.ProgrammingError` and aborted the whole batch (a list). One bad element never refuses the good ones beside it (T664). |
| `list_expiring` | `(max_age_days)` | List old summaries (TTL cleanup) |

**Every summaries action takes `profile`, and it is a two-word vocabulary:
`"own"` (default) or `"all"`.** Anything else is refused by `_profile_sql` /
`_row_profile_ok` — the two helpers all nine backend doors cross — with a
warning naming what to pass. Until 0.8.74 only `sync` checked: the other eight
read any unknown word as if the caller had said `"own"`, so `profile="al"`
answered with the caller's own rows and no indication that the cross-profile
read it asked for had not happened (T663).
| `sync` | `(profile)` | Sync DB with filesystem: clears rows whose `.md` file is gone. Only **this profile's** rows — a foreign row's `summary_path` is relative to *its* profile's directory, so it cannot be resolved here and is reported as `skipped_foreign` rather than deleted (0.7.76). A row still holding `full_text` keeps the record and clears the path instead. |

**A filter that reaches a SQL statement must be a string, on both doors.**
`delete_many`'s `content_like` / `data_type` / `data_id` / `source`,
`graph_health`'s `data_type`, `get_taxonomy`'s `filter`, and the summaries
`list` filters (`source_type`, `source_url`, `tag`, `sort_by`) are refused with
a message naming the parameter. Until 0.8.75 they were bound straight into the
query on the plugin door, so a list or dict raised
`sqlite3.ProgrammingError: type 'list' is not supported` and an int matched
nothing and was reported as success — on `delete_many`, as a filter that
matches nothing; on `graph_health`, as a healthy graph. The MCP twin types them
through its pydantic annotations and always refused. One message,
`backend/constants.py`'s `str_filter_error` (`T668`).

**Summaries storage:** `digests.db` (SQLite + FTS5, WAL mode) + `.md` files in `~/Documents/hlm-summaries/` (env: `HLM_SUMMARIES_DIR`).

**Data extraction:** The `markitdown` package (Hermes-level, not an HLM dependency) can convert PDF/DOCX/PPTX/YouTube/etc. to Markdown before summarizing.

### layered_advanced

| Action | Signature | Purpose |
|--------|-----------|---------|
| `enrich` | `(since, max_items)` | Batch-enrich existing memories with empty metadata |
| `feedback` | `(uuid, helpful)` | Increments/decrements trust_score (+/-0.1). Exactly ±0.1 even on a prefetch-injected record — the implicit +0.02 reinforcement is not stacked on top. |
| `compact` | `(similarity_threshold, topic, max_groups, execute)` | Merge duplicate/overlapping memories via LLM. **Dry-run by default** — `dry_run` is a deprecated alias; the parameter is `execute`. Corroboration boost for trust score. `similarity_threshold` defaults to **0.90** on every door; it read 0.85 in three of five places until 0.8.71, including the plugin's own published schema, and lower means more pairs count as similar (`_C.COMPACT_SIMILARITY_DEFAULT`, pinned by T648). A merged record's embedding is written to **SQLite as well as Qdrant** — until 0.8.76 `_embed_and_upsert` wrote Qdrant alone, leaving every merged record invisible to the five readers that take their vectors from SQLite, `_brute_force_search` (the Qdrant-outage fallback) and `graph_health` among them (`T670`). `max_groups` must be **positive on both doors** — the plugin accepted `0`/negative and reported a completed compaction over a bound the backend had silently replaced with its 200-record seed floor, while MCP refused the same value by name (0.8.74, T661). **`protected` records are never merged** (0.8.113, T711): the seed scan carries the same `COALESCE(protected, 0) = 0` guard as `sleep()` and `decay()`, because compact names a threshold and a group cap and never a uuid — it is automatic maintenance, the guarded side of the line drawn in the `sleep` row above. Before the fix a pinned record was soft-deleted and its content folded into a new uuid that merely inherited the flag. |
| `traces` | `()` | List recent retrieval traces |
| `reenrich` | `(topic_only, keyword_only, limit)` | Backfill missing topic/keywords on existing records using LLM. |
| `stats` | `()` | Retrieval usage metrics: prefetch (auto-hook) vs explicit (agent-initiated). An unreadable extraction ledger leaves `extraction` present and **null** rather than dropping the key — a caller reading `stats["extraction"]["runs"]` got a `KeyError` of its own with the only trace at DEBUG (0.8.74, T662; the MCP door has answered `None` since T649). |
| `graph_health` | `(threshold, limit, data_type)` | Records that cluster with nothing — nearest neighbour below `threshold` (default 0.5), loneliest first. Either a fact captured once and never reinforced, or one worded so unlike any question that retrieval will never surface it. Read-only in both senses: it also registers **no session tags**, so a uuid it surfaces does not become updatable or deletable by an agent that never retrieved the record. It did until 0.8.74 — `_register_uuid` writes the map the seen-UUID gate reads, so one diagnostic call unlocked every uuid in its report (T659). The scan is quadratic so `limit` caps it (200, newest first) and the result reports `truncated`. |

### layered_io

| Action | Signature | Purpose |
|--------|-----------|---------|
| `obsidian_ingest` | `(vault_path)` | Ingest Obsidian vault notes. `vault_path` must resolve inside an allowed root (see `HLM_OBSIDIAN_VAULT_ROOTS`) or a `ValueError` is raised. |
| `export` | `(format, status, data_type, topic, scope, profile_name, cross_profile)` | Export memories as JSON or Markdown |
| `import` | `(data, mode, target_status)` | Import memories from JSON (or a `.json`/`.md` file path in `data`, contained by `HLM_IMPORT_ALLOWED_ROOTS`). Modes: `skip_existing`, `overwrite`, `new_uuid`. |
| `backup` | `(dest_dir, keep_days)` | Atomic SQLite backup of both memories and summaries DBs. WAL-safe. Default: 7-day retention. |

## Configuration

### Data Type Mapping

| data_type | data_id | Use for |
|-----------|---------|---------|
| SYSTEM | identity | Agent persona, system identity |
| SYSTEM | rules | Behavioral constraints, system rules |
| USER-DATA | preferences | User prefs, communication style |
| USER-DATA | habits | Working style, coding conventions |
| USER-DATA | conventions | Naming, structure, format standards |
| USER-DATA | style | Tone, language, response preferences |
| ENV-DATA | hw | Hardware specs, GPU, RAM |
| ENV-DATA | sw | Software, tools, versions |
| ENV-DATA | net | Network, proxies, endpoints |
| ENV-DATA | workflow | Operational patterns, kanban, delegation |
| ENV-DATA | conventions | Project conventions, file structure |
| ENV-DATA | troubleshooting | Known issues, fixes, workarounds |
| SESSION-DATA | \<session_id\> | Per-session context + session_name |
| OBSIDIAN | folder-name | Obsidian vault notes (via `layered_io(action="obsidian_ingest")`) |
| CUSTOM | (omit) | Free-form notes |

### Environment Variables

Add to `~/.hermes/profiles/<profile>/.env` (profile-specific) or `~/.hermes/.env` (global). Loaded automatically on session start.

| Env var | Default | Purpose |
|---------|---------|---------|
| `HLM_DB_PATH` | `~/.hermes/hermes-layered-memory-dbs/<profile>.db` | SQLite database path (auto-created). |
| `HLM_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (REQUIRED). |
| `HLM_QDRANT_ENABLED` | `true` | Set `false` for SQLite-only mode (no Qdrant). Dedup bypassed when offline. |
| `HLM_QDRANT_COLLECTION_SUFFIX` | `true` | Key collection names to the embedding model — `memories` becomes `memories_<model>_<dim>`. A Qdrant collection has one fixed vector size, so a shared name breaks every upsert the moment the embedder changes; a keyed name means the new model gets a new collection and the old one keeps serving until it is rebuilt. Keyed on the *model*, not the dimension: two different 4096-dim models share a size but not an embedding space. Set `false` to keep flat names (and the collision). |
| `HLM_QDRANT_TIMEOUT` | `30` | HTTP timeout (seconds) for the Qdrant client. Without it the underlying httpx default applies (version-dependent, currently 5s) — invisible, and often too short for the large batch upserts done by `layered_maintenance(action="rebuild")`. |
| `HLM_DEDUP_THRESHOLD` | _(none)_ | Override semantic dedup threshold. Must be in [0.0, 1.0]. Default 0.97 from JSON config. |
| `HLM_QUERY_INSTRUCTION` | _(none)_ | Query-side instruction prefix for asymmetric embedding models. Off by default. |
| `HLM_LAYER0_TOP_K` | `30` | Max candidates per data_type collection returned by Layer 0 (Qdrant/FTS5). Caps the input to Layer 2 fusion scoring. Lower values reduce CPU at the cost of recall. |
| `HLM_MAX_LAYER` | `2` | Default max layer depth: 0=Qdrant, 1=+SQLite, 2=+fusion, 3=+LLM reranker, 4=+gap detect. |
| `HLM_EMBED_URL` | _(none)_ | Remote embedding endpoint (Ollama `/api/embed`, etc.) |
| `HLM_EMBED_MODEL` | _(none)_ | Embedding model name (e.g. `qwen3-embedding:8b`) |
| `HLM_LOCAL_EMBED_MODEL` | _(none)_ | Local FastEmbed model (no Ollama needed). Set `intfloat/multilingual-e5-large` for 1024-dim multilingual. Default: `sentence-transformers` MiniLM (384-dim). |
| `HLM_LAYER3_MODE` | `inline` | Reranker mode: `inline` (single call), `delegate` (sub-agent), `self` (in-context). Only if `max_layer >= 3`. |
| `HLM_LAYER3_MODEL` | _(none)_ | LLM model for Layer 3 reranker. Only if `max_layer >= 3`. |
| `HLM_LAYER3_BASE_URL` | _(none)_ | Base URL for Layer 3 LLM endpoint (OpenAI-compatible). |
| `HLM_LAYER3_API_KEY` | _(none)_ | API key for Layer 3 LLM endpoint. |
| `HLM_REASONING_EFFORT` | `"none"` | Reasoning budget for all HLM LLM calls. **Defaults to disabled** — every call HLM makes (rerank, classify, merge, extract) is JSON formatting, where reasoning costs seconds and buys nothing. `"low"`/`"medium"`/`"high"`/`"minimal"`/`"xhigh"`/`"max"`/`"ultra"` request a budget instead; `"provider_default"` (or boolean `true`) sends nothing and lets the model decide. Every spelling of off — `"none"`, `"false"`, `"off"`, `"no"`, `"disabled"`, boolean `false`, empty — disables, and an unrecognized value warns and disables rather than being forwarded. Until 0.7.67 anything outside `"none"`/`""` was passed straight through as `reasoning_effort=<that string>`, so writing `false` emitted an invalid enum: the endpoint rejected it, the repair path stripped it, and the call ran with no reasoning control at all — the opposite of what was asked, cached for the process. The level vocabulary matches Hermes' `agent.reasoning_effort` so one config teaches the other. Before v0.3.0 an unset value sent nothing, so a reasoning model reasoned, returned `content: null`, and every LLM-backed feature silently no-opped — the failure T189 was written for. |
| `HLM_REASONING_STYLE` | `"auto"` | Which spelling of "don't think" to send. **Read since 0.8.72 only** — this row described the variable in full, with a default and two instructions to set it, while nothing in the code read it; `layer3_reasoning_style` was settable only through `layered_config` or the JSON file. An unrecognised value now warns and is ignored rather than falling through to auto-detection (T656).  Run `scripts/measure-reasoning-suppression.py` against your endpoint before changing it — on a Qwen/vLLM measured 2026-08-20 only two of the five permissive keys did anything (`reasoning_effort` and `chat_template_kwargs.enable_thinking`), and the other three were accepted and ignored, which is indistinguishable from working unless you count tokens. `"auto"` picks by endpoint host. `"openai"` sends only `reasoning_effort` (strict APIs). `"permissive"` sends the full superset. `"off"` sends nothing. Only needed when auto-detection guesses wrong about a proxy. |
| `HLM_ENRICH_LLM` | `false` | Enable LLM fallback for topic/keywords during enrichment. Default `false` (heuristic-only, instant). Set `true` to run LLM classification when heuristics fill data_type/data_id but leave topic/keywords empty. Costs ~1-2s and ~100 tokens per add with missing metadata. |
| `HLM_LOW_TRUST_ARCHIVE_DAYS` | `365` | Days before low-trust (<0.3) records are auto-archived by `layered_maintenance(action="sleep")()`. Records with priority >= 2 are never auto-archived. Takes precedence over the `cleanup.archive_age_days` JSON key. |
| `HLM_CONFLICT_THRESHOLDS` | _(JSON defaults)_ | JSON string: `cosine_min`, `jaccard_min`, `jaccard_max`, `temporal_guard_days`, `source_guard_days`. Defaults: `{"cosine_min": 0.85, "jaccard_min": 0.7, "jaccard_max": 1.0, "temporal_guard_days": 1, "source_guard_days": 30}`. Lower `cosine_min` for more sensitive conflict detection. |
| `HLM_OBSIDIAN_VAULT_ROOTS` | _(real home dir)_ | Colon-separated absolute paths where `layered_io(action="obsidian_ingest")` may read vaults from. Path-traversal containment: a `vault_path` resolving outside all allowed roots is rejected with a `ValueError`. Defaults to the real home directory (resolved via `pwd`, not `$HOME`) — set this for vaults stored elsewhere. |
| `HLM_IMPORT_ALLOWED_ROOTS` | _(real home dir)_ | Colon-separated absolute paths where `layered_io(action="import")` may read a file from. The `data` argument is treated as a file path when it ends in `.json`/`.md`; same containment rationale as above (returns an error result rather than raising). |
| `HLM_LOG_FILE` | `<hermes-home>/logs/hermes-layered-memory.log` | Log file path |
| `HLM_TRACING` | _(unset)_ | Enable query-path tracing (opt-in). Set `true` to record each retrieval's layer steps, duration, and top match to `{db_dir}/.hlm-traces/`. Use `layered_advanced(action="traces")` to list recent traces. Disabled by default to avoid disk I/O. |
| `HLM_SEED_OVERVIEW` | _(unset)_ | Inject knowledge base overview into system prompt at session start. Shows record groups, counts, avg trust, and top trusted memory. Set `true` to enable. Helps the agent understand what's in memory before searching. |
| `HLM_LOG` | `INFO` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |
| `HLM_SUMMARIES_DIR` | `~/Documents/hlm-summaries/` | Summaries .md files directory |
| `HLM_SUMMARIES_DB` | `~/.hermes/hermes-layered-memory-dbs/digests.db` | Shared summaries SQLite DB (all profiles write to the same file; WAL mode handles concurrency) |

### JSON Config

Config file: `~/.hermes/profiles/<profile>/hermes-layered-memory.json`

| Key | Default | Purpose |
|-----|---------|---------|
| `enrich_on_add` | `"heuristics_only"` | Enrichment mode at add-time. `"heuristics_only"` (default) uses keyword heuristics only — no LLM call, no background thread. `"low_confidence"` runs the LLM only when heuristic confidence < 0.6. `"true"` always runs it — note this is a **cost increase** relative to pre-v0.3.0, where `"true"` and `"low_confidence"` were identical and both only fired on low confidence; use `"low_confidence"` if you want that older behaviour. `"false"` is a synonym for `"heuristics_only"`. An unrecognised value is rejected by `layered_config(action="set")` rather than silently falling through to "no enrichment". Before v0.3.0 this key was read only inside `_enrich_metadata`, whose callers all pass an explicit `llm=` flag that overrode it — so it had no effect and background enrichment ran on every write. It now gates `_enrich_background` directly. A profile that never set the key gets a one-time WARNING at startup saying enrichment has stopped, so the change is not discovered weeks later via an empty `topic` column. Separately, `"true"` and `"low_confidence"` used to be indistinguishable — a high-confidence heuristic short-circuited before the mode was consulted — and now differ as documented. |
| `min_bm25_threshold` | `0.0` | Drop results below BM25 threshold. Default `0.0` (disabled). Set `0.1` to filter out weak matches (BM25=0 noise). Applied in `_layer1()` after BM25 scoring. |
| `dedup_threshold` | `0.97` | Semantic dedup at add-time. Compares new content against existing memories via vector similarity. Default `0.97` (very high bar — avoids false positives on distinct content about same topic). Set `0.0` to disable. On duplicate: returns existing UUID with similarity score, does NOT store. Agent can use `layered_memory(action="update")` to merge content. |
| `query_expand` | `false` | BM25 query expansion. When true, expands BM25 query with keywords from top-5 vector matches. Improves cross-vocabulary recall (e.g., "GPU" → picks up "NVIDIA", "CUDA", "video memory"). Set `true` for broader recall. |
| `layer3_reasoning_effort` | `"none"` | See `HLM_REASONING_EFFORT` above. |
| `layer3_reasoning_style` | `"auto"` | See `HLM_REASONING_STYLE` above. |
| `layer3_timeout_seconds` | `120` | Per-attempt timeout (seconds) for LLM calls (L3 rerank, L4 gap detection, classification, enrichment, merge). `_call_llm()` retries up to 3 times with exponential backoff, so the worst case is roughly 3x this plus backoff (~366s). Was `30`, which a reasoning model cannot meet — an unsuppressed Qwen3 call measured 72.3s, so every attempt timed out. Lower it if you need a tighter latency bound and are not running a reasoning model. No `HLM_*` env var — settable only via `hermes-layered-memory.json` or `layered_config`. |
| `entity_patterns_path` | _(unset)_ | Override entity patterns file path. Default: all `.json` files in `backend/entity-patterns/` folder. Set to a single file path for full replacement. |
| `auto_extract` | `false` | Enable automatic fact extraction from conversations (session end and pre-compress). **Off by default because the model decides what counts as durable, not because of a laundering path.** `source="extraction"` was removed from `SELF_AUTHORED_SOURCES` in v0.4.1, so extracted records *are* fenced on every later retrieval, and `source="compaction"` followed in v0.7.53. This paragraph claimed the opposite for many releases — the same stale sentence was corrected in `docs/architecture.md` in 0.7.72 and survived here, which is the third place it has had to be fixed. The real reason to opt in deliberately is judgement: the input conversation is fenced going into the prompt, but what the model chooses to store is its own call. Since 0.7.74 the self-loop shield also stops it re-learning records retrieval surfaced in the same session. |
| `auto_extract_per_turn` | `false` | Enable per-turn extraction (every 5th turn) in addition to session end. Requires `auto_extract`. Off even when `auto_extract` is on: mid-conversation is the hardest place to tell a durable fact from scratch state, and each run writes fence-exempt records. |
| `dedup_exact_scan` | `false` | Restore the pre-v0.3.0 behaviour of running the exact SQLite vector scan on *every* write, not only when Qdrant could not answer. Catches a duplicate the vector index has not caught up on yet, at the cost of unpacking every active embedding of that data_type per write (O(corpus) on the hot write path — a large part of why bulk ingest tripped the circuit breaker). |

### Scoring Weights

All weights are configurable in `hermes-layered-memory.json` under `scoring`. Defaults are baked into the code — config section is optional.

**These defaults are measured, not chosen.** They were 6x larger (bm25 0.7,
topic 0.3, the rest 0.15) and at those magnitudes Layer 2 ranked *worse than
Layer 1* — worse than plain vector order — on both embedders. The fusion base is
a normalised reciprocal rank that compresses the whole vector ordering into a
~0.33-wide band, so a 0.7 BM25 term reorders it at will. Measured on 540 records
/ 61 queries (production embedder, recall@5 / hard@5 / dense@5 / MRR),
measured with the lexical arm running on every query (0.7.24):

    all boosts zeroed         0.869 / 0.897 / 0.5 / 0.758   = L1 exactly
    bm25 0.15                 0.869 / 0.897 / 0.5 / 0.737
    bm25 0.30 (current)       0.885 / 0.897 / 0.6 / 0.735   <- the knee
    bm25 0.50                 0.836 / 0.793 / 0.6 / 0.670
    bm25 0.70                 0.869 / 0.828 / 0.7 / 0.669
    bm25 0.85 (0.7.23)        0.852 / 0.759 / 0.8 / 0.674
    bm25 1.00                 0.852 / 0.724 / 0.9 / 0.677

`hard@5` bounds it and the cliff is steep: 0.897 → 0.793 in one step from 0.30
to 0.50. The band is 0.15-0.30, down from 0.6-0.85 in 0.7.23 — that derivation
measured a term firing on a minority of queries, this one a term firing on all
of them, and the two are not comparable. T332 pins the values and carries the
table; re-measure with `tests/eval_retrieval.py --spread` before changing them.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm25_weight` | 0.30 | Lexical match importance (FTS5 BM25). Higher = exact keyword matches dominate. Lower = vector similarity matters more. **Measured, not chosen** — see the derivation above; 0.30 is the knee. |
| `importance_weight` | 0.05 | Influence of `trust_score` (0-1). Higher = trusted memories surface more. |
| `recency_weight` | 0.05 | Influence of freshness. Higher = newer memories favored. Pinned items (priority=3) exempt from decay. |
| `topic_boost` | 0.05 | Flat bonus when query words appear in `topic` field. Use to prioritize topic-based retrieval. |
| `keyword_boost` | 0.05 | Flat bonus when query words appear in `keywords` array. |
| `pinned_boost` | 1.5 | Multiplier for priority=3 memories. Reduces to 1.0 to disable pinning. |
| `recency_half_life_days` | 21 | Days until a memory's recency score drops by half. Lower = memories age faster. |
| `rrf_constant` | 60 | RRF (Rank Fusion) constant. Higher = rank position matters less, scores more uniform. Lower = top-ranked memories dominate. |

**Do not tune `bm25_weight` up to 0.8-0.9.** This line used to advise exactly
that, three paragraphs below a measurement showing those magnitudes rank *worse
than plain vector order* — recall 0.852 and MRR 0.674 at 0.85, against 0.885 /
0.735 at the 0.30 default. 0.7.24 lowered it from 0.85 for that reason. If
exact keyword matches matter more for your corpus, re-measure with
`tests/eval_retrieval.py` rather than reaching for a larger number; the fusion base is a
normalised reciprocal rank in a ~0.33-wide band, so a large lexical term
reorders the whole vector ordering at will.

`recency_weight` and the two boosts are likewise measured at 0.05. Adjust
`topic_boost` and `keyword_boost` only if your ingestion produces genuinely
good topic/keyword metadata, and check the result against `tests/eval_retrieval.py`.

### Cleanup Configuration

All cleanup thresholds are configurable in `hermes-layered-memory.json` under `cleanup`. Defaults work without it.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duplicate_archive_trust` | `0.2` | Archive records below this trust within a data_type:data_id group. Lower = keep more candidates. |
| `archive_age_days` | `365` | Days before low-trust (<0.3) records are auto-archived. Honored by `layered_maintenance(action="sleep")`. Precedence: `HLM_LOW_TRUST_ARCHIVE_DAYS` (if set) wins over this key. |
| `purge_deleted_hours` | `24` | Hours before soft-deleted records are permanently removed. Default threshold for `layered_maintenance(action="purge")` as well as background maintenance. |
| `purge_archived_hours` | `168` | Hours before archived records are purged (if `purge_archived=true`). Default threshold for `layered_maintenance(action="purge")` as well as background maintenance. |
| `decay_min_age_days` | `30` | Only decay records older than this. |
| `decay_max_age_days` | `365` | Age at which full decay effect applies. |
| `decay_rate` | `0.05` | Maximum decay factor per full age cycle. |
| `decay_min_score` | `0.1` | Floor — scores never drop below this. |
| `enrichment_batch` | `20` | Records to enrich per maintenance cycle. |
| `maintenance_after_inserts` | `10` | Run maintenance only after this many new records (event-driven). |
| `vacuum_free_page_pct` | `20` | Only VACUUM if free pages exceed this percentage. |
| `maintenance_budget` | `2` | Seconds per operation before the overrun is logged (DEBUG). |
| `maintenance_total_budget` | `10` | Seconds total for a session-end maintenance cycle. Remaining operations are skipped once exceeded. |
| `automatic` | `false` | Master switch for unattended mutation at session end. **Off by default.** When false, `enrich_existing()`, `sleep()` and `decay()` do not run at teardown — all three mutate on a heuristic with no operator present, and each has been the site of a data-loss defect. `purge()` is deliberately *not* gated on this: with `purge_archived=False` it only collects rows already soft-deleted past `purge_deleted_hours`, which is garbage collection rather than a judgement, and it is what bounds database growth. |

**Destructive operations are dry-run by default.** `layered_advanced(action="compact")`,
`layered_maintenance(action="resolve_conflicts")` and `layered_maintenance(action="review")`
report what they would change and alter nothing unless called with `execute=true`.
The same applies to the MCP `memory_write(action="maintenance")` endpoint, which is
unauthenticated. Inspect the dry-run report, then re-run with `execute=true`.

**Decay priority curve:** Priority controls decay rate (not a binary skip). `0`=1.00x (full), `1`=0.50x (half), `2`=0.20x (slow), `3+`=0.00x (exempt/pinned). `protected=True` records never decay. ENV-DATA/SYSTEM default to `priority=1` (half decay).

### Embedding Model

**Configuration is per-profile.** The embedding endpoint and model are set in the
active profile's `.env` file (`~/.hermes/profiles/<profile>/.env`), not in
shared config. Under the CLI path (`hermes -p <profile>`), these vars land in
`os.environ` before the process starts. Under a multiplex gateway, they go
through the secret scope — see `docs/profile-isolation.md`.

**Local (default):** `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) —
computed in Python before upserting to Qdrant. ~2s cold start on first call,
then cached for the session. **Limitation:** `all-MiniLM-L6-v2` truncates at
512 tokens. Long documents (Obsidian notes, pasted code) will be silently
truncated, producing inaccurate embeddings. Use a longer-context model via
`HLM_EMBED_URL`/`HLM_EMBED_MODEL` or
`HLM_LOCAL_EMBED_MODEL` if ingesting long content.

**Local (enhanced):** `fastembed` (intfloat/multilingual-e5-large, 1024-dim) — set `HLM_LOCAL_EMBED_MODEL=intfloat/multilingual-e5-large` in `.env`. Multilingual support (100+ languages), higher precision. Requires `fastembed` package (already shipped).

**Remote (enhanced):** Point to any OpenAI-compatible `/embeddings` or Ollama `/api/embed` endpoint via `HLM_EMBED_URL` and `HLM_EMBED_MODEL` env vars (see table above). Supports Ollama, llama.cpp, or any service with the same API format.

**Priority:** Remote endpoint > FastEmbed (if set) > sentence-transformers (fallback).

**Progress bar:** `sentence-transformers` shows `Batches: 100%|...|` on first call. Suppress with `export TRANSFORMERS_VERBOSITY=error`.

**Note:** Changing embedding dimension (e.g. 384 → 1024) requires recreating Qdrant collections with the new vector size. On startup, the backend checks existing collection dimensions and logs a WARNING if mismatched. Run `layered_maintenance(action="rebuild")()` to fix — it re-embeds all records and upserts with the new dimension.

### Entity Patterns

Named entity extraction during enrichment pulls entity keywords from content (GPU models, software versions, LLM names, hostnames). Patterns are configurable via JSON files in `backend/entity-patterns/` folder.

**Default patterns** (shipped): `default.json` — GPU/CPU models, software versions, LLM models, service hostnames. Do not edit — use as reference.

**Add custom patterns:** Create `.json` files in `backend/entity-patterns/` folder:
```bash
cp backend/entity-patterns/default.json backend/entity-patterns/hardware.json
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

**Auto-restore:** If `backend/entity-patterns/` folder is missing/empty, auto-creates with `default.json`. User files survive — only restores if nothing exists.

### Summaries (SQLite only — intentionally no Qdrant)

Summaries are stored separately from layered memory. Lookup-based search via FTS5 with Porter stemmer (not semantic similarity). Full summaries saved as portable `.md` files. Deep search indexes full_text in FTS5. Lifecycle metadata (status, coverage_status, token_estimate) + content_hash for freshness detection. URL canonicalization (YouTube, GitHub) + idempotency (already_exists). FTS5 health check at startup + auto-rebuild on corruption. Relative paths for portability. Tag normalization (lowercase, hyphens).

**Storage model (dual copy):**
- **DB (`digests.db`):** Stores `full_text` for FTS5 deep search indexing. Without this, FTS5 can only search highlights/title/tags, missing content buried in the summary body.
- **`.md` files:** Portable export, human-readable, version-control friendly. This is the primary read path — `layered_summaries(action="get")()` returns the `.md` file path, not the `full_text` inline. The agent reads `.md` via `read_file(offset, limit)` for context budget control.
- **Why both?** The DB copy enables fast full-text search. The `.md` copy enables portability, offline access, and controlled reading. The redundancy is intentional — search needs DB, reading needs files.

### Defaults and Parameter Reference

**Default limit:** `layered_memory(action="retrieve")` returns 5 results by default; `limit` is clamped to 1..200. `layered_memory(action="list")` defaults to 20 and clamps `limit` to 1..500.

**Write-payload limits:** `layered_memory(action="add"|"update")` raises `ValueError` above 50,000 chars of `content`, 20,000 of serialized `metadata`, 50,000 each for `summary`/`topic`/`keywords` (serialized and per entry), or 500 for `data_id`.

**Parameter defaults:** `scope`, `data_type`, `data_id`, `session_name` default to `None` (no filter). When `None`, the agent searches across all partitions. The agent chooses filters based on query context — general questions get unfiltered searches, specific questions get targeted filters (e.g. `data_type=SYSTEM` for rules, `data_id=preferences` for user prefs). You can instruct the agent to use specific filters in prompts.

### Metadata enrichment

When `data_type`, `data_id`, `topic`, or `keywords` are omitted, HLM auto-enriches using a hybrid approach: keyword heuristics first (zero cost, synchronous), LLM fallback for low-confidence matches (asynchronous, runs in background after `add()` returns). Controlled by `enrich_on_add` config (`"heuristics_only"` default — heuristics only, no LLM and no background thread; `"low_confidence"` for LLM on low-confidence matches; `"true"` always). Use `layered_advanced(action="enrich")()` to retroactively enrich existing records with empty metadata.

**Note:** Because LLM enrichment runs asynchronously, a query immediately after `add()` may see heuristic-only metadata (topic/keywords) rather than the final LLM-enriched values. This resolves on the next query once the background thread completes. Lost enrichment jobs on agent restart are harmless — records still function with heuristic metadata.

### Reasoning / "thinking" suppression across providers

Every LLM call HLM makes is functional — rank these, classify this, merge those,
emit JSON. None of it benefits from a reasoning budget, and paying one costs
seconds on a path that runs per retrieval. Suppressing it portably is harder
than it looks, because providers disagree on two axes at once:

**Different spellings.** OpenAI uses `reasoning_effort` (values `minimal`,
`low`, `medium`, `high` — note that `none` is *not* valid and is rejected).
vLLM serving a Qwen3 chat template needs
`chat_template_kwargs: {"enable_thinking": false}`. Ollama uses `think`.
OpenRouter uses `reasoning: {"exclude": true}`.

**Different tolerance for unknown keys.** OpenAI 400s on an unrecognised body
parameter. Most local servers (vLLM, llama.cpp, LM Studio) ignore them.

Note that "permissive" means *unknown* keys are ignored, not that known keys go
unvalidated. A live vLLM 0.11 accepts all five spellings below, but validates
`reasoning_effort` against its own enum and returns a pydantic `literal_error`
for a value outside it — while still honouring `chat_template_kwargs`.

That combination rules out the obvious approach of trying each spelling until
one stops erroring, because the two failure modes are not symmetrical:

| | strict provider | permissive provider |
|---|---|---|
| wrong key sent | **400, loud** | accepted and ignored — **silently ineffective** |

A trial loop stops at the first variant that does not error, which on a
permissive server is the first variant it *ignores*. It looks like success while
the model keeps thinking. So HLM chooses the spelling **by host** and only falls
back on error:

1. `api.openai.com` gets exactly the parameter its API documents
   (`reasoning_effort: minimal`). It is the only special case, because it is the
   only one that can be justified: its enum is documented, and getting it wrong
   is expensive — the superset would 400 on an unknown key, the repair below
   would strip all reasoning control, and the model would then reason at its
   default on every call, silently and cached.
2. Everything else — local vLLM, llama.cpp, LM Studio, Ollama, OpenRouter,
   proxies, unknown hosts — gets the superset. Each key in it was sent to a live
   vLLM and a live Ollama, together and individually, and all were accepted. An
   untested host is deliberately not a special case: it is the default, and the
   repair below covers being wrong about it.
3. If the endpoint rejects the payload anyway with a 4xx that **names one of
   those parameters**, HLM strips *only the keys the error named*, retries, and
   caches that decision per `(endpoint, model)`. Only the named ones, because a
   server can refuse `reasoning_effort` on enum grounds while still honouring
   `chat_template_kwargs` — dropping both would discard suppression that was
   about to work. A generic 400 (context length, bad model name) is deliberately
   *not* treated as a parameter problem, even when it happens to mention one.

Set `HLM_REASONING_STYLE` to pin a spelling if auto-detection guesses wrong
about a proxy.

#### Picking a value, and what your endpoint actually accepts

`reasoning_effort` is the one key servers **validate**, so its accepted values
vary by server. vLLM 0.11 accepts:

    'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'

OpenAI's is narrower — `minimal`, `low`, `medium`, `high`; `none` is invalid
there and 400s. Rather than guess, ask the server: send a deliberately invalid
value and read the enum back out of the error.

```bash
curl -s "$HLM_LAYER3_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HLM_LAYER3_API_KEY" \
  -d '{"model": "'"$HLM_LAYER3_MODEL"'",
       "messages": [{"role": "user", "content": "hi"}],
       "max_tokens": 16,
       "reasoning_effort": "TEST"}'
```

```json
{"error":{"message":"1 validation error:\n  {'type': 'literal_error',
 'loc': ('body', 'reasoning_effort'),
 'msg': \"Input should be 'none', 'minimal', 'low', 'medium', 'high',
 'xhigh' or 'max'\", 'input': 'TEST'}"}}
```

A server that answers `200` to that ignores the key entirely — it is not a
lever there, and suppression has to come from one of the other spellings.

**Acceptance is not effectiveness — verify with a timer, not a status code.**
Measured against a live vLLM 0.11 serving Qwen3, with a prompt that induces
reasoning:

| sent | latency | content |
|---|---|---|
| `reasoning_effort: "none"` | 13.9s | 1301 chars |
| `chat_template_kwargs: {"enable_thinking": false}` | 13.9s | 1301 chars |
| `think` / `thinking` / `reasoning` | 27.9s | **accepted, ignored** |
| nothing at all | 27.9s | **0 chars**, `finish_reason: length` |

The last row is why suppression is on by default. Reasoning consumed the whole
`max_tokens` budget and the response came back **empty** — not slower, *empty*.
HLM sizes internal calls at `max(1024, len(prompt)//4 + 512)` with a 30s
timeout, which is exactly that regime. Raising the budget to 4000 tokens does
produce an answer, in 72.3s: three timeouts, then a failed call.

If you *want* the model to reason on HLM's internal calls, that is one setting
— `HLM_REASONING_EFFORT=provider_default` (or `HLM_REASONING_STYLE=off`) sends
nothing at all. `layer3_timeout_seconds` defaults to 120s, which clears the
72.3s measured above with margin; check your own endpoint against the table if
you change it. Covered by T189, T319-T321 and T334-T335.
