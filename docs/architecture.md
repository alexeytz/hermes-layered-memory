# Architecture

## 5-Layer Retrieval Pipeline

```
Query → [Layer 0: Qdrant ANN] → [Layer 1: SQLite filters] → [Layer 2: Fusion scoring] → [Layer 3: Reranker] → [Layer 4: Gap detect] → Agent
```

| Layer | Operation | Cost | Filter |
|-------|-----------|------|--------|
| 0 | Qdrant vector search (HNSW) | ~5ms | data_type |

> `max_layer=0` does **not** cost ~5ms. Layer 0 returns `(uuid, distance)` pairs and Layer 1 is the SQLite read that turns them into records, so a caller asking for depth 0 gets exactly the work of depth 1 — and cannot get less, because skipping that read returns candidates with no content. The two are documented as equivalent in the tool schema; the ~5ms above is the cost of the *layer*, not of the setting.

| 1 | SQLite join + FTS5 BM25 | ~10ms | data_id, session_name, scope, TTL, temporal |
| 2 | Compound scoring | ~2ms | RRF over Layer-0 rank, plus normalized FTS5 rank, trust, recency, topic/keyword boosts. Worth +0.039 recall / +0.069 hard-recall over plain Layer-1 vector order at the current weights; at the pre-v0.3.0 weights it was *worse* than Layer 1 — the weights are load-bearing. Weights are configurable multipliers, **not** a percentage split — see [Scoring](reference.md#scoring-weights). |
| 3 | LLM reranker | ~5.2s measured (or instant if skipped) | Requests `limit*3` candidates from Layer 2. LLM decides if reranking is needed — returns `{"rerank_needed": false}` if BM25 already ranks well, saving ~30s. If ambiguous, re-ranks and returns top `limit`. Falls back to pass-through if LLM unavailable. Config: `layer3_mode`, `layer3_model`, `layer3_provider_config`. |
| 4 | Gap detection | ~1.3s measured | Calls LLM to check if results answer the query, identifies missing info. Annotates records with gaps and confidence. **Does not reorder anything** — measured identical to Layer 2 on recall, hard-recall and MRR, so it is an annotation step, not a ranking depth. Falls back to `gap_checked=true, gaps=[], confidence=1.0`. |

### Depth is explicit

`max_layer` means `max_layer` — nothing escalates on your behalf. Until v0.3.0 a
heuristic silently promoted any query with more than two non-stopword concepts
from 2 to 3, i.e. most real questions, at ~19x the latency (5.2s vs 0.28s) with
nobody asking. `_suggest_max_layer()` survives as advice and is logged at DEBUG.
- `max_layer=0`: Qdrant vector search only
- `max_layer=1`: +SQLite filters
- `max_layer=2`: +fusion scoring (default)
- `max_layer=3`: +LLM reranker
- `max_layer=4`: gap detection **instead of** the reranker — not in addition to
  it. `_run_pipeline` takes the `if max_layer == 4` branch straight to
  `_layer4()`, so depth 4 costs one LLM call rather than two and **never
  exercises L3**. Deliberate: L3 costs ~4.8x the latency and trades recall for
  ranking (measured 2026-09-18 — see the ladder table in `AGENTS.md`), and L4
  only needs summaries. Reaching the reranker requires `max_layer=3` or
  `rerank=True`.

  The `+` in the four lines above is cumulative; this line is the exception,
  and it is the one people get wrong. Two independent evaluations have asserted
  the opposite and recommended config changes on that basis — one of them while
  its own log read `layer4: skipped rerank`. `T627` fails if this line stops
  saying so.

**Smart-skip:** Layer 3 skips the LLM when BM25 scores are sufficient. Layer 3/4 LLM calls retry 3x with exponential backoff, and reasoning is suppressed by default (see [reference](reference.md#reasoning--thinking-suppression-across-providers)) — every LLM call HLM makes is JSON formatting, where a reasoning budget costs seconds and buys nothing.

Layer 3 is worth asking for: on a 540-record corpus it beats Layer 2 by +0.039 recall@5 and +0.136 MRR. It is not worth being billed for silently, which is why the escalation went rather than the layer.

### Circuit Breaker

Qdrant offline → degraded retrieval, 120s cooldown. The fallback first tries brute-force cosine over the locally stored vectors — exact and still semantic, just slower — and drops to FTS5/BM25 lexical ranking when that is unavailable.

**It is also skipped when it cannot answer the question asked.** The brute-force scan reads this backend's own packed embedding column and nothing else, so a `cross_profile=true` or targeted-other read goes straight to the lexical arm, which opens the target profile's database. Until 0.7.64 it ran anyway: it returned own-profile uuids, which is a non-empty result, so the fallback returned early and the profile-aware lexical arm never ran — and `_layer1`, which hydrates by uuid against the *target* database, then dropped every candidate. Nothing crossed a profile boundary, but the caller got a confident empty list from a path that had a working answer available. Those reads now log `[R010]` naming what is unavailable and why.

`sync_check()` and `layered_advanced(action="stats")` report four fields for this, because a single boolean cannot answer the question an operator is actually asking:

| Field | Meaning |
|---|---|
| `retrieval_mode` | How the **most recent** retrieval ranked: `vector` (Qdrant ANN), `brute_force` (exact local cosine — no quality loss), `lexical` (BM25 only — measured recall@5 0.227) |
| `degraded_retrieval` | Sticky: has **any** retrieval on this backend run without Qdrant? |
| `degraded_count` | How many retrievals have been degraded |
| `degraded_last` | ISO timestamp of the most recent degraded retrieval |

`degraded_retrieval` alone conflated `brute_force` with `lexical` — the difference between "slower" and "quietly returning worse answers" — and, being sticky, latched `true` forever in a long-lived process after one blip. `retrieval_mode` is set only on the read path (`_layer0`), deliberately **not** in `_record_qdrant_success()`: three of that method's four call sites are writes (`add`/`update`/`delete`), so resetting there would let ordinary write traffic mask degraded reads.

`qdrant_circuit_open` remains the breaker's own state and is independent of all four.

## Data Partitioning

Memories are partitioned for precise retrieval:

| Field | Purpose | Examples |
|-------|---------|----------|
| `data_type` | Structural partition + collection routing | USER-DATA, ENV-DATA, SESSION-DATA, SYSTEM, OBSIDIAN, CUSTOM (user-defined, mapped to collections in `hermes-layered-memory.json`) |
| `data_id` | Sub-partition | HW, SW, preferences, session UUID |
| `session_name` | Human-readable label | "vLLM GPU tuning", "memory plugin review" |

### Multi-collection routing

`data_type` maps to Qdrant collections via config `collections`:
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

- Collections are **shared across all profiles** — no profile prefix
- The name above is the *configured* one. Since v0.6.0 the physical collection
  appends the embedding model and dimension (`memories_qwen3-embedding_8b_4096`),
  resolved in `_init_qdrant` once the embedder has been probed and applied
  idempotently through `_physical_collection()` — `_load_runtime_config` and
  `register_taxonomy` both rebuild the map afterwards. A collection has one
  fixed vector size, so a shared name meant changing embedder broke every
  upsert until a full rebuild; a keyed name gives the new model its own
  collection while the old one keeps serving reads. Keyed on the model rather
  than the dimension, because two different 4096-dim models share a size but
  not an embedding space, and one collection for both returns confident
  nonsense instead of a loud error
- **Profile isolation is at SQLite level** (each profile has its own .db file), not at Qdrant level
- Qdrant points are tagged with `profile_name` payload; queries filter by it by default
- Upserts route to the correct collection by `data_type`
- Queries without `data_type` filter search all configured collections, merge results
- Queries with `data_type` filter target that collection directly
- Each collection is independent: different vector config, backup, deletion
- Default: all types use `memories` collection if `collections` not specified

### Cross-profile search

Each profile has its own SQLite DB, but all share one Qdrant index. The `profile_name` payload in Qdrant enables:
- Default: search current profile only
- `cross_profile=true`: search all profiles
- `profile_name="other-profile"`: search specific profile
- General session (no profile): sees everything

## Package Structure

```
backend/
  __init__.py          # re-exports LayeredBackend, logger (backward compat)
  constants.py         # Config defaults, heuristic maps, trust boundaries
  core.py              # Shared primitives: logger, settings, embeddings, encodings
  backend.py           # LayeredBackend class shell + method bindings
  store.py             # SQLite CRUD, migrations, config
  index.py             # Qdrant client, embeddings, ANN search
  pipeline.py          # Retrieval funnel (layers 0-4)
  maintenance.py       # rebuild, purge, compact, import/export
  llm.py               # _call_llm, enrich, classify, session state
```

**How the pieces attach.** The files under `backend/` other than `core.py` and
`constants.py` hold plain functions whose first argument is the backend
instance. They are bound in `LayeredBackend`'s **class body**
(`add = _store.add`), so `self` inside them is the backend and every call site
is navigable by an editor.

`core.py` imports none of the method modules. That is what makes the dependency
graph a DAG and lets the method modules import `logger`, `_setting` and the
embedding helpers directly. Before v0.3.0 they could not: methods were attached
by a `setattr` loop after the class, and shared names were fetched by string out
of `sys.modules` — 261 call sites that no editor or type checker could follow.

Line counts are deliberately not listed here; they drift the moment anyone edits
a file, and `wc -l backend/*.py` is authoritative.

Other files:
- `__init__.py` — MemoryProvider ABC impl, 6 meta-tools with action dispatch
- `summaries.py` — SummariesBackend: SQLite + FTS5 + .md files
- `mcp_server.py` — MCP server wrapper (FastMCP). Exposes the same 43 actions as
  the plugin, grouped into 8 tools plus optional `web_search`; `tests/test_mcp.py`
  (T400) reads the plugin's dispatch table and fails if any action is unreachable
- `mcp_tools/` — `memory_io` and `memory_config`, split out of `mcp_server.py`.
  They import nothing from it: the shared helpers arrive as a `ctx` namespace at
  registration, and each module declares its own actions and access policy, which
  are merged into the central tables at start-up

## Lifecycle Hooks

Declared in `plugin.yaml`:

| Hook | Purpose |
|------|---------|
| `on_session_end` | Maintenance pipeline (enrich → consolidate → purge → decay → sync → rebuild) |
| `prefetch` | Auto-searches memory on each user turn, injecting top 5 results into context. Rationale: local models are unreliable at following tool instructions — they often guess from context instead. Pushing candidates directly into the context window ensures the model sees relevant facts even if it never calls the tool intentionally. Yes, some turns will produce noisy results. That's an accepted trade-off: visible noise you can debug is preferable to invisible tool-call ignorance that causes silent wrong answers. |
| `sync_turn` | Spawns background fact extraction every 5 turns (throttled: 1 per 30s) |
| `on_pre_compress` | Records session in chain, extracts facts before context compression |
| `on_memory_write` | Mirrors MEMORY.md/USER.md writes to HLM (if `memory_enabled` is still on) |

## Session-end Maintenance

**Most of this is off by default since v0.3.0.** `enrich_existing`, `sleep` and
`decay` all mutate on a heuristic with no operator present, and each has been
the site of a data-loss defect; they now require `cleanup.automatic: true`.

| step | runs by default? | |
|---|---|---|
| `enrich_existing(max_items=20)` | no — `cleanup.automatic` | fills missing metadata |
| `sleep(max_items=50, min_age_hours=24)` | no — `cleanup.automatic` | archives low-trust duplicates |
| `decay(...)` | no — `cleanup.automatic` | reduces `trust_score` on stale records |
| `purge(purge_deleted=True, purge_archived=False, min_age_hours=24)` | **yes** | collects rows already soft-deleted 24h+ ago |
| `summaries.sync(profile="own")` | yes | reconciles the summaries DB with the filesystem |
| `sync_check()` → `rebuild()` | yes, on material drift | only past a threshold and no more than hourly |

`purge` stays automatic deliberately: with `purge_archived=False` it only
collects tombstones past the age threshold, which is garbage collection rather
than a judgement, and it is what stops the database growing without bound.

Fact extraction at session end is also off by default (`auto_extract`), for a
provenance reason rather than a cost one — see [Prompt Injection Defense](#prompt-injection-defense).

**Optional cron job** for heavy weekly maintenance:
```bash
hermes cronjob create --schedule "0 3 * * 0" --name "HLM weekly heavy purge" \
  --prompt "Call layered_maintenance(action="purge")(purge_deleted=true, purge_archived=true, min_age_hours=168), then layered_maintenance(action="sleep")(max_items=50, min_age_hours=0). Only run if there are records older than 7 days."
```

## Conflict Detection

Post-L2 pipeline: behavioral guards (data_id, temporal distance) → cosine similarity → Jaccard keyword overlap → L3 annotation → XML conflict alert. Configurable thresholds via `HLM_CONFLICT_THRESHOLDS`. Note: at *write* time the guards only compare against records older than `temporal_guard_days` (default 1), so recent records are never flagged.

## Supersession

`add(..., supersedes=<uuid>)` records that a fact changed. The replacement is returned by retrieval; the old value stays queryable with `status="all"` but no longer competes with the current answer. This is the right tool when a value has *changed*; `force=true` is for when two similar facts should genuinely coexist.

## Compaction Source Preservation

Merged records inherit `source="obsidian"` if any parent was obsidian-sourced (security-first OR).

## Auto-priority

ENV-DATA and SYSTEM records default to `priority=1` (half decay rate). Set `priority=3` for pinned (immune to decay). Set `protected=true` for permanent exemption.

## Prompt Injection Defense

Content from any source this system did not author itself (obsidian, web, MCP, importers) is wrapped in `<untrusted_external_doc>` tags on **every** path that reaches the model: explicit retrieve, peek, the per-turn prefetch injection, and the L3 reranker prompt. The delimiter is stripped from the payload first, so a document containing the closing tag cannot end its own fence. See [docs/security.md](security.md) for what is and is not defended.

**Why auto-extraction is off by default.** Extracted records used to be written
with `source="extraction"`, which was inside `SELF_AUTHORED_SOURCES` and
therefore permanently exempt from the fence. The conversation is fenced *going
into* the extraction prompt, but whatever the model decides is a durable fact
comes out the other side trusted and is replayed unfenced on every later
retrieval. That is a laundering path from fetched web or tool output into
trusted memory, so `auto_extract` must be turned on deliberately (and
`auto_extract_per_turn` separately again). `extraction` was removed from
`SELF_AUTHORED_SOURCES` in v0.4.1 for exactly this reason — this paragraph
described the fixed state as current for 47 releases, because the sentence
was never revisited when the code changed under it.

The session-start compaction extraction (`initialize()`'s `_extract_compaction`)
reads the same kind of input through a different door: a compaction summary the
LLM generated from a conversation that may itself carry fetched web/tool
output. It was written `source="compaction"`, which stayed in
`SELF_AUTHORED_SOURCES` until v0.7.53 — the identical laundering path, missed
when `extraction` was fixed because it is a different code path writing a
different string to the same set. Not gated by `auto_extract`; it runs
unconditionally whenever a compaction summary exists for the session.

Both extractors now share one storage body (`_store_extracted_facts`). They
were separate copies until 0.7.74 and the copies drifted exactly as copies do:
the compaction one never gained the contradiction retry, so a fact whose value
*changed* could not be relearned from a compaction summary — every later
extraction of the new value lost to the stale one. It had also gone months
without the `raw_decode` parsing fix its sibling received. One body, so a fix
to the write path cannot land in half of it.

## Extraction Ledger

Every auto-extraction run appends `{hook, candidates, stored, rejected,
superseded, echoes, reason}` to the history sidecar, surfaced via
`layered_advanced(action="stats")`. What memory learned and what it discarded
is answerable.

`echoes` is counted apart from `rejected` on purpose: a *rejected* fact is one
the store disagreed with (duplicate, possible-duplicate, a write that raised),
while an *echo* is one HLM itself surfaced into the conversation earlier in the
session and the extractor tried to learn back. Folding them together is how the
self-loop stayed invisible until 0.7.74. If the echo shield is ever mistuned,
this counter is the only place it shows.

### The self-loop shield

Retrieval injects records into the conversation; extraction reads the
conversation. Before writing each candidate, `_store_extracted_facts` calls
`check_surfaced_echo`, which compares the candidate's vector against the
records **this session surfaced** — the plugin's `_uuid_to_tag`, filled by
`_register_uuid` from prefetch and from every explicit retrieve/peek/list.
A match at or above `EXTRACTION_ECHO_THRESHOLD` (0.95, measured — see the
constant) is discarded and counted.

The threshold sits in a measured gap: a reworded echo scores 0.9643 while a
version bump scores 0.9021 on `qwen3-embedding:8b`. A *mutation must pass
through* to `add()` so the contradiction path can turn it into a supersession;
a shield tight enough to catch mutations would leave HLM unable to learn that a
fact changed. The check fails **open** on any error — an unavailable embedder
must not silently discard what the session learned.

This is deliberately independent of `data_type`. Dedup is scoped
`WHERE data_type = ?`, which only compares when two writers agree on a label,
and nothing enforces that agreement; the shield compares text, not labels.

## Prefetch Gating

Turns carrying no retrievable intent ("ok", "thanks") inject nothing, and records with `sensitivity > 0` are never auto-injected (they remain explicitly retrievable). Absolute scores cannot make this call: measured against a fixed corpus, "ok" scores *higher* than a real question.

## Embedding Storage

Embeddings are stored as packed float32 (`_pack_embedding`/`_unpack_embedding`), ~16KB for 4096 dims. Rows written before the change hold JSON text and are read transparently; `rebuild()` repacks them opportunistically. Embeddings are stripped from tool responses (saves 280K+ chars per call).

## Known Limitations

- `data_type:data_id` is the natural grouping for consolidation — `layered_maintenance(action="sleep")()` archives low-trust duplicates within the same group
- Fact extraction (`sync_turn`, `on_pre_compress`, `on_session_end`) runs as a single direct LLM call on a background thread, bounded by a concurrency cap. Per-turn extraction is throttled to every 5 turns / 30s; `on_session_end` is never throttled. Every run is recorded in the history sidecar — see `layered_advanced(action="stats")`.
- Qdrant collections must exist for all configured data_types (initialized on backend start)
- **UUID format:** 32-char hex (no dashes). Dashed format only used at Qdrant boundary. This is intentional — saves storage and avoids parsing overhead.
- **Soft-delete/Qdrant desync:** Soft-deleted records persist in Qdrant until `layered_maintenance(action="purge")` runs. `layered_maintenance(action="sync_check")` may show `in_sync=false`.
- **BM25 scoring:** `bm25_score` is a min-max normalization of FTS5 rank *within the candidate set*, not an absolute BM25 magnitude. The best candidate always scores 1.0 and the worst 0.0, so `min_bm25_threshold` filters by relative position, not match strength. Use the absolute `score` field (vector similarity) when you need a cross-query comparable number.
- **Semantic dedup limitation:** Requires Qdrant. When Qdrant is offline (circuit breaker) or disabled (`HLM_QDRANT_ENABLED=false`), dedup is bypassed — near-duplicate records will be stored. Run `layered_advanced(action="compact")` after Qdrant recovers to merge accumulated duplicates.