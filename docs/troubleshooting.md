# Troubleshooting

## Qdrant Not Responding

```bash
docker compose ps    # check container status
docker compose logs  # check for errors
curl -s http://localhost:6333/  # health check
```

## Circuit Breaker Triggered (Qdrant offline → SQLite fallback)

```bash
# Check Qdrant
curl -s http://localhost:6333/collections
# Restart if needed
docker compose restart qdrant
```

The breaker opens after 5 consecutive Qdrant failures and cools down for 120s, then lets a single half-open trial request through; if that probe fails it goes straight back to open. Search failures increment the counter (they previously did not — the breaker never tripped on `retrieve()` traffic), so an outage now opens the circuit instead of retrying a dead Qdrant on every query. Look for `[Q002] Qdrant circuit OPEN` / `[Q001] re-OPENED` / `[Q003] Qdrant failure (n/5)` in the log.

**Retrieval during an outage:** not lexical-only. Whenever Qdrant is unreachable — breaker cooldown, `_init_qdrant()` failure at startup, or `HLM_QDRANT_ENABLED=false` — layer 0 falls to `_degraded_semantic_fallback()`, a brute-force cosine scan over the embeddings already stored in SQLite, so real semantic ranking survives the outage. FTS5/BM25 lexical ranking is only the second fallback, used if `numpy` is missing or the embedder can't produce a query vector (measured recall@5 on the lexical-only path: 0.227). Practical consequence: latency rises with corpus size, but results stay useful.

**Limitation:** Semantic dedup at `add()` time is bypassed only when there is no Qdrant *client* — i.e. `HLM_QDRANT_ENABLED=false`, or Qdrant was already unreachable when the backend initialized. In those cases near-duplicate records will be stored; run `layered_advanced(action="compact")` after Qdrant recovers to merge them. If Qdrant was reachable at startup and failed later (the circuit-breaker case), `_check_duplicate()` falls back to a cosine scan over stored SQLite embeddings and dedup keeps working.

## PyTorch warning: `KernelPreference` deprecation

```
W0611 ... torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass...
```

From `sentence-transformers`/`torch` when loading the embedding model. Harmless — won't affect functionality. Suppress with `export PYTHONWARNINGS="ignore::DeprecationWarning"`.

## `Vault path ... is outside allowed roots`

```
Vault path '/mnt/data/vault' is outside allowed roots ['/home/user'].
Set HLM_OBSIDIAN_VAULT_ROOTS (colon-separated) to allow additional vault locations.
```

`layered_io(action="obsidian_ingest")` resolves `vault_path` with `realpath()` and requires the result to live under an allowed root. The only default root is the real home directory, so **vaults stored outside home now fail where they previously ingested.** Add the location:

```bash
export HLM_OBSIDIAN_VAULT_ROOTS=/mnt/data:/srv/notes   # colon-separated absolute paths
```

Roots are `realpath()`-resolved too, so give the real path, not a symlink. Notes reached through a symlink that escapes the vault are skipped individually (counted in the returned `skipped`, logged at DEBUG as `symlink escape:`) rather than failing the run.

## `Import path ... is outside allowed roots`

```
Import path '/mnt/backups/export.json' is outside allowed roots ['/home/user'].
Set HLM_IMPORT_ALLOWED_ROOTS (colon-separated) to allow additional locations.
```

Same containment check for `layered_io(action="import")` when `data` is a file path (anything ending in `.json` or `.md`). Fix with `HLM_IMPORT_ALLOWED_ROOTS`, or pass the JSON inline as a string instead of a path — inline data is not path-checked. Returned as `{"imported": 0, "skipped": 0, "failed": 1, "errors": [...]}`, not a raised error.

## `content exceeds max length` / `metadata exceeds max length` / `summary` / `topic` / `data_id` / `keywords`

```
content exceeds max length (63120 > 50000 chars)
metadata exceeds max length (24880 > 20000 chars)
summary exceeds max length (61004 > 50000 chars)
topic exceeds max length (52310 > 50000 chars)
data_id exceeds max length (900 > 500 chars)
keywords exceeds max length (58120 > 50000 chars)
keywords entry exceeds max length (51200 > 50000 chars)
```

**Five fields are bounded, not two.** `content` and `metadata` are the ones
usually hit, but `summary`, `topic`, `data_id` and `keywords` are guarded too
(`_check_fts_field_bounds` in `backend/store.py`) because all four are written
into the FTS5 index, where an unbounded value bloats it exactly as content
does. `data_id` has its own, much smaller bound — `MAX_DATA_ID_CHARS` = 500 —
because it is a label, not prose. `keywords` is checked twice: the serialized
list against `MAX_FIELD_CHARS`, and each entry individually, so one enormous
term cannot pass by hiding in a short list. `action="update"` enforces the same
five bounds as `action="add"`.

`layered_memory(action="add")` bounds the write payload before doing any embedding or dedup work: 50,000 chars of content, 20,000 chars of serialized (JSON) metadata — `MAX_CONTENT_CHARS` / `MAX_METADATA_CHARS` in `backend/constants.py`. An unbounded record bloats the FTS5 index and the shared Qdrant collection and degrades retrieval for the whole profile. `action="update"` enforces the same content bound and rejects empty content (v0.7.54) — it did neither before, so an in-place edit could grow a record past the cap or clear it to a contentless row nothing could retrieve.

Fixes:
- Split the content into several records along its natural boundaries — smaller records retrieve better anyway.
- For bulk text, use `layered_summaries(action="summarize")` instead: the full text is written to a `.md` file (default `~/Documents/hlm-summaries/`) and only the highlights live in the DB.

## Sync Drift

```bash
# Check
layered_maintenance(action="sync_check")()
# Fix
layered_maintenance(action="rebuild")()
```

## A healthy tool call shows `[error]` in the terminal

**Symptom.** A `layered_advanced(action="stats")` call returns a correct payload,
and the progress line still reads `⚡ layered_a 0.0s [error]`. `agent.log` carries
a matching `WARNING ... Tool layered_advanced returned error (0.02s): {"prefetch":
1, ...` whose body is a valid stats dict, and the *next* log line says
`tool layered_advanced completed`.

**Cause — Hermes' display layer, not HLM.** `_detect_tool_failure`
(`~/.hermes/hermes-agent/agent/display.py`, a separate checkout) greps the first 500 characters of a serialized
result for the substrings `"error"` and `"failed"`:

```python
lower = result[:500].lower()
if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
    return True, " [error]"
```

HLM's stats payload legitimately contains `"failed"` — it is the count of failed
extractions, inside the `extraction` block. Measured 2026-09-13: the payload is
629 characters and `"failed"` sits at **offset 444**, inside the window. There is
no `error` key at any level.

**It flickers, which is the worst part.** 444 is only 56 characters from the
cutoff, and everything ahead of it is variable-width counters
(`last_purge: 2406`, review counts, extraction totals). A profile with larger
numbers pushes `"failed"` past 500 and the label disappears. So the same healthy
call is labelled `[error]` on one profile and not on another, and intermittent
false errors are what teach people to stop reading error labels.

**What it does not affect.** `scripts/check-e2e-log.py` reads only
`hermes-layered-memory.log` and matches specific patterns, so this never reaches
the e2e log gate. It is cosmetic — but a driver or reviewer who trusts the
`[error]` suffix will score a passing diagnostics step as a failure, which is how
it was found.

**Do not "fix" it in HLM by renaming the field.** `failed` is the correct name
for a count of failures, the defect is a substring heuristic in another
repository, and contorting a public payload to dodge someone else's grep would
break silently the moment that grep changed.

## Console logging during `/quit` (shutdown)

Console logs from `on_session_end()` maintenance appear at the top of the terminal screen, not between "Shutting down…" and the session summary. This is expected behavior — Hermes CLI suppresses stderr during shutdown. All maintenance logs are written to the log file (`~/.hermes/logs/hermes-layered-memory.log`) and are fully visible there.

## Plugin Not Discovered

```bash
# Verify symlink
ls -la ~/.hermes/profiles/<profile>/plugins/hermes-layered-memory
# Should point to ~/.hermes/plugins/hermes-layered-memory
# Enable
hermes -p <profile> plugins enable hermes-layered-memory
```

## Qdrant upsert fails (point ID format)

The plugin handles UUID format conversion internally. If you see "invalid point ID" errors, ensure you're running the latest code (post-fix for qdrant-client 1.18 compatibility).

## Embedding dimension mismatch

Mostly historical since v0.6.0: the physical collection name carries the
embedding model (`memories_qwen3-embedding_8b_4096`), so a different embedder
gets its own collection instead of colliding with one sized for the old model.
The warning can still appear with `HLM_QDRANT_COLLECTION_SUFFIX=false`, or
against a collection created before the upgrade.

When it does: a Qdrant collection has one fixed vector size, so every upsert
fails and retrieval falls back to brute force. `layered_maintenance(action="rebuild")()`
re-embeds all records and upserts at the current dimension.

## Profile looks empty after upgrading to v0.6.0

**Symptom:** `sync_check` reports `qdrant_profile: 0` against a healthy
`sqlite_active`, `retrieval_mode` is `brute_force`, and retrieval quality drops
— but nothing was deleted.

**Cause:** collection names became model-keyed in v0.6.0. The old vectors are
still in the flat `memories`/`sessions`/`vault` collections, which nothing reads
any more.

**Fix:** `layered_maintenance(action="rebuild")()` per profile. SQLite is
authoritative and holds the embeddings, so this is a re-index, not a data
recovery. Drop the old collections once every profile has been rebuilt.

**Check you are not chasing a ghost first.** `qdrant_profile` counts points
whose `profile_name` payload equals *this backend's* profile name. A script or
one-off that opens the backend with a different spelling than the plugin uses
(`hlmtest` vs `hlm-test`) will report 0 for a perfectly healthy profile — and a
"rebuild" launched from that script relabels every point, which is the only way
to actually break it. Compare against the plugin's own line in the log:
`on_session_end: sync check — SQLite=N, Qdrant=N, in_sync=True`.

**Before 0.7.77, archiving was the other cause** — and it looked exactly like
stale vectors that a rebuild ought to fix, except a rebuild never did.
`update(status="archived")` and `sleep()` left the Qdrant point behind, while
`sync_check()` counts `status='active'` rows, so `in_sync` went False the first
time anything was archived and stayed there: `rebuild()` selects
`status='active'`, so it re-embedded every live record and never removed the
orphan. Both doors now retire the point and release the record's supersessions.
On a profile carrying orphans from before 0.7.77, one `purge` of archived rows
(or a `test_cleanup` on a test profile) clears them; a rebuild alone still will
not.

## Merged records show `first_observed_at: "personal"` or `source: consolidated-untrusted`

Symptoms of a column mis-read in `_compact_group` fixed on 2026-08-08: for every
merge, `created_at` held the record's *scope* and `source` held a timestamp. Two
consequences on rows written before the fix:

- `first_observed_at` holds a scope string (`"personal"`, `"work"`) instead of a
  timestamp — in the row, in `metadata`, and in the Qdrant payload.
- `source` is `consolidated-untrusted`, so HLM's own consolidated text is wrapped
  in `<untrusted_external_doc>` fences on every retrieval, telling the model to
  treat its own memory as hostile input.

Fixing the code does not fix existing rows. **One-time repair:**

```bash
# 1. Back up first — this rewrites rows in place.
#    (or: layered_io(action="backup"))
cp ~/.hermes/hermes-layered-memory-dbs/<profile>.db /tmp/backup.db

# 2. See what would change. Dry-run is the default; nothing is written.
python3 scripts/repair-compaction-damage.py --db ~/.hermes/hermes-layered-memory-dbs/<profile>.db

# 3. Apply.
python3 scripts/repair-compaction-damage.py --db ... --apply

# 4. Refresh the Qdrant payloads (first_observed_at is mirrored there).
#    layered_maintenance(action="rebuild")

# All profiles at once:
python3 scripts/repair-compaction-damage.py --all-profiles --apply
```

It is idempotent — a second run reports 0 repairs — so it is safe to re-run if
interrupted. It only downgrades `consolidated-untrusted` back to
`hlm-consolidated` when **every** parent in `metadata.compacted_from` was itself
self-authored; a merge that genuinely absorbed external content keeps its fence,
and so does one whose parents have been purged (no evidence either way, so it
fails safe). Records with no damage are left untouched.

## Dependencies Not Installed

HLM's Python dependencies are **not auto-installed** by Hermes. They must be present in the Hermes agent venv. After a Hermes upgrade (which rebuilds the venv), reinstall:

```bash
# Install into Hermes agent venv (adjust path if yours differs).
# torch first, from the CPU index: sentence-transformers depends on torch and the
# default PyPI wheel is the CUDA build, which drags in the nvidia-* runtime and
# triton — measured at ~4.5G on a CPU-only host, none of it reachable, because
# HLM hardcodes `device="cpu"`.
~/.hermes/hermes-agent/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.hermes/hermes-agent/venv/bin/pip install qdrant-client numpy sentence-transformers
```

Optional packages (install only if you use the feature):

```bash
# Multilingual embeddings (1024-dim, 100+ languages)
~/.hermes/hermes-agent/venv/bin/pip install fastembed
```

| Package | Required? | Purpose |
|---------|-----------|---------|
| qdrant-client >= 1.18 | Yes | Qdrant vector search client |
| numpy | Yes | Brute-force semantic search (Qdrant-less fallback), cosine similarity |
| sentence-transformers >= 5.5 | Yes | Default embedding model (all-MiniLM-L6-v2, 384-dim) |
| fastembed | No | Local multilingual embedding (intfloat/multilingual-e5-large, 1024-dim) |
| Docker + Docker Compose | No | Qdrant container (optional — remote Qdrant or brute-force mode work without it) |

**Degradation without dependencies:**
- **No `qdrant-client`:** Vector search disabled, falls back to brute-force SQLite scan (requires `numpy`)
- **No `numpy`:** Brute-force search returns empty — retrieval falls back to FTS5 BM25 only
- **No `sentence-transformers` and no remote embed URL:** Embedding fails entirely; `add()` and `rebuild()` cannot compute vectors
- **No `fastembed`:** Only affects users who set `HLM_LOCAL_EMBED_MODEL` — falls through to sentence-transformers

## Logging Issues

Logs written to `<hermes-home>/logs/hermes-layered-memory.log`.
- Default profile: `~/.hermes/logs/hermes-layered-memory.log`
- Named profile: `~/.hermes/profiles/<profile>/logs/hermes-layered-memory.log`

Override via `HLM_LOG_FILE` (see env vars in [docs/reference.md](reference.md)).

```bash
# Follow the log in real-time
tail -f ~/.hermes/logs/hermes-layered-memory.log

# Control verbosity (INFO default, DEBUG for full layer pipeline)
HLM_LOG=DEBUG
```

Example log output:
```
22:35:12 INFO hermes-layered-memory.backend retrieve: query='who am I?' max_layer=3 scope=None data_type=None data_id=None session_name=None limit=5
22:35:12 DEBUG hermes-layered-memory.backend layer0: 30 candidates
22:35:12 DEBUG hermes-layered-memory.backend layer1: 17 records after filters
22:35:12 DEBUG hermes-layered-memory.backend layer2: 17 scored
22:35:12 DEBUG hermes-layered-memory.backend layer3: 17 reranked
22:35:12 INFO hermes-layered-memory.backend retrieve: returned 5 results
```

## LLM calls return empty, or time out (reasoning model)

Symptoms: `[L003] LLM returned no response`, `LLM returned no content`, rerank
and fact extraction producing nothing, or L3 calls taking 30s+ and failing all
three attempts.

Cause is usually that reasoning is **not** being suppressed. The thinking block
consumes the `max_tokens` budget and the response comes back with
`finish_reason: "length"` and an empty `content` — a successful HTTP 200 that
contains nothing. Measured on a live vLLM 0.11 serving Qwen3: 27.9s and **0
characters** unsuppressed at an 800-token budget, versus 13.9s and a full
answer with suppression on.

Check what the endpoint does with each spelling — status codes are not enough,
because a permissive server *accepts and ignores* keys it does not implement
(`think`, `thinking` and `reasoning` are all accepted-and-ignored by vLLM).
Time it instead:

```bash
for extra in '"reasoning_effort":"none"' \
             '"chat_template_kwargs":{"enable_thinking":false}' \
             '"think":false'; do
  echo -n "$extra -> "
  curl -s -o /tmp/r.json -w '%{time_total}s ' \
    "$HLM_LAYER3_BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $HLM_LAYER3_API_KEY" \
    -d "{\"model\":\"$HLM_LAYER3_MODEL\",\"max_tokens\":800,
         \"messages\":[{\"role\":\"user\",
           \"content\":\"Solve the proof for 2+2=5 in a fictional universe.\"}],
         $extra}"
  python3 -c "import json;m=json.load(open('/tmp/r.json'))['choices'][0];\
print(len(m['message'].get('content') or ''),'chars',m.get('finish_reason'))"
done
```

The variant that is both fast and returns a non-empty `content` is the one that
works. `finish_reason: length` with 0 chars means it reasoned.

To discover the accepted `reasoning_effort` values, send an invalid one and
read the enum out of the 400 — see "Picking a value" in `docs/reference.md`.

HLM suppresses by default and repairs on rejection, so this normally needs no
intervention. If your endpoint needs a spelling auto-detection gets wrong, pin
it with `HLM_REASONING_STYLE`; if you deliberately want reasoning on internal
calls, set `HLM_REASONING_EFFORT=provider_default` and check
`layer3_timeout_seconds` against your endpoint — the 120s default clears the
72.3s measured above, but a slower model or a longer prompt will need more.

## Retrieval Quality Issues

`tests/eval_retrieval.py` measures recall@5, MRR and latency per layer against a fixed corpus, so scoring changes can be judged instead of guessed:

```bash
python3 tests/eval_retrieval.py                                   # 40 golden + 500 distractors
python3 tests/eval_retrieval.py --max-layer 4 --profile <name>    # incl. L3/L4 (needs an LLM)
python3 tests/eval_retrieval.py --scoring bm25_weight=0.2         # try a Layer-2 weight
```

**Regression floor.** Numbers nobody compares against are numbers nobody notices
moving, so the harness carries a baseline:

```bash
python3 tests/eval_retrieval.py --check           # exit 1 if recall/MRR dropped >5%
python3 tests/eval_retrieval.py --write-baseline  # re-anchor after an intended change
```

`--check` refuses to compare across a fingerprint change (corpus size, query
count, `k`, or embedder) rather than silently comparing incomparable runs — so
switching to `--profile <name>` will tell you to re-anchor instead of reporting a
false regression. `tests/eval-baseline.json` is anchored on the local fallback
embedder and needs no endpoint; `tests/eval-baseline-prod.json` records the
production-embedder numbers for reference.

There is no CI in this repository to wire this into. Until there is, run
`--check` before committing anything that touches `_layer0`-`_layer2`, the
scoring weights, or the embedding configuration. As a git hook:

```bash
cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
# Retrieval regression gate. Needs Qdrant; ~90s.
python3 tests/eval_retrieval.py --check || {
  echo "retrieval regression — push aborted (use --no-verify to override)"; exit 1; }
EOF
chmod +x .git/hooks/pre-push
```

**Reading the report:** `hard@5` is the number to watch. The easy queries
saturate, so the headline `recall@5` mixes a ceiling with the only queries able
to show movement.

## Verification After E2E Runs


## Version Mismatch

Check that `plugin.yaml` version matches `backend/constants.py`:
```bash
python3 scripts/check-version.py
```

If mismatched, update `backend/constants.py` to the desired version and regenerate `plugin.yaml`.

## Testing

Full reference — every suite, flag, requirement and isolation guarantee: `docs/testing.md` (development repository).

```bash
# Run full regression suite (~7-18 min — background it)
cd hermes-layered-memory
docker compose up -d
python3 tests/run-regression.py

# Run dispatch-layer tests (37 tests, ~15s, catches schema/type bugs)
python3 tests/test_dispatch.py

# Degraded-mode tests (Qdrant offline, dead embedder, recovery)
python3 tests/test_chaos.py

# Measure retrieval quality — recall@5, MRR, latency per layer
python3 tests/eval_retrieval.py --filler 500

# Results written to tests/regression-results.json and tests/dispatch-results.json
# Individual test files in tests/ can be run directly:
python3 tests/test_core.py       # T1-T30: CRUD, filters, layers
python3 tests/test_summaries.py  # T31-T46: Summaries
```

## MCP Server Issues

**Security:** HTTP mode defaults to `127.0.0.1` (localhost only). Use `--host 0.0.0.0` to expose on network — secure via firewall/VPN/SSH tunnel.

**Multi-agent profile targeting:** `memory_query` and `memory_write` accept an optional `profile` parameter to target a specific profile. Use `memory_list_profiles` to discover available profiles. Use `cross_profile=true` in `memory_query` to search across all profiles.

**`Writes to profile 'X' are not allowed`:**

```json
{"error": "Writes to profile 'pi' are not allowed — only the default profile ('profile-a') is writable unless 'pi' is listed in HLM_MCP_ALLOWED_PROFILES."}
```

`memory_write` actions `add`/`update`/`delete`/`maintenance` are restricted to the server's default profile (`HLM_MCP_PROFILE`). Reads are unaffected — `memory_query`, `list` and `status` still reach any profile. To allow cross-profile writes, list the targets explicitly:

```bash
export HLM_MCP_ALLOWED_PROFILES=profile-a,pi,shared   # comma-separated
```

Note that setting this variable also tightens *targeted* reads to the listed profiles, so include the default profile in the list. It does not restrict `cross_profile=true` searches, which discover profile DBs independently. Running a second server process with its own `HLM_MCP_PROFILE` is the alternative when profiles should stay mutually unwritable.

**`Invalid profile name`:**

```json
{"error": "Invalid profile name: '../../tmp/evil' — use alphanumeric, hyphens, underscores only"}
```

Profile names are validated against `^[A-Za-z0-9_-]+$` before any DB path is built. Dots, slashes, spaces and an empty string are all rejected. Rename the profile (and its `<profile>.db` file) to a conforming name.

**Generic tool errors (`query failed: OperationalError`):** MCP tool errors are deliberately sanitized — the client gets `"<context> failed: <ExceptionType>"` and nothing more, because the server is unauthenticated. The full exception and traceback are in the server log (`~/.hermes/logs/hermes-layered-memory.log`); look there when diagnosing.

See [docs/mcp.md](mcp.md) for standalone usage (without Hermes) and the full multi-agent design.