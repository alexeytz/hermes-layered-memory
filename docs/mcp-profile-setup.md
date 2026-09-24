# Setting up an MCP-only profile for HLM

How to stand up a Hermes profile that reaches HLM **only** over MCP, and the
server fleet to drive `docs/mcp-test-prompt.md` against it.

Adapted 2026-08-27 from a working setup on a second host. Endpoints here are
placeholders: the original hardcoded real hostnames, which `T159` fails the
build over, and that test exists because a real inference host once reached a
tracked file.

An MCP-only profile is the *opposite* shape to a plugin profile:

| | plugin profile (e.g. the e2e target) | MCP-only profile |
|---|---|---|
| HLM plugin symlink | yes | **no** |
| `memory.provider` | `hermes-layered-memory` | **unset** |
| built-in Hermes memory | off | off |
| MCP server registered | no | **yes** |
| tools visible | `layered_*` | `memory_*` (+ `web_search`) |

Both can coexist on one host, which is what makes a cross-door comparison
possible.

---

## 1. Create the profile

```bash
hermes profile create <profile> --no-skills
```

**A profile does not inherit the root model config.** Until you set one, the
first `chat` exits with "No inference provider configured" — before HLM is
reached at all, so it looks like a plugin failure and is not:

```bash
hermes -p <profile> config set model.default   <model-id>
hermes -p <profile> config set model.provider  custom
hermes -p <profile> config set model.base_url  http://<your-llm-host>:8000/v1
hermes -p <profile> config set model.api_mode  chat_completions
```

Turn the built-in memory off, or you get a second, undocumented store beside
the one you are testing:

```bash
hermes -p <profile> config set memory.memory_enabled       false
hermes -p <profile> config set memory.user_profile_enabled false
hermes -p <profile> config set memory.write_approval       false
```

**Do not create the plugin symlink and do not set `memory.provider`.** Verify
the absence rather than assuming it:

```bash
grep -c "layered_memory" ~/.hermes/profiles/<profile>/config.yaml   # expect 0
```

> The MCP server is a **separate process** and never reads this profile's
> `.env`. Every variable it needs is passed at launch (below). A `.env` here is
> reference material only, and the two drift apart silently.

---

## 2. Register the server on the profile

`hermes -p <profile> mcp add <name> --url http://127.0.0.1:3801/mcp` prompts
"Enable all N tools?" — and in a **non-TTY shell it cancels and writes
nothing**. Either run it in a TTY, pipe `Y` into it, or write the entry
directly:

```yaml
mcp_servers:
  hlm-memory:
    url: "http://127.0.0.1:3801/mcp"
    timeout: 300
    connect_timeout: 10
```

Then check it:

```bash
hermes -p <profile> mcp test hlm-memory
# ✓ Connected   ✓ Tools discovered: 9   (8 without SearXNG configured)
```

The nine are `memory_query`, `memory_write`, `memory_list_profiles`,
`memory_maintenance`, `memory_summaries`, `memory_advanced`, `memory_io`,
`memory_config`, `web_search`. **No `layered_*` tool may appear** — if one
does, the plugin is enabled somewhere and the split is broken.

---

## 3. The server fleet

One server per (default profile, admin posture) pair, all streamable-http,
sharing one Qdrant:

| port | name | admin | purpose |
|---|---|---|---|
| 3801 | main | yes | the profile under test; SearXNG if configured; cross-profile reads |
| 3802 | prof-a | no | isolation fixture |
| 3803 | prof-b | no | isolation fixture |
| 3805 | plain | **no** | the admin-gate test (Phase 14) |
| 3806 | nobudget | yes | `HLM_MCP_LLM_BUDGET=1` — the exhaustion refusal |

```bash
export HLM_EMBED_URL=http://<embed-host>:11434/api/embed
export HLM_EMBED_MODEL=qwen3-embedding:8b
export HLM_LAYER3_BASE_URL=http://<llm-host>:8000/v1
export HLM_LAYER3_MODEL=<model-id>
export HLM_SEARXNG_URL=http://<searxng-host>:8888     # optional; 9 tools vs 8

bash scripts/mcp-fleet-start.sh     # idempotent — skips ports already answering
bash scripts/mcp-fleet-stop.sh      # pidfile, then port sweep, then verifies dark
```

Fixture profiles are throwaway and must exist first — the provider flag is what
makes `_discover_profile_dbs` see them:

```bash
hermes profile create mcp-test-a --no-skills
hermes -p mcp-test-a config set memory.provider hermes-layered-memory
# after the run: hermes profile delete mcp-test-a --yes
#                rm ~/.hermes/hermes-layered-memory-dbs/mcp-test-a.db
```

**Point `HLM_SUMMARIES_DB` inside a temp directory.** The start script does this
for you. `digests.db` is one file shared by every profile, and a suite that
writes into the operator's real one leaves rows there for a month — that is
the defect `T582` now pins for the dispatch suite.

**After editing a profile's `hermes-layered-memory.json`, restart its server.**
Config is read at backend construction and backends are cached per profile for
the process lifetime, the same startup model the plugin uses. Verified live:
changing `scoring` moved the top fusion score only after a restart.

---

## 4. Running the suite

`docs/mcp-test-prompt.md` is written for an MCP-capable agent, and this profile
is one. Drive it either way:

* **Agent-driven** — `hermes -p <profile> chat -q "<phase instructions>"`, the
  agent calls the registered tools. Better for phases needing judgement.
* **Scripted** — a small streamable-http client, one session per call, with a
  state file carrying uuids between phases.

Things that only work over a real transport: the tool-count difference between
servers (8 vs 9), per-action admin gating, budget-exhaustion text, the refusal
JSON shape, and cross-profile reads.

**Cleanup is part of the suite.** Prefix written content with `[MCP-TEST] `,
delete by uuid, rebuild per profile, re-query to confirm zero. Never
`purge min_age_hours=0` — it is not marker-scoped and hard-deletes other
sessions' soft-deleted records.

**Never run the regression suite beside the fleet.** `T11` stops and restarts
the Qdrant container mid-run and the servers notice.

---

## 5. Traps

* **Application refusals are not protocol errors.** `is_error` stays `false`
  and the refusal is a JSON object with a single `error` key. A checker scoring
  on `is_error` alone counts every refusal as a pass — which is how an
  import-JSON defect once survived a green run of this suite.
* **`mcp` 2.0.0 renamed the flag** to `CallToolResult.is_error`, not
  `isError`. A client written against older docs raises `AttributeError` inside
  an `ExceptionGroup` — "unhandled errors in a TaskGroup" with no visible cause
  until the sub-exceptions are unwrapped.
* **Non-TTY `hermes mcp add` cancels silently.** See §2.
* **`pgrep -f "mcp_server.py ..."` typed at a prompt matches the shell running
  it.** Kill from a script file, where argv is just `bash mcp-fleet-stop.sh`,
  or from the pidfile. `mcp-fleet-stop.sh` does both.
* **Summaries parameters are `uuid` and `source_url`**, not `id`/`url`. The
  prompt's prose is looser than the schema; the schema wins.
* **The server never reads the profile `.env`.** Miss `HLM_EMBED_URL` in the
  *server process* and every vector call falls back silently while the suite
  reports green — the same trap `AGENTS.md` records for the regression suite.
* **`HLM_DB_PATH` on a server serving cross-profile reads.** Before 0.8.31 the
  override applied to every profile-qualified read, so
  `memory_query(profile=X)` returned the override database's records as X's,
  silently, while `list_profiles` still showed X's real store. That is now
  **refused** rather than redirected (`T583`), but single-DB mode still has no
  business on a server answering for other profiles.
* **Bare `eval_retrieval.py --check` exits 2** on a host with the production
  embedder exported: the default baseline is anchored to the local fallback.
  Pass `--baseline tests/eval-baseline-prod.json`, which is what `T364` does.
