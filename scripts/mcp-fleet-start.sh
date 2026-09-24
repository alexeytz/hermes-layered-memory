#!/bin/bash
# Start an HLM MCP test fleet: several streamable-http servers on one Qdrant,
# each owning a distinct default profile, for driving docs/mcp-test-prompt.md.
#
# Adapted 2026-08-27 from a working fleet on another host. Endpoints come from
# the environment with localhost defaults — the original hardcoded real
# infrastructure hostnames, which T159 fails the build for, and rightly: this
# file ships.
#
#   3801  main     ADMIN, SearXNG if configured, serves cross-profile reads
#   3802  prof-a   non-admin, isolation fixture profile
#   3803  prof-b   non-admin, isolation fixture profile
#   3805  plain    NON-admin, same profile as main — the admin-gate test
#   3806  nobudget ADMIN with HLM_MCP_LLM_BUDGET=1 — exhaustion refusal test
#
# Fixture profiles are throwaway and must exist first (the provider flag is what
# makes _discover_profile_dbs see them):
#   hermes profile create mcp-test-a --no-skills
#   hermes -p mcp-test-a config set memory.provider hermes-layered-memory
#   (same for mcp-test-b; delete both, and their .db files, after the run)
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${HLM_FLEET_PYTHON:-python3}
LOGDIR=${HLM_FLEET_LOGDIR:-/tmp/hlm-mcp-fleet-logs}
SUMDIR=${HLM_FLEET_SUMDIR:-/tmp/hlm-mcp-summaries-$(date -u +%Y%m%d)}
PROFILE=${HLM_FLEET_PROFILE:-hlm-mcp-test}
PROF_A=${HLM_FLEET_PROFILE_A:-mcp-test-a}
PROF_B=${HLM_FLEET_PROFILE_B:-mcp-test-b}

mkdir -p "$LOGDIR" "$SUMDIR"
: > "$LOGDIR/pids"

# Endpoints: override via the environment. Defaults are local so nothing
# private is baked in.
EMBED=(HLM_EMBED_URL="${HLM_EMBED_URL:-http://localhost:11434/api/embed}"
       HLM_EMBED_MODEL="${HLM_EMBED_MODEL:-qwen3-embedding:8b}")
QD=(HLM_QDRANT_URL="${HLM_QDRANT_URL:-http://localhost:6333}")
LLM=(HLM_LAYER3_MODE="${HLM_LAYER3_MODE:-inline}"
     HLM_LAYER3_MODEL="${HLM_LAYER3_MODEL:-}"
     HLM_LAYER3_BASE_URL="${HLM_LAYER3_BASE_URL:-}"
     HLM_REASONING_EFFORT="${HLM_REASONING_EFFORT:-none}")
# No API key here on purpose. If your endpoint needs one, export
# HLM_LAYER3_API_KEY in the shell that runs this; it must not live in a
# tracked file.
[ -n "${HLM_LAYER3_API_KEY:-}" ] && LLM+=(HLM_LAYER3_API_KEY="$HLM_LAYER3_API_KEY")

if [ -z "${HLM_EMBED_URL:-}" ]; then
  echo "note: HLM_EMBED_URL unset — servers will use the local fallback embedder."
  echo "      Every vector call then exercises the FALLBACK path while the suite"
  echo "      reports green. Export it before trusting a run."
fi

alive() {  # port -> 0 if something already answers MCP there
  local code
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" \
    -X POST "http://127.0.0.1:$1/mcp" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"startcheck","version":"1"}}}' 2>/dev/null)
  [ "$code" != "000" ]
}

start() {  # name port profile extra_env...
  local name=$1 port=$2 profile=$3; shift 3
  # Idempotent: a half-restart must not double-launch. The loser dies on bind
  # and its log ends in "address already in use", which reads like a server
  # defect exactly when you are trying to triage one.
  if alive "$port"; then
    echo "$name port=$port already up — skipped (run mcp-fleet-stop.sh for a clean restart)"
    return
  fi
  env "${EMBED[@]}" "${QD[@]}" "${LLM[@]}" \
      HLM_MCP_PROFILE="$profile" \
      HLM_SUMMARIES_DB="$SUMDIR/digests.db" \
      HLM_SUMMARIES_DIR="$SUMDIR/md" \
      HLM_LOG=DEBUG \
      "$@" \
      "$PY" "$REPO/mcp_server.py" --transport streamable-http --port "$port" \
      >> "$LOGDIR/$name.log" 2>&1 &
  echo "$! $name" >> "$LOGDIR/pids"
  echo "$name pid=$! port=$port profile=$profile"
}

# HLM_DB_PATH is deliberately NOT set on `main`. Before 0.8.31 the override was
# applied to every profile-qualified read, so memory_query(profile=X) silently
# returned the override database's records as X's. That is now refused rather
# than silently redirected, but a server serving cross-profile reads still has
# no business in single-DB mode. Default resolution lands on the same file.
start main    3801 "$PROFILE" HLM_MCP_ADMIN=true \
      ${HLM_SEARXNG_URL:+HLM_SEARXNG_URL="$HLM_SEARXNG_URL"}
start prof-a  3802 "$PROF_A"
start prof-b  3803 "$PROF_B"
start plain   3805 "$PROFILE"
start nobudget 3806 "$PROFILE" HLM_MCP_ADMIN=true HLM_MCP_LLM_BUDGET=1

echo
echo "fleet: 3801 main / 3802 prof-a / 3803 prof-b / 3805 plain / 3806 nobudget"
echo "logs:  $LOGDIR/    summaries: $SUMDIR/"
[ -z "${HLM_SEARXNG_URL:-}" ] && echo "note: HLM_SEARXNG_URL unset — web_search absent, 8 tools not 9."
exit 0
