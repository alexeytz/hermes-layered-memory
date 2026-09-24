#!/bin/bash
# Stop the HLM MCP test fleet started by mcp-fleet-start.sh.
# Pidfile first, then a sweep by port, then verify every port is dark.
set -u
LOGDIR=${HLM_FLEET_LOGDIR:-/tmp/hlm-mcp-fleet-logs}
PORTS=${HLM_FLEET_PORTS:-"3801 3802 3803 3805 3806"}
stopped=0

if [ -f "$LOGDIR/pids" ]; then
  while read -r pid name; do
    [ -z "${pid:-}" ] && continue
    if kill "$pid" 2>/dev/null; then
      echo "stopped $name (pid $pid)"
      stopped=$((stopped+1))
    fi
  done < "$LOGDIR/pids"
  rm -f "$LOGDIR/pids"
fi

# Sweep strays by port. Safe against self-match because the pattern lives in
# this FILE, not in the running shell's argv (which is just
# "bash mcp-fleet-stop.sh"). Typing the same pgrep at a prompt kills the shell
# that typed it — that cost real time on 2026-08-27, twice.
for port in $PORTS; do
  for pid in $(pgrep -f "mcp_server.py --transport streamable-http --port $port" 2>/dev/null); do
    if kill "$pid" 2>/dev/null; then
      echo "swept stray on port $port (pid $pid)"
      stopped=$((stopped+1))
    fi
  done
done

sleep 1
echo "--- port check ---"
for port in $PORTS; do
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" \
    -X POST "http://127.0.0.1:$port/mcp" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"stopcheck","version":"1"}}}' 2>/dev/null)
  if [ "$code" != "000" ]; then echo "WARNING: port $port still answering ($code)"
  else echo "port $port dark"; fi
done
echo "fleet stopped: $stopped server(s)"
