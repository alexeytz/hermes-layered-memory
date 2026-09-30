#!/bin/bash
# Build the development venv the test suites run in.
#
#   scripts/make-dev-venv.sh [DIR]        # default ~/.venvs/hlm-dev
#
# Why this exists. The suites need two things at once: Hermes' own packages
# (the plugin imports `agent.memory_provider` and `tools.registry` at module
# level) and the suites' own dependencies — `sentence-transformers` above all,
# because they exercise the 384-dim MiniLM fallback. Until 2026-09-28 the
# in-tree `~/.hermes/hermes-agent/venv` had both and every document said to use
# it. Hermes' package manager (pm/) then took over the runtime: Hermes runs in a
# PM-built environment that holds only what Hermes and the enabled plugins
# DECLARE — no sentence-transformers, by design (see README "Dependencies") —
# and PM's own docstring says it deletes the in-tree venv once a generation is
# committed. So the suites need an interpreter nobody else manages.
#
# What it builds: a venv on the same Python Hermes runs on (PM's managed
# interpreter when present), with Hermes installed editable from its checkout,
# torch from PyTorch's CPU index (the PyPI wheel is the ~4.5G CUDA build, and
# HLM hardcodes device="cpu"), and the suites' dependencies. Nothing here
# touches Hermes' runtime environment, its lockfile or any profile.
#
# Idempotent: re-running upgrades the venv in place.
set -euo pipefail

DEV="${1:-$HOME/.venvs/hlm-dev}"
HERMES="${HERMES_AGENT_ROOT:-$HOME/.hermes/hermes-agent}"

[ -d "$HERMES" ] || { echo "no Hermes checkout at $HERMES" >&2; exit 1; }

# The interpreter: PM's managed Python when this Hermes has one (the suites
# then run on the Python the plugin runs on), else whatever python3 is.
PY="$(ls -d "$HOME"/.hermes/tools/python-3.*/bin/python3 2>/dev/null | sort -V | tail -1 || true)"
PY="${PY:-$(command -v python3)}"
# uv when PM ships one (fast, and what PM itself uses), else pip.
UV="$(ls -d "$HOME"/.hermes/tools/uv-*/uv 2>/dev/null | sort -V | tail -1 || true)"

echo "dev venv : $DEV"
echo "python   : $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
echo "installer: ${UV:-pip}"

if [ -n "$UV" ]; then
  [ -x "$DEV/bin/python" ] || "$UV" venv --python "$PY" "$DEV"
  inst() { "$UV" pip install --python "$DEV/bin/python" "$@"; }
else
  [ -x "$DEV/bin/python" ] || "$PY" -m venv "$DEV"
  inst() { "$DEV/bin/python" -m pip install "$@"; }
fi

# torch first and alone, from the CPU index, so nothing later pulls the CUDA
# build from PyPI to satisfy sentence-transformers.
inst torch --index-url https://download.pytorch.org/whl/cpu
inst -e "$HERMES" qdrant-client numpy sentence-transformers fastembed mcp pyyaml packaging

"$DEV/bin/python" - <<'PY'
import importlib.util as u, sys
need = ("agent.memory_provider", "tools.registry", "qdrant_client", "numpy",
        "sentence_transformers", "fastembed", "mcp", "yaml", "packaging")
missing = [m for m in need if u.find_spec(m) is None]
import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (cuda build: {torch.version.cuda})")
if missing:
    sys.exit(f"missing after install: {missing}")
print("ok — run the suites with:", sys.executable)
PY
