#!/usr/bin/env bash
# Convenience wrapper for local Phase 0 runs.
# Sets up venv if missing, installs deps, runs the generator.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "[run_local] creating .venv"
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

echo "[run_local] installing deps"
pip install -e ".[datagen,dev]" --quiet

echo "[run_local] generating logs"
log-rca-gen "$@"
