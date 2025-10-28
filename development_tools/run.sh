#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(dirname "$0")/src:${PYTHONPATH:-}"
export ORCH_HOST="${ORCH_HOST:-0.0.0.0}"
export ORCH_PORT="${ORCH_PORT:-8088}"

exec uvicorn orchestrator.main:app --host "$ORCH_HOST" --port "$ORCH_PORT" --reload


