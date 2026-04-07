#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_SCRIPT="$ROOT_DIR/scripts/nemotron_streaming_server.py"
PORT="${NEMOTRON_PORT:-8765}"
PATTERN="$SERVER_SCRIPT --host"

if pgrep -f "$SERVER_SCRIPT" >/dev/null 2>&1; then
  PID="$(pgrep -f "$SERVER_SCRIPT" | head -n 1)"
  echo "Nemotron server is already running with PID $PID."
  echo "Stop it first with: kill $PID"
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-nemotron}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-nemotron}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/media/yaroslav/DATA/ai_models/cache}"
export HF_HOME="${HF_HOME:-/media/yaroslav/DATA/ai_models/huggingface}"
export NEMOTRON_CHUNK_SIZE_MS="${NEMOTRON_CHUNK_SIZE_MS:-160}"
export NEMOTRON_ATT_RIGHT_CONTEXT="${NEMOTRON_ATT_RIGHT_CONTEXT:-1}"
export NEMOTRON_FINALIZE_SILENCE_SEC="${NEMOTRON_FINALIZE_SILENCE_SEC:-0.45}"

mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

exec "$ROOT_DIR/.venv-nemotron/bin/python" "$SERVER_SCRIPT" --host 0.0.0.0 --port "$PORT"
