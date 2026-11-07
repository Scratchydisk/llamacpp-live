#!/usr/bin/env bash
# Start the always-on llama.cpp log monitor.
#
# Examples:
#   scripts/start-monitor.sh
#   scripts/start-monitor.sh /path/to/llama-server.log 0.0.0.0 8765
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${1:-${LLAMA_TRACKER_LOG_FILE:-/var/log/llama/server.log}}"
HOST="${2:-${LLAMA_TRACKER_HOST:-0.0.0.0}}"
PORT="${3:-${LLAMA_TRACKER_PORT:-8765}}"

exec "$ROOT_DIR/scripts/run-monitor-only.sh" \
  --log-file "$LOG_FILE" \
  --host "$HOST" \
  --port "$PORT"
