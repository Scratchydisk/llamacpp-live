#!/usr/bin/env bash
# Monitor-only mode: attach the llama-tracker dashboard to an ALREADY RUNNING
# llama.cpp server by tailing its log file. Starts no llama process itself.
#
# Usage:
#   scripts/run-monitor-only.sh [--host HOST] [--port PORT] [--log-file FILE]
#
# Example (Glimmer server in tmux session "glimmer"):
#   scripts/run-monitor-only.sh --port 8765
#   scripts/run-monitor-only.sh --log-file /mnt/llm-fast/glimmer/server.log --port 8766
#
# Environment:
#   LLAMA_TRACKER_HOST      Default tracker bind host, default 0.0.0.0
#   LLAMA_TRACKER_PORT      Default tracker web port, default 8765
#   LLAMA_TRACKER_LOG_FILE  Default llama.cpp log file to tail
set -euo pipefail

HOST="${LLAMA_TRACKER_HOST:-0.0.0.0}"
PORT="${LLAMA_TRACKER_PORT:-8765}"
LOG_FILE="${LLAMA_TRACKER_LOG_FILE:-/mnt/llm-fast/glimmer/server.log}"

usage() {
  cat <<USAGE
Usage:
  scripts/run-monitor-only.sh [--host HOST] [--port PORT] [--log-file FILE]

Tails FILE (default: \$LLAMA_TRACKER_LOG_FILE or /mnt/llm-fast/glimmer/server.log)
into the tracker. No llama.cpp process is started; point it at the log of the
server that is already running.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "llama-tracker will follow: $LOG_FILE" >&2
echo "It can start before the file exists and will reopen it after rotation." >&2
exec env PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m llama_tracker.server "$LOG_FILE" --host "$HOST" --port "$PORT"
