#!/usr/bin/env bash
set -euo pipefail

HOST="${LLAMA_TRACKER_HOST:-0.0.0.0}"
PORT="${LLAMA_TRACKER_PORT:-8765}"
LOG_COPY="${LLAMA_TRACKER_LOG_COPY:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run-llama-tracked.sh [--host HOST] [--port PORT] [--log-copy FILE] -- LLAMA_COMMAND [ARGS...]

Example:
  scripts/run-llama-tracked.sh --host 0.0.0.0 --port 8765 -- \
    ./llama-server -m /models/model.gguf --host 0.0.0.0 --port 8080

Environment:
  LLAMA_TRACKER_HOST      Default tracker bind host, default 0.0.0.0
  LLAMA_TRACKER_PORT      Default tracker web port, default 8765
  LLAMA_TRACKER_LOG_COPY  Optional file to append a copy of llama.cpp output
EOF
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
    --log-copy)
      LOG_COPY="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option before --: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing llama.cpp command after --" >&2
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIFO="$(mktemp -u "${TMPDIR:-/tmp}/llama-tracker.XXXXXX.fifo")"
mkfifo "$FIFO"

TRACKER_PID=""
cleanup() {
  if [[ -n "$TRACKER_PID" ]] && kill -0 "$TRACKER_PID" 2>/dev/null; then
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  rm -f "$FIFO"
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m llama_tracker.server "$FIFO" --host "$HOST" --port "$PORT" &
TRACKER_PID="$!"

echo "llama-tracker dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/" >&2
echo "llama-tracker local URL:  http://127.0.0.1:${PORT}/" >&2

if [[ -n "$LOG_COPY" ]]; then
  "$@" 2>&1 | tee -a "$LOG_COPY" | tee "$FIFO"
else
  "$@" 2>&1 | tee "$FIFO"
fi
