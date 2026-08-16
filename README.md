# llama-tracker

Experimental live dashboard and API for watching `llama.cpp` server completions from streamed logs.

`llama-tracker` sits beside a command-line `llama.cpp` server process, reads the logs that normally go to stdout/stderr, and turns the useful parts into a live browser view and JSON/SSE API. It is intended for running on the same Linux server as `llama.cpp`, then viewing completion status from another PC on the network.

## Experimental Status

This project is an experiment.

It currently parses the human-readable `llama.cpp` server log stream with regular expressions. That makes it lightweight and easy to run, but also means it can break if `llama.cpp` changes its log wording. Treat this as a practical prototype for observability, not a stable monitoring product.

## What It Shows

The tracker currently extracts:

- active completions by `slot id` and `task id`
- prompt processing status and progress
- prompt token count, current token count, final token count, and truncation status
- generation status from sampler and reasoning-budget log lines
- prompt/eval/total timing and tokens per second
- completed and cancelled tasks
- prompt cache size and limits
- checkpoint creation/restoration counts
- recent HTTP request completion lines
- the latest raw log line for debugging

## How It Works

`llama-tracker` can read logs in two ways:

- tail an existing log file
- read streamed log lines from stdin

It then exposes:

- `GET /` - web dashboard
- `GET /api/state` - current JSON snapshot
- `GET /api/events` - Server-Sent Events stream for live clients
- `GET /widget.js` - browser widget script used by the dashboard

The implementation uses only Python standard library modules.

## Requirements

- Linux
- Python 3.10 or newer
- `llama.cpp` server output that includes slot/task timing logs

No Python package installation is required.

## Quick Start

Run the tracker against the included sample log:

```bash
python3 -m llama_tracker.server logs.txt --host 0.0.0.0 --port 8765
```

Open the dashboard from another machine:

```text
http://SERVER_IP:8765/
```

Replace `SERVER_IP` with the Linux server's address.

## Run With llama.cpp

Add `-lv 4` to your `llama-server` command so its logs include the verbose slot/task detail this app parses.

The easiest option is the launcher script. Put your normal `llama-server` command after `--`:

```bash
scripts/run-llama-tracked.sh --host 0.0.0.0 --port 8765 -- \
  ./llama-server -m /models/model.gguf --host 0.0.0.0 --port 8080
```

The script:

- starts the tracker web server
- runs your `llama.cpp` command
- keeps the `llama.cpp` output visible in the terminal
- streams that same output to the tracker in real time
- prints the dashboard URL

To keep a raw copy of the logs as well:

```bash
scripts/run-llama-tracked.sh --log-copy ./llama-server.log -- \
  ./llama-server -m /models/model.gguf --host 0.0.0.0 --port 8080
```

You can also pipe output directly into the tracker:

```bash
./llama-server -m /models/model.gguf --host 0.0.0.0 --port 8080 2>&1 \
  | python3 -m llama_tracker.server - --host 0.0.0.0 --port 8765
```

## Track An Existing Log File

If `llama.cpp` already writes to a file:

```bash
python3 -m llama_tracker.server /path/to/llama-server.log --host 0.0.0.0 --port 8765
```

For a one-shot parse of the current file without waiting for new lines:

```bash
python3 -m llama_tracker.server /path/to/llama-server.log --no-follow
```

## API

Current state:

```bash
curl http://SERVER_IP:8765/api/state
```

Live event stream:

```bash
curl http://SERVER_IP:8765/api/events
```

The `/api/state` response includes active tasks, recent completed tasks, slots, recent requests, cache state, counters, and the last parsed line.

## MCP And Widget Ideas

The HTTP API is deliberately simple so other clients can sit on top of it.

A future MCP adapter could expose read-only tools such as:

- `get_llama_state`
- `list_llama_completions`
- `get_llama_completion(task_id)`

A browser widget can either poll `/api/state` or subscribe to `/api/events` and refresh when updates arrive.

## Security Notes

By default, examples bind to `0.0.0.0` so a client PC can reach the dashboard. Only do that on a trusted network.

This experiment does not currently include authentication, TLS, persistence, or access control. Put it behind a reverse proxy or firewall if running outside a private development environment.

## Development

Compile-check the Python files:

```bash
python3 -m py_compile llama_tracker/parser.py llama_tracker/server.py
```

Check the launcher script syntax:

```bash
bash -n scripts/run-llama-tracked.sh
```

## License

MIT. See [LICENSE](LICENSE).
