# Repository Guidelines

## Project Structure & Module Organization

`llama_tracker/` is the application package. `parser.py` converts streamed
`llama.cpp` log lines into task and metrics state; `server.py` tails a file or
stdin and serves the dashboard, JSON, and SSE endpoints. Keep parsing rules and
state changes in `parser.py`; keep HTTP, process, and host-metrics concerns in
`server.py`. `tests/` contains executable parser regression checks. `scripts/`
contains shell launchers, while root-level `logs*.txt` files are sample or local
capture data, not application code.

## Build, Test, and Development Commands

The project targets Python 3.10+ and has no third-party dependencies.

```bash
python3 -m py_compile llama_tracker/parser.py llama_tracker/server.py
python3 tests/test_timestamp_prefix.py
bash -n scripts/run-llama-tracked.sh
python3 -m llama_tracker.server logs.txt --host 127.0.0.1 --port 8765
```

The first command catches syntax errors, the second runs the parser smoke
checks, and the third validates launcher-shell syntax. The final command starts
the dashboard against the included sample log.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations for new public functions and
stateful attributes, and standard-library modules unless a dependency is
clearly justified. Follow existing `snake_case` names for functions, variables,
and modules; use `PascalCase` for classes (for example, `LlamaLogParser`). Keep
log-pattern handling narrow and defensive: malformed or unfamiliar lines should
not stop ingestion. Prefer small, focused helpers over broad parsing changes.

## Testing Guidelines

Add a regression case in `tests/` whenever supporting a new log format or
fixing a parser bug. Name test files `test_<behavior>.py` and use direct
assertions that verify both the returned event and resulting parser state. Tests
are runnable scripts rather than a configured test framework; ensure each can
run with `python3 tests/test_<behavior>.py`.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, such as `Add cost estimator`
and `Fix tracker cost estimation UI`. Keep each commit focused and avoid
committing generated logs or local launcher scripts. Pull requests should
explain the behavior change, identify relevant log samples or issue links, list
commands run, and include a dashboard screenshot for visible UI changes.

## Security & Configuration

Binding to `0.0.0.0` exposes the unauthenticated dashboard to the network. Use
`127.0.0.1` for local development; use a firewall or reverse proxy before
exposing it beyond a trusted network.
