from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO

from .parser import LlamaLogParser


class EventHub:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except queue.Empty:
                    pass


class TrackerApp:
    def __init__(self, log_path: Path | None, follow: bool, stream: TextIO | None = None) -> None:
        self.log_path = log_path
        self.follow = follow
        self.stream = stream
        self.parser = LlamaLogParser()
        self.hub = EventHub()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()

    def start(self) -> None:
        thread = threading.Thread(target=self._read_log, name="log-reader", daemon=True)
        thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return self.parser.state.snapshot()

    def _read_log(self) -> None:
        if self.stream is not None:
            self._read_stream(self.stream)
            return

        if self.log_path is None:
            return

        while not self.log_path.exists() and not self._stop.is_set():
            time.sleep(0.5)

        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            while not self._stop.is_set():
                line = handle.readline()
                if not line:
                    if not self.follow:
                        return
                    time.sleep(0.25)
                    continue
                self._parse_and_publish(line)

    def _read_stream(self, stream: TextIO) -> None:
        while not self._stop.is_set():
            line = stream.readline()
            if not line:
                return
            self._parse_and_publish(line)

    def _parse_and_publish(self, line: str) -> None:
        with self._state_lock:
            event = self.parser.parse_line(line)
        if event:
            self.hub.publish(event)


class Handler(BaseHTTPRequestHandler):
    app: TrackerApp

    def do_HEAD(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_headers("text/html; charset=utf-8", len(INDEX_HTML.encode("utf-8")))
        elif self.path == "/api/state":
            body = json.dumps(self.app.snapshot(), indent=2).encode("utf-8")
            self._send_headers("application/json", len(body), cors=True)
        elif self.path == "/widget.js":
            self._send_headers("application/javascript; charset=utf-8", len(WIDGET_JS.encode("utf-8")), cors=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        elif self.path == "/api/state":
            self._send_json(self.app.snapshot())
        elif self.path == "/api/events":
            self._send_events()
        elif self.path == "/widget.js":
            self._send_js(WIDGET_JS)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._send_headers("application/json", len(body), cors=True)
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self._send_headers("text/html; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_js(self, js: str) -> None:
        body = js.encode("utf-8")
        self._send_headers("application/javascript; charset=utf-8", len(body), cors=True)
        self.wfile.write(body)

    def _send_headers(self, content_type: str, content_length: int, cors: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _send_events(self) -> None:
        subscriber = self.app.hub.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(b"event: snapshot\n")
            self.wfile.write(f"data: {json.dumps(self.app.snapshot())}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                event = subscriber.get(timeout=15)
                self.wfile.write(b"event: update\n")
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except queue.Empty:
            pass
        finally:
            self.app.hub.unsubscribe(subscriber)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llama.cpp tracker</title>
  <script src="/widget.js" defer></script>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #111317; color: #edf1f5; }
    body { margin: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 18px; }
    h1 { font-size: 24px; font-weight: 650; margin: 0; }
    .muted { color: #9da7b2; font-size: 13px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric, .panel { background: #1b2028; border: 1px solid #303844; border-radius: 8px; }
    .metric { padding: 12px; }
    .metric span { display: block; color: #9da7b2; font-size: 12px; }
    .metric strong { display: block; margin-top: 4px; font-size: 22px; font-weight: 650; }
    .grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); gap: 16px; }
    .panel { overflow: hidden; }
    .panel h2 { margin: 0; padding: 12px 14px; border-bottom: 1px solid #303844; font-size: 15px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #29313b; vertical-align: top; }
    th { color: #9da7b2; font-weight: 550; }
    .bar { height: 8px; background: #2b333e; border-radius: 999px; overflow: hidden; min-width: 90px; }
    .bar div { height: 100%; background: #5bc0be; }
    .status { display: inline-block; padding: 2px 7px; border: 1px solid #3d4856; border-radius: 999px; color: #d5dde6; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; padding: 12px; color: #c3cad3; font-size: 12px; }
    @media (max-width: 800px) { main { padding: 14px; } .grid { grid-template-columns: 1fr; } header { display: block; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>llama.cpp tracker</h1>
      <div class="muted" id="lastEvent">Waiting for logs</div>
    </header>
    <section class="metrics" id="metrics"></section>
    <section class="grid">
      <div class="panel">
        <h2>Active completions</h2>
        <div id="active"></div>
      </div>
      <div class="panel">
        <h2>Recent completions</h2>
        <div id="completed"></div>
      </div>
    </section>
    <section class="panel" style="margin-top:16px">
      <h2>Last log line</h2>
      <pre id="lastLine"></pre>
    </section>
  </main>
</body>
</html>
"""


WIDGET_JS = """
const state = { data: null };

function fmtNumber(value) {
  return value === null || value === undefined ? "-" : Number(value).toLocaleString();
}

function fmtMs(value) {
  if (value === null || value === undefined) return "-";
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value.toFixed(0)}ms`;
}

function taskRow(task) {
  const progress = Math.round((task.prompt_progress || 0) * 100);
  return `<tr>
    <td>#${task.task_id}<br><span class="muted">slot ${task.slot_id}</span></td>
    <td><span class="status">${task.status}</span></td>
    <td><div class="bar"><div style="width:${progress}%"></div></div><span class="muted">${progress}% prompt</span></td>
    <td>${fmtNumber(task.prompt_tokens)}</td>
    <td>${fmtNumber(task.generated_tokens ?? task.eval_tokens)}</td>
    <td>${fmtMs(task.total_ms)}</td>
    <td>${task.eval_tps ? task.eval_tps.toFixed(2) : "-"}</td>
  </tr>`;
}

function renderTable(tasks, empty) {
  if (!tasks || tasks.length === 0) return `<pre>${empty}</pre>`;
  return `<table>
    <thead><tr><th>Task</th><th>Status</th><th>Progress</th><th>Prompt</th><th>Output</th><th>Total</th><th>tok/s</th></tr></thead>
    <tbody>${tasks.map(taskRow).join("")}</tbody>
  </table>`;
}

function render(data) {
  state.data = data;
  const active = data.active || [];
  const completed = data.completed || [];
  const cache = data.cache || {};
  document.getElementById("metrics").innerHTML = `
    <div class="metric"><span>Active</span><strong>${active.length}</strong></div>
    <div class="metric"><span>Completed</span><strong>${fmtNumber(data.counters.completed)}</strong></div>
    <div class="metric"><span>Tasks seen</span><strong>${fmtNumber(data.counters.tasks)}</strong></div>
    <div class="metric"><span>Cache</span><strong>${cache.mib ? `${cache.mib.toFixed(0)} MiB` : "-"}</strong></div>
    <div class="metric"><span>Log lines</span><strong>${fmtNumber(data.counters.lines)}</strong></div>`;
  document.getElementById("active").innerHTML = renderTable(active, "No active completions.");
  document.getElementById("completed").innerHTML = renderTable(completed.slice(0, 8), "No completed completions yet.");
  document.getElementById("lastLine").textContent = data.last_line || "";
  document.getElementById("lastEvent").textContent = data.last_event_at ? `Last event ${data.last_event_at}` : "Waiting for logs";
}

async function loadState() {
  const response = await fetch("/api/state");
  render(await response.json());
}

loadState();
const events = new EventSource("/api/events");
events.addEventListener("snapshot", event => render(JSON.parse(event.data)));
events.addEventListener("update", () => loadState());
"""


def main() -> None:
    parser = ArgumentParser(description="Track llama.cpp streamed logs and expose a small dashboard/API.")
    parser.add_argument(
        "log_path",
        nargs="?",
        default=os.environ.get("LLAMA_TRACKER_LOG", "logs.txt"),
        help="Path to a llama.cpp log file, or '-' to read streamed logs from stdin.",
    )
    parser.add_argument("--host", default=os.environ.get("LLAMA_TRACKER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLAMA_TRACKER_PORT", "8765")))
    parser.add_argument("--no-follow", action="store_true", help="Read the current file and exit the log reader instead of tailing.")
    args = parser.parse_args()

    log_path = None if args.log_path == "-" else Path(args.log_path)
    app = TrackerApp(log_path, follow=not args.no_follow, stream=sys.stdin if args.log_path == "-" else None)
    Handler.app = app
    app.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    source = "stdin" if args.log_path == "-" else args.log_path
    print(f"llama-tracker serving http://{args.host}:{args.port} from {source}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
