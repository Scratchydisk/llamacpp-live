from __future__ import annotations

from argparse import ArgumentParser
import copy
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
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
    def __init__(
        self,
        log_path: Path | None,
        follow: bool,
        stream: TextIO | None = None,
        tracker_host: str = "127.0.0.1",
        tracker_port: int = 8765,
    ) -> None:
        self.log_path = log_path
        self.follow = follow
        self.stream = stream
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        self.started_at = time.time()
        self.parser = LlamaLogParser()
        self.hub = EventHub()
        self._state_lock = threading.Lock()
        self._cpu_lock = threading.Lock()
        self._last_cpu_times: tuple[int, int] | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        thread = threading.Thread(target=self._read_log, name="log-reader", daemon=True)
        thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = copy.deepcopy(self.parser.state.snapshot())
        snapshot["metadata"]["tracker"] = self._tracker_info()
        snapshot["metadata"]["live_gpu"] = live_gpu_status()
        snapshot["metadata"]["live_cpu"] = self._live_cpu_status()
        return snapshot

    def _tracker_info(self) -> dict[str, Any]:
        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        ips = local_ip_addresses()
        display_host = ips[0] if self.tracker_host == "0.0.0.0" and ips else self.tracker_host
        return {
            "hostname": hostname,
            "fqdn": fqdn,
            "pid": os.getpid(),
            "dashboard_host": self.tracker_host,
            "dashboard_port": self.tracker_port,
            "dashboard_url": f"http://{display_host}:{self.tracker_port}",
            "local_ips": ips,
            "log_source": "stdin" if self.stream is not None else str(self.log_path) if self.log_path else None,
            "follow": self.follow,
            "started_at_unix": self.started_at,
            "uptime_seconds": max(0, int(time.time() - self.started_at)),
        }

    def _live_cpu_status(self) -> dict[str, Any]:
        with self._cpu_lock:
            cpu_percent = None
            current_times = cpu_times()
            if current_times and self._last_cpu_times:
                previous_idle, previous_total = self._last_cpu_times
                idle, total = current_times
                total_delta = total - previous_total
                idle_delta = idle - previous_idle
                if total_delta > 0:
                    cpu_percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
            if current_times:
                self._last_cpu_times = current_times

        load_1, load_5, load_15 = os.getloadavg()
        cpu_count = os.cpu_count()
        return {
            "model": cpu_model_name(),
            "logical_cpus": cpu_count,
            "load_1m": load_1,
            "load_5m": load_5,
            "load_15m": load_15,
            "load_1m_per_cpu": load_1 / cpu_count if cpu_count else None,
            "cpu_percent": cpu_percent,
            "updated_at_unix": time.time(),
        }

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
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(b"event: update\n")
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
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
    .metrics { position: sticky; top: 0; z-index: 10; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 0 -4px 16px; padding: 8px 4px; background: #111317; }
    .metric, .panel { background: #1b2028; border: 1px solid #303844; border-radius: 8px; }
    .metric { padding: 12px; min-height: 66px; }
    .metric span { display: block; color: #9da7b2; font-size: 12px; }
    .metric strong { display: block; margin-top: 4px; font-size: 22px; font-weight: 650; }
    .metric svg { display: block; width: 100%; height: 28px; margin-top: 6px; }
    .metric small { display: block; color: #9da7b2; font-size: 11px; margin-top: 4px; }
    .grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); gap: 16px; }
    .panel { overflow: hidden; }
    .panel h2 { margin: 0; font-size: 15px; }
    .panel > h2 { padding: 12px 14px; border-bottom: 1px solid #303844; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #303844; }
    .panel-head h2 { padding: 0; border: 0; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    select, button { background: #171c23; color: #edf1f5; border: 1px solid #3d4856; border-radius: 6px; padding: 6px 8px; font: inherit; font-size: 12px; }
    button { cursor: pointer; }
    details.panel summary { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #303844; cursor: pointer; font-size: 15px; font-weight: 650; }
    details.panel summary::after { content: "Show"; color: #9da7b2; font-size: 12px; font-weight: 500; }
    details.panel[open] summary::after { content: "Hide"; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #29313b; vertical-align: top; }
    th { color: #9da7b2; font-weight: 550; }
    .bar { height: 8px; background: #2b333e; border-radius: 999px; overflow: hidden; min-width: 90px; }
    .bar div { height: 100%; background: #5bc0be; }
    .status { display: inline-block; padding: 2px 7px; border: 1px solid #3d4856; border-radius: 999px; color: #d5dde6; }
    .status.live { border-color: #2f9e44; color: #9be7ad; }
    .status.stale { border-color: #b08900; color: #ffd166; }
    .status.offline { border-color: #b34040; color: #ffadad; }
    .details-row td { padding: 0; background: #171c23; }
    .task-details { padding: 10px 12px 12px; color: #c3cad3; }
    .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px 14px; margin-bottom: 10px; font-size: 12px; }
    .detail-grid span { color: #9da7b2; display: block; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; padding: 12px; }
    .meta-card { border: 1px solid #303844; border-radius: 8px; overflow: hidden; background: #171c23; }
    .meta-card h3 { margin: 0; padding: 10px 12px; border-bottom: 1px solid #29313b; font-size: 13px; color: #dce3eb; }
    .meta-card table { font-size: 12px; }
    .meta-card th { width: 42%; }
    .warn { color: #ffd166; }
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
    <details class="panel" style="margin-top:16px">
      <summary>Server and model</summary>
      <div id="metadata"></div>
    </details>
    <section class="panel" style="margin-top:16px">
      <div class="panel-head">
        <h2>Active completions</h2>
      </div>
        <div id="active"></div>
    </section>
    <section class="panel" style="margin-top:16px">
      <div class="panel-head">
        <h2>Recent completions</h2>
        <div class="controls">
          <select id="completedFilter" aria-label="Completed filter">
            <option value="all">All statuses</option>
            <option value="completed">Completed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <select id="completedLimit" aria-label="Completed limit">
            <option value="8">Last 8</option>
            <option value="25">Last 25</option>
            <option value="100">Last 100</option>
            <option value="all">All loaded</option>
          </select>
        </div>
      </div>
        <div id="completed"></div>
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
const DEFAULT_UI = { completedFilter: "all", completedLimit: "8", metadataOpen: false };
const state = {
  data: null,
  history: { cpu: [], gpu: {}, gpuMemory: {}, evalTps: [] },
  seenTps: new Set(),
  openTaskDetails: new Set(),
  loading: false,
  failures: 0,
  ui: loadUiState(),
};
const HISTORY_LIMIT = 60;
const STALE_SECONDS = 30;

function fmtNumber(value) {
  return value === null || value === undefined ? "-" : Number(value).toLocaleString();
}

function fmtMs(value) {
  if (value === null || value === undefined) return "-";
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value.toFixed(0)}ms`;
}

function loadUiState() {
  try {
    return { ...DEFAULT_UI, ...JSON.parse(localStorage.getItem("llamaTrackerUi") || "{}") };
  } catch {
    return { ...DEFAULT_UI };
  }
}

function saveUiState() {
  localStorage.setItem("llamaTrackerUi", JSON.stringify(state.ui));
}

function pushSample(series, value) {
  if (typeof value !== "number" || Number.isNaN(value)) return;
  series.push(Math.max(0, Math.min(100, value)));
  if (series.length > HISTORY_LIMIT) series.splice(0, series.length - HISTORY_LIMIT);
}

function rememberStats(metadata, completed) {
  const liveCpu = metadata?.live_cpu || {};
  const liveGpu = metadata?.live_gpu || {};
  pushSample(state.history.cpu, liveCpu.cpu_percent);
  for (const device of liveGpu.devices || []) {
    const index = device.index ?? "gpu";
    state.history.gpu[index] ||= [];
    state.history.gpuMemory[index] ||= [];
    pushSample(state.history.gpu[index], device.utilization_gpu_pct);
    if (typeof device.memory_used_mib === "number" && typeof device.memory_total_mib === "number" && device.memory_total_mib > 0) {
      pushSample(state.history.gpuMemory[index], (device.memory_used_mib / device.memory_total_mib) * 100);
    }
  }
  for (const task of (completed || []).slice().reverse()) {
    if (task.task_id === undefined || state.seenTps.has(task.task_id)) continue;
    state.seenTps.add(task.task_id);
    if (typeof task.eval_tps === "number") pushSample(state.history.evalTps, task.eval_tps);
  }
}

function sparkline(values, maxValue = 100) {
  const width = 120;
  const height = 28;
  if (!values.length) {
    return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="M0 ${height - 1} H${width}" stroke="#303844" fill="none"/></svg>`;
  }
  const scaleMax = Math.max(maxValue, ...values, 1);
  const points = values.map((value, index) => {
    const x = values.length === 1 ? width : (index / (values.length - 1)) * width;
    const y = height - (value / scaleMax) * (height - 3) - 1.5;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
    <polyline points="${points}" fill="none" stroke="#5bc0be" stroke-width="2" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

function metric(label, value, sparkValues, note, maxValue = 100) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${value}</strong>${sparkValues ? sparkline(sparkValues, maxValue) : ""}${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.length ? value.map(escapeHtml).join(", ") : "-";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? fmtNumber(value) : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return escapeHtml(value);
}

function latest(values) {
  return values.length ? values[values.length - 1] : null;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function statusInfo(data, metadata) {
  if (state.failures > 0) return { label: "Retrying", cls: "offline", detail: `${state.failures} failed` };
  const liveCpuAt = metadata.live_cpu?.updated_at_unix ? metadata.live_cpu.updated_at_unix * 1000 : null;
  const liveGpuAt = metadata.live_gpu?.updated_at_unix ? metadata.live_gpu.updated_at_unix * 1000 : null;
  const liveAt = Math.max(liveCpuAt || 0, liveGpuAt || 0);
  const eventAt = data.last_event_at ? Date.parse(data.last_event_at) : null;
  const newest = Math.max(liveAt || 0, eventAt || 0);
  if (!newest) return { label: "Starting", cls: "stale", detail: "waiting" };
  const ageSeconds = Math.max(0, Math.round((Date.now() - newest) / 1000));
  if (ageSeconds > STALE_SECONDS) return { label: "Stale", cls: "stale", detail: `${ageSeconds}s old` };
  return { label: "Live", cls: "live", detail: `${ageSeconds}s old` };
}

function rows(items) {
  const visible = items.filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (!visible.length) return `<tr><td colspan="2" class="muted">Waiting for startup metadata.</td></tr>`;
  return visible.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${fmtValue(value)}</td></tr>`).join("");
}

function metaCard(title, items) {
  return `<div class="meta-card"><h3>${escapeHtml(title)}</h3><table><tbody>${rows(items)}</tbody></table></div>`;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    return metaCard("Startup notices", [["Status", "No startup warnings parsed"]]);
  }
  return `<div class="meta-card"><h3>Startup notices</h3><table><tbody>${warnings.slice(0, 8).map(warn => `
    <tr><th class="warn">${escapeHtml(warn.category || "notice")}</th><td>${escapeHtml(warn.message || "")}</td></tr>
  `).join("")}</tbody></table></div>`;
}

function renderLiveGpuCards(liveGpu) {
  if (!liveGpu || !liveGpu.available) {
    return metaCard("Live GPU", [["Status", liveGpu?.message || "nvidia gpu not found"]]);
  }

  return (liveGpu.devices || []).map(device => metaCard(`Live GPU ${device.index ?? ""}`.trim(), [
    ["GPU", device.name],
    ["Driver", device.driver_version],
    ["Utilization", device.utilization_gpu_pct !== undefined ? `${device.utilization_gpu_pct}%` : undefined],
    ["Memory", device.memory_used_mib !== undefined ? `${fmtNumber(device.memory_used_mib)} / ${fmtNumber(device.memory_total_mib)} MiB` : undefined],
    ["Free memory", device.memory_free_mib !== undefined ? `${fmtNumber(device.memory_free_mib)} MiB` : undefined],
    ["Temperature", device.temperature_c !== undefined ? `${device.temperature_c} C` : undefined],
    ["Power", device.power_draw_w !== undefined ? `${device.power_draw_w} / ${device.power_limit_w} W` : undefined],
    ["Processes", liveGpu.processes ? liveGpu.processes.filter(process => process.gpu_uuid === device.uuid).length : undefined],
  ])).join("");
}

function renderMetadata(metadata) {
  const meta = metadata || {};
  const tracker = meta.tracker || {};
  const server = meta.server || {};
  const build = meta.build || {};
  const hardware = meta.hardware || {};
  const model = meta.model || {};
  const runtime = meta.runtime || {};
  const memory = meta.memory || {};
  const buffers = memory.buffers || {};
  const vision = meta.vision || {};
  const liveGpu = meta.live_gpu || {};
  const liveCpu = meta.live_cpu || {};
  const gpu = (hardware.devices || [])[0] || {};

  return `<div class="meta-grid">
    ${metaCard("Tracker host", [
      ["Dashboard", tracker.dashboard_url],
      ["Hostname", tracker.hostname],
      ["Local IPs", tracker.local_ips],
      ["PID", tracker.pid],
      ["Uptime", tracker.uptime_seconds !== undefined ? `${tracker.uptime_seconds}s` : undefined],
      ["Log source", tracker.log_source],
    ])}
    ${metaCard("llama.cpp server", [
      ["Server URL", server.llama_url],
      ["Bind host", server.llama_host],
      ["Port", server.llama_port],
      ["SSL", server.ssl],
      ["HTTP threads", server.http_threads],
      ["Build", build.build_info],
      ["CPU threads", build.n_threads !== undefined ? `${build.n_threads} / ${build.threads_total}` : undefined],
    ])}
    ${metaCard("Model", [
      ["Name", model.name || model.general_name],
      ["File", model.filename],
      ["Architecture", model.architecture],
      ["Type", model.model_type],
      ["Params", model.model_params],
      ["Quantization", model.file_type],
      ["File size", model.file_size],
      ["GGUF", model.gguf_version || model.file_format],
      ["License", model.license],
      ["Quantized by", model.quantized_by],
    ])}
    ${metaCard("GPU and memory", [
      ["GPU", gpu.name || hardware.active_device_name],
      ["VRAM", gpu.vram_mib ? `${fmtNumber(gpu.vram_mib)} MiB` : undefined],
      ["CUDA capability", gpu.compute_capability],
      ["Projected use", memory.projected_device_use_mib ? `${fmtNumber(memory.projected_device_use_mib)} MiB` : undefined],
      ["Free after fit", memory.projected_free_after_fit_mib ? `${fmtNumber(memory.projected_free_after_fit_mib)} MiB` : undefined],
      ["CUDA model buffer", buffers.cuda0_model_buffer_size_mib ? `${fmtNumber(buffers.cuda0_model_buffer_size_mib)} MiB` : undefined],
      ["KV buffer", buffers.cuda0_kv_buffer_size_mib ? `${fmtNumber(buffers.cuda0_kv_buffer_size_mib)} MiB` : undefined],
      ["Compute buffer", buffers.cuda0_compute_buffer_size_mib ? `${fmtNumber(buffers.cuda0_compute_buffer_size_mib)} MiB` : undefined],
    ])}
    ${renderLiveGpuCards(liveGpu)}
    ${metaCard("Live CPU", [
      ["Model", liveCpu.model],
      ["Logical CPUs", liveCpu.logical_cpus],
      ["CPU utilization", liveCpu.cpu_percent !== null && liveCpu.cpu_percent !== undefined ? `${liveCpu.cpu_percent.toFixed(1)}%` : "sampling"],
      ["Load 1m", liveCpu.load_1m !== undefined ? liveCpu.load_1m.toFixed(2) : undefined],
      ["Load 5m", liveCpu.load_5m !== undefined ? liveCpu.load_5m.toFixed(2) : undefined],
      ["Load 15m", liveCpu.load_15m !== undefined ? liveCpu.load_15m.toFixed(2) : undefined],
      ["Load / CPU", liveCpu.load_1m_per_cpu !== undefined && liveCpu.load_1m_per_cpu !== null ? liveCpu.load_1m_per_cpu.toFixed(2) : undefined],
    ])}
    ${metaCard("Runtime", [
      ["Slots", runtime.n_slots || runtime.n_seq_max],
      ["Parallel", runtime.n_parallel],
      ["Context", runtime.n_ctx],
      ["Model max context", model.n_ctx_train],
      ["Batch", runtime.n_batch],
      ["Microbatch", runtime.n_ubatch],
      ["Flash attention", runtime.flash_attn],
      ["KV unified", runtime.kv_unified],
      ["GPU layers", runtime.offloaded_layers !== undefined ? `${runtime.offloaded_layers}/${runtime.total_layers}` : undefined],
      ["Prompt cache", runtime.prompt_cache_enabled ? `${fmtNumber(runtime.prompt_cache_limit_mib)} MiB` : runtime.prompt_cache_enabled],
      ["Thinking template", runtime.chat_template_thinking],
    ])}
    ${metaCard("Vision", [
      ["Multimodal", model.multimodal || vision.has_vision_encoder],
      ["Projector", vision.mmproj_filename || vision.projector],
      ["Backend", vision.backend],
      ["Image size", vision.image_size],
      ["Patch size", vision.patch_size],
      ["Image pixels", vision.image_min_pixels && vision.image_max_pixels ? `${fmtNumber(vision.image_min_pixels)} - ${fmtNumber(vision.image_max_pixels)}` : undefined],
      ["Warmup image", vision.warmup_image_width ? `${vision.warmup_image_width} x ${vision.warmup_image_height}` : undefined],
    ])}
    ${renderWarnings(meta.warnings)}
  </div>`;
}

function taskKey(task) {
  return task.task_id !== undefined ? `task-${task.task_id}` : `slot-${task.slot_id}-${task.created_at || ""}`;
}

function taskRow(task) {
  const progress = Math.round((task.prompt_progress || 0) * 100);
  const key = escapeHtml(taskKey(task));
  const open = state.openTaskDetails.has(taskKey(task)) ? " open" : "";
  return `<tr>
    <td>#${task.task_id}<br><span class="muted">slot ${task.slot_id}</span></td>
    <td><span class="status">${task.status}</span></td>
    <td><div class="bar"><div style="width:${progress}%"></div></div><span class="muted">${progress}% prompt</span></td>
    <td>${fmtNumber(task.prompt_tokens)}</td>
    <td>${fmtNumber(task.generated_tokens ?? task.eval_tokens)}</td>
    <td>${fmtMs(task.total_ms)}</td>
    <td>${task.eval_tps ? task.eval_tps.toFixed(2) : "-"}</td>
  </tr>
  <tr class="details-row"><td colspan="7">
    <details data-task-key="${key}"${open}>
      <summary class="task-details">Details</summary>
      <div class="task-details">
        <div class="detail-grid">
          <div><span>Created</span>${escapeHtml(task.created_at || "-")}</div>
          <div><span>Updated</span>${escapeHtml(task.updated_at || "-")}</div>
          <div><span>Completed</span>${escapeHtml(task.completed_at || "-")}</div>
          <div><span>Prompt eval</span>${fmtMs(task.prompt_eval_ms)} / ${fmtNumber(task.prompt_eval_tokens)} tokens</div>
          <div><span>Generation</span>${fmtMs(task.eval_ms)} / ${fmtNumber(task.eval_tokens)} tokens</div>
          <div><span>Checkpoint</span>${fmtNumber(task.checkpoints_created)} created, ${fmtNumber(task.checkpoints_restored)} restored</div>
          <div><span>Request</span>${task.request ? `${escapeHtml(task.request.method)} ${escapeHtml(task.request.path)} ${escapeHtml(task.request.status)}` : "-"}</div>
          <div><span>Client</span>${task.request ? escapeHtml(task.request.client) : "-"}</div>
        </div>
        <pre>${escapeHtml((task.raw_tail || []).join("\\n"))}</pre>
      </div>
    </details>
  </td></tr>`;
}

function renderTable(tasks, empty) {
  if (!tasks || tasks.length === 0) return `<pre>${empty}</pre>`;
  return `<table>
    <thead><tr><th>Task</th><th>Status</th><th>Progress</th><th>Prompt</th><th>Output</th><th>Total</th><th>tok/s</th></tr></thead>
    <tbody>${tasks.map(taskRow).join("")}</tbody>
  </table>`;
}

function rememberOpenTaskDetails() {
  document.querySelectorAll("details[data-task-key]").forEach(details => {
    if (details.open) {
      state.openTaskDetails.add(details.dataset.taskKey);
    } else {
      state.openTaskDetails.delete(details.dataset.taskKey);
    }
  });
}

function pruneOpenTaskDetails(tasks) {
  const visibleKeys = new Set((tasks || []).map(taskKey));
  state.openTaskDetails = new Set([...state.openTaskDetails].filter(key => visibleKeys.has(key)));
}

function filteredCompleted(completed) {
  const filtered = state.ui.completedFilter === "all"
    ? completed
    : completed.filter(task => task.status === state.ui.completedFilter);
  if (state.ui.completedLimit === "all") return filtered;
  return filtered.slice(0, Number(state.ui.completedLimit));
}

function renderMetricStrip(data, active, completed, cache, metadata) {
  const liveCpu = metadata.live_cpu || {};
  const liveGpu = metadata.live_gpu || {};
  const status = statusInfo(data, metadata);
  const cards = [
    metric("Status", `<span class="status ${status.cls}">${status.label}</span>`, null, status.detail),
    metric("Active", active.length),
    metric("CPU", typeof liveCpu.cpu_percent === "number" ? `${liveCpu.cpu_percent.toFixed(1)}%` : "-", state.history.cpu),
  ];

  for (const device of liveGpu.devices || []) {
    const index = device.index ?? "gpu";
    const utilHistory = state.history.gpu[index] || [];
    const memHistory = state.history.gpuMemory[index] || [];
    const memoryPct = latest(memHistory);
    cards.push(metric(`GPU ${index}`, typeof device.utilization_gpu_pct === "number" ? `${device.utilization_gpu_pct}%` : "-", utilHistory, device.name));
    cards.push(metric(`VRAM ${index}`, memoryPct !== null ? `${memoryPct.toFixed(1)}%` : "-", memHistory, `${fmtNumber(device.memory_used_mib)} / ${fmtNumber(device.memory_total_mib)} MiB`));
  }

  const avgTps = average(state.history.evalTps);
  cards.push(metric("tok/s", avgTps !== null ? avgTps.toFixed(2) : "-", state.history.evalTps, "completed avg", Math.max(25, ...state.history.evalTps)));
  cards.push(metric("Completed", fmtNumber(data.counters.completed)));
  cards.push(metric("Cache", cache.mib ? `${cache.mib.toFixed(0)} MiB` : "-"));
  return cards.join("");
}

function render(data) {
  rememberOpenTaskDetails();
  state.data = data;
  const active = (data.active || []).slice().sort((left, right) => (left.slot_id ?? 0) - (right.slot_id ?? 0));
  const completed = data.completed || [];
  const cache = data.cache || {};
  const metadata = data.metadata || {};
  rememberStats(metadata, completed);
  const completedToShow = filteredCompleted(completed);
  pruneOpenTaskDetails([...active, ...completedToShow]);
  document.getElementById("metrics").innerHTML = renderMetricStrip(data, active, completed, cache, metadata);
  document.getElementById("metadata").innerHTML = renderMetadata(metadata);
  document.getElementById("active").innerHTML = renderTable(active, "No active completions.");
  document.getElementById("completed").innerHTML = renderTable(completedToShow, "No completed completions match the current filter.");
  document.getElementById("lastLine").textContent = data.last_line || "";
  const suffix = state.failures ? `, reconnecting (${state.failures})` : "";
  document.getElementById("lastEvent").textContent = data.last_event_at ? `Last event ${data.last_event_at}${suffix}` : `Waiting for logs${suffix}`;
}

async function loadState() {
  if (state.loading) return;
  state.loading = true;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.failures = 0;
    render(await response.json());
  } catch (error) {
    state.failures += 1;
    const lastEvent = document.getElementById("lastEvent");
    if (lastEvent) lastEvent.textContent = `Update failed, retrying (${state.failures})`;
  } finally {
    state.loading = false;
  }
}

loadState();
const events = new EventSource("/api/events");
events.addEventListener("snapshot", event => render(JSON.parse(event.data)));
events.addEventListener("update", () => loadState());
events.addEventListener("error", () => {
  state.failures += 1;
  setTimeout(loadState, 1000);
});
setInterval(loadState, 2000);

document.addEventListener("DOMContentLoaded", () => {
  const completedFilter = document.getElementById("completedFilter");
  const completedLimit = document.getElementById("completedLimit");
  const metadataPanel = document.querySelector("details.panel");
  if (completedFilter) completedFilter.value = state.ui.completedFilter;
  if (completedLimit) completedLimit.value = state.ui.completedLimit;
  if (metadataPanel) metadataPanel.open = Boolean(state.ui.metadataOpen);
  completedFilter?.addEventListener("change", event => {
    state.ui.completedFilter = event.target.value;
    saveUiState();
    if (state.data) render(state.data);
  });
  completedLimit?.addEventListener("change", event => {
    state.ui.completedLimit = event.target.value;
    saveUiState();
    if (state.data) render(state.data);
  });
  metadataPanel?.addEventListener("toggle", event => {
    state.ui.metadataOpen = event.target.open;
    saveUiState();
  });
  document.addEventListener("toggle", event => {
    const details = event.target.closest?.("details[data-task-key]");
    if (!details) return;
    if (details.open) {
      state.openTaskDetails.add(details.dataset.taskKey);
    } else {
      state.openTaskDetails.delete(details.dataset.taskKey);
    }
  }, true);
});
"""


def local_ip_addresses() -> list[str]:
    addresses: set[str] = set()
    hostname = socket.gethostname()
    for name in {hostname, socket.getfqdn(), "localhost"}:
        try:
            for result in socket.getaddrinfo(name, None, family=socket.AF_INET):
                address = result[4][0]
                if not address.startswith("127."):
                    addresses.add(address)
        except socket.gaierror:
            continue

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass

    return sorted(addresses)


def cpu_model_name() -> str:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
                if line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def cpu_times() -> tuple[int, int] | None:
    try:
        with Path("/proc/stat").open("r", encoding="utf-8", errors="replace") as handle:
            line = handle.readline()
    except OSError:
        return None

    if not line.startswith("cpu "):
        return None

    values = [int(value) for value in line.split()[1:]]
    if len(values) < 5:
        return None

    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


def live_gpu_status() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "temperature.gpu",
        "utilization.gpu",
        "memory.total",
        "memory.used",
        "memory.free",
        "power.draw",
        "power.limit",
    ]
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "message": f"nvidia gpu not found ({exc.__class__.__name__})"}

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "nvidia gpu not found").strip()
        return {"available": False, "message": f"nvidia gpu not found: {message}"}

    devices = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        item = dict(zip(fields, values, strict=True))
        devices.append(
            {
                "index": to_int(item["index"]),
                "name": item["name"],
                "uuid": item["uuid"],
                "driver_version": item["driver_version"],
                "temperature_c": to_int(item["temperature.gpu"]),
                "utilization_gpu_pct": to_int(item["utilization.gpu"]),
                "memory_total_mib": to_int(item["memory.total"]),
                "memory_used_mib": to_int(item["memory.used"]),
                "memory_free_mib": to_int(item["memory.free"]),
                "power_draw_w": to_float(item["power.draw"]),
                "power_limit_w": to_float(item["power.limit"]),
            }
        )

    return {
        "available": bool(devices),
        "message": "ok" if devices else "nvidia gpu not found",
        "updated_at_unix": time.time(),
        "devices": devices,
        "processes": live_gpu_processes(),
    }


def live_gpu_processes() -> list[dict[str, Any]]:
    fields = ["gpu_uuid", "pid", "process_name", "used_memory"]
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-compute-apps={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return []

    if result.returncode != 0:
        return []

    processes = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        item = dict(zip(fields, values, strict=True))
        processes.append(
            {
                "gpu_uuid": item["gpu_uuid"],
                "pid": to_int(item["pid"]),
                "process_name": item["process_name"],
                "used_memory_mib": to_int(item["used_memory"]),
            }
        )
    return processes


def to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


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
    app = TrackerApp(
        log_path,
        follow=not args.no_follow,
        stream=sys.stdin if args.log_path == "-" else None,
        tracker_host=args.host,
        tracker_port=args.port,
    )
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
