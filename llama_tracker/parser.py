from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any


TASK_PREFIX = re.compile(
    r"^slot\s+(?P<event>[\w_]+):\s+id\s+(?P<slot>\d+)\s+\|\s+task\s+(?P<task>-?\d+)\s+\|\s*(?P<body>.*)$"
)
SLOT_PREFIX = re.compile(r"^slot\s+(?P<event>[\w_]+):\s+id\s+(?P<slot>\d+)\s+\|\s*(?P<body>.*)$")
SERVER_REQUEST = re.compile(
    r"^srv\s+log_server_r:\s+done request:\s+(?P<method>\S+)\s+(?P<path>\S+)\s+(?P<client>\S+)\s+(?P<status>\d+)"
)
CACHE_STATE = re.compile(
    r"cache state:\s+(?P<prompts>\d+)\s+prompts,\s+(?P<mib>[\d.]+)\s+MiB\s+"
    r"\(limits:\s+(?P<limit_mib>[\d.]+)\s+MiB,\s+(?P<limit_tokens>\d+)\s+tokens,\s+(?P<est>\d+)\s+est\)"
)
CUDA_INIT = re.compile(r"found\s+(?P<count>\d+)\s+CUDA devices\s+\(Total VRAM:\s+(?P<vram>\d+)\s+MiB\)")
CUDA_DEVICE = re.compile(
    r"Device\s+(?P<index>\d+):\s+(?P<name>.+?),\s+compute capability\s+(?P<capability>[\d.]+),\s+"
    r"VMM:\s+(?P<vmm>\w+),\s+VRAM:\s+(?P<vram>\d+)\s+MiB"
)
N_PARALLEL = re.compile(r"using n_parallel = (?P<n_parallel>\d+) and kv_unified = (?P<kv_unified>\w+)")
BUILD_INFO = re.compile(r"^build_info:\s+(?P<build>\S+)")
SYSTEM_INFO = re.compile(
    r"n_threads = (?P<n_threads>\d+) \(n_threads_batch = (?P<n_threads_batch>\d+)\) / (?P<threads_total>\d+)"
)
HTTP_THREADS = re.compile(r"^init: using (?P<threads>\d+) threads for HTTP server")
MODEL_PATH = re.compile(r"load_model: loading model '(?P<path>[^']+)'")
MM_PROJ_PATH = re.compile(r"loaded multimodal model, '(?P<path>[^']+)'")
METADATA_SUMMARY = re.compile(
    r"loaded meta data with (?P<kv>\d+) key-value pairs and (?P<tensors>\d+) tensors from (?P<path>.+?) "
    r"\(version (?P<version>.*)\)$"
)
GGUF_KV = re.compile(r"llama_model_loader: - kv\s+\d+:\s+(?P<key>\S+)\s+(?P<type>\S+)\s+=\s+(?P<value>.*)")
TENSOR_TYPE = re.compile(r"llama_model_loader: - type\s+(?P<type>\S+):\s+(?P<count>\d+) tensors")
PRINT_INFO = re.compile(r"^print_info:\s+(?P<key>.+?)\s+=\s+(?P<value>.+)$")
CONTEXT_INFO = re.compile(r"^llama_context:\s+(?P<key>.+?)\s+=\s+(?P<value>.+)$")
BUFFER_SIZE = re.compile(r"(?P<label>CUDA0|CUDA_Host|CPU|Host|CPU_Mapped).+?(?P<mib>[\d.]+)\s+MiB")
OFFLOADED_LAYERS = re.compile(r"offloaded\s+(?P<offloaded>\d+)/(?P<total>\d+)\s+layers")
GPU_LAYERS = re.compile(r"offloading\s+(?P<count>\d+)\s+repeating layers to GPU")
PROMPT_CACHE = re.compile(r"prompt cache is enabled, size limit:\s+(?P<mib>[\d.]+)\s+MiB")
SLOTS = re.compile(r"initializing slots, n_slots = (?P<slots>\d+)")
SLOT_CONTEXT = re.compile(r"new slot, n_ctx = (?P<context>\d+)")
SERVER_LISTEN = re.compile(r"server is listening on (?P<url>\S+)")
PROJECTION = re.compile(r"projected to use (?P<use>\d+) MiB of device memory vs. (?P<free>\d+) MiB of free device memory")
FREE_MEMORY_LEFT = re.compile(r"will leave (?P<free>\d+) >= (?P<minimum>\d+) MiB of free device memory")
DEVICE_FREE = re.compile(r"using device (?P<device>\S+) \((?P<name>.+?)\).* - (?P<free>\d+) MiB free")
MEMORY_ROW = re.compile(
    r"- (?P<target>CUDA\d+ \([^)]+\)|Host)\s+\|\s+(?P<total>\d+)?\s*=?\s*(?P<rest>.*)"
)
VISION_HPARAM = re.compile(r"^load_hparams:\s+(?P<key>[^:]+):\s+(?P<value>.*)$")
CLIP_INFO = re.compile(r"^clip_model_loader:\s+(?P<key>[^:]+):\s+(?P<value>.*)$")
WARMUP_IMAGE = re.compile(r"warmup with image size = (?P<width>\d+) x (?P<height>\d+)")
NEW_PROMPT = re.compile(r"new prompt,\s+n_ctx_slot\s+=\s+(?P<context>\d+),\s+n_keep\s+=\s+(?P<keep>\d+),\s+task\.n_tokens\s+=\s+(?P<tokens>\d+)")
PROMPT_PROGRESS = re.compile(
    r"prompt processing progress,\s+n_tokens\s+=\s+(?P<n_tokens>\d+),\s+batch\.n_tokens\s+=\s+(?P<batch>\d+),\s+progress\s+=\s+(?P<progress>[\d.]+)"
)
PROMPT_DONE = re.compile(r"prompt processing done,\s+n_tokens\s+=\s+(?P<n_tokens>\d+),\s+batch\.n_tokens\s+=\s+(?P<batch>\d+)")
CHECKPOINT_CREATED = re.compile(
    r"created context checkpoint\s+(?P<index>\d+)\s+of\s+(?P<total>\d+).*n_tokens\s+=\s+(?P<n_tokens>\d+),\s+size\s+=\s+(?P<size>[\d.]+)\s+MiB"
)
CHECKPOINT_RESTORED = re.compile(r"restored context checkpoint .*n_tokens\s+=\s+(?P<n_tokens>\d+).*size\s+=\s+(?P<size>[\d.]+)\s+MiB")
SAMPLER_INIT = re.compile(r"init sampler,\s+took\s+(?P<ms>[\d.]+)\s+ms,\s+tokens:\s+text\s+=\s+(?P<text>\d+),\s+total\s+=\s+(?P<total>\d+)")
RELEASE = re.compile(r"stop processing:\s+n_tokens\s+=\s+(?P<n_tokens>\d+),\s+truncated\s+=\s+(?P<truncated>\d+)")
PROMPT_EVAL = re.compile(r"prompt eval time\s+=\s+(?P<ms>[\d.]+)\s+ms\s+/\s+(?P<tokens>\d+)\s+tokens.*\s+(?P<tps>[\d.]+)\s+tokens per second")
EVAL = re.compile(r"eval time\s+=\s+(?P<ms>[\d.]+)\s+ms\s+/\s+(?P<tokens>\d+)\s+tokens.*\s+(?P<tps>[\d.]+)\s+tokens per second")
TOTAL = re.compile(r"total time\s+=\s+(?P<ms>[\d.]+)\s+ms\s+/\s+(?P<tokens>\d+)\s+tokens")


@dataclass
class Task:
    task_id: int
    slot_id: int
    status: str = "processing"
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())
    completed_at: str | None = None
    prompt_tokens: int | None = None
    context_size: int | None = None
    current_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_progress: float = 0.0
    prompt_batch_tokens: int | None = None
    sampler_init_ms: float | None = None
    prompt_eval_ms: float | None = None
    prompt_eval_tokens: int | None = None
    prompt_eval_tps: float | None = None
    eval_ms: float | None = None
    eval_tokens: int | None = None
    eval_tps: float | None = None
    total_ms: float | None = None
    total_tokens: int | None = None
    final_tokens: int | None = None
    truncated: bool | None = None
    checkpoints_created: int = 0
    checkpoints_restored: int = 0
    checkpoint_mib: float | None = None
    request: dict[str, Any] | None = None
    last_event: str | None = None
    raw_tail: list[str] = field(default_factory=list)

    def touch(self, line: str, event: str) -> None:
        self.updated_at = now_iso()
        self.last_event = event
        self.raw_tail.append(line)
        if len(self.raw_tail) > 8:
            del self.raw_tail[:-8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServerState:
    active: dict[str, Task] = field(default_factory=dict)
    completed: list[Task] = field(default_factory=list)
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "server": {},
            "build": {},
            "hardware": {"devices": []},
            "model": {"gguf_metadata": {}, "print_info": {}, "tensor_types": {}},
            "runtime": {},
            "memory": {"initial": {}, "final": {}, "buffers": {}},
            "vision": {},
            "warnings": [],
        }
    )
    counters: dict[str, int] = field(default_factory=lambda: {"lines": 0, "tasks": 0, "completed": 0, "cancelled": 0})
    last_line: str | None = None
    last_event_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": [task.to_dict() for task in sorted(self.active.values(), key=lambda item: (item.slot_id, item.task_id))],
            "completed": [task.to_dict() for task in self.completed[-100:]][::-1],
            "slots": self.slots,
            "requests": self.requests[-100:][::-1],
            "cache": self.cache,
            "metadata": self.metadata,
            "counters": self.counters,
            "last_line": self.last_line,
            "last_event_at": self.last_event_at,
        }


class LlamaLogParser:
    def __init__(self) -> None:
        self.state = ServerState()
        self._timing_task_key: str | None = None
        self._last_task_key: str | None = None

    def parse_line(self, line: str) -> dict[str, Any] | None:
        line = line.rstrip("\n")
        if not line:
            return None

        self.state.counters["lines"] += 1
        self.state.last_line = line
        self.state.last_event_at = now_iso()

        task_match = TASK_PREFIX.match(line)
        if task_match:
            return self._parse_task_line(line, task_match)

        slot_match = SLOT_PREFIX.match(line)
        if slot_match:
            return self._parse_slot_line(line, slot_match)

        request_match = SERVER_REQUEST.match(line)
        if request_match:
            return self._parse_request(line, request_match)

        cache_match = CACHE_STATE.search(line)
        if cache_match:
            self.state.cache = {
                "prompts": int(cache_match.group("prompts")),
                "mib": float(cache_match.group("mib")),
                "limit_mib": float(cache_match.group("limit_mib")),
                "limit_tokens": int(cache_match.group("limit_tokens")),
                "estimated_tokens": int(cache_match.group("est")),
                "updated_at": now_iso(),
            }
            return self._event("cache", line, cache=self.state.cache)

        metadata = self._parse_metadata_line(line)
        if metadata:
            return metadata

        timing = self._parse_timing_line(line)
        if timing:
            return timing

        if "reasoning-budget: activated" in line:
            return self._mark_last_task(line, "reasoning", {"status": "generating"})
        if "reasoning-budget: deactivated" in line:
            return self._mark_last_task(line, "reasoning_done", {})
        if "cancel task, id_task" in line:
            match = re.search(r"id_task\s+=\s+(?P<task>\d+)", line)
            if match:
                task = self._find_task_by_task_id(int(match.group("task")))
                if task:
                    task.status = "cancelled"
                    task.completed_at = now_iso()
                    task.touch(line, "cancelled")
                    self.state.counters["cancelled"] += 1
                    self._complete_task(task)
                    return self._event("cancelled", line, task=task.to_dict())

        return self._event("line", line)

    def _parse_metadata_line(self, line: str) -> dict[str, Any] | None:
        meta = self.state.metadata
        server = meta["server"]
        build = meta["build"]
        hardware = meta["hardware"]
        model = meta["model"]
        runtime = meta["runtime"]
        memory = meta["memory"]
        vision = meta["vision"]

        if match := CUDA_INIT.search(line):
            hardware["cuda_device_count"] = int(match.group("count"))
            hardware["cuda_total_vram_mib"] = int(match.group("vram"))
            return self._event("metadata", line, section="hardware")

        if match := CUDA_DEVICE.search(line):
            device = {
                "index": int(match.group("index")),
                "name": match.group("name"),
                "compute_capability": match.group("capability"),
                "vmm": match.group("vmm") == "yes",
                "vram_mib": int(match.group("vram")),
            }
            devices = [item for item in hardware["devices"] if item.get("index") != device["index"]]
            devices.append(device)
            hardware["devices"] = sorted(devices, key=lambda item: item["index"])
            return self._event("metadata", line, section="hardware")

        if match := N_PARALLEL.search(line):
            runtime["n_parallel"] = int(match.group("n_parallel"))
            runtime["kv_unified"] = match.group("kv_unified") == "true"
            return self._event("metadata", line, section="runtime")

        if match := BUILD_INFO.match(line):
            build["build_info"] = match.group("build")
            return self._event("metadata", line, section="build")

        if line.startswith("system_info:"):
            build["system_info"] = line.removeprefix("system_info:").strip()
            if match := SYSTEM_INFO.search(line):
                build["n_threads"] = int(match.group("n_threads"))
                build["n_threads_batch"] = int(match.group("n_threads_batch"))
                build["threads_total"] = int(match.group("threads_total"))
            build["features"] = [part.strip() for part in line.split("|")[1:] if part.strip()]
            return self._event("metadata", line, section="build")

        if line == "Running without SSL":
            server["ssl"] = False
            return self._event("metadata", line, section="server")

        if match := HTTP_THREADS.match(line):
            server["http_threads"] = int(match.group("threads"))
            return self._event("metadata", line, section="server")

        if match := MODEL_PATH.search(line):
            path = match.group("path")
            model["path"] = path
            model["filename"] = path.rsplit("/", 1)[-1]
            return self._event("metadata", line, section="model")

        if match := METADATA_SUMMARY.search(line):
            model["metadata_kv_count"] = int(match.group("kv"))
            model["tensor_count"] = int(match.group("tensors"))
            model["metadata_path"] = match.group("path")
            model["gguf_version"] = match.group("version")
            return self._event("metadata", line, section="model")

        if match := GGUF_KV.search(line):
            key = match.group("key")
            value = parse_scalar(match.group("value"))
            model["gguf_metadata"][key] = value
            if key == "general.name":
                model["name"] = value
            elif key == "general.architecture":
                model["architecture"] = value
            elif key == "general.quantized_by":
                model["quantized_by"] = value
            elif key == "general.license":
                model["license"] = value
            elif key == "general.repo_url":
                model["repo_url"] = value
            elif key == "general.size_label":
                model["size_label"] = value
            return self._event("metadata", line, section="model")

        if match := TENSOR_TYPE.search(line):
            model["tensor_types"][match.group("type")] = int(match.group("count"))
            return self._event("metadata", line, section="model")

        if match := PRINT_INFO.match(line):
            key = clean_key(match.group("key"))
            value = parse_scalar(match.group("value"))
            model["print_info"][key] = value
            for public_key in (
                "file_format",
                "file_type",
                "file_size",
                "model_type",
                "model_params",
                "vocab_type",
                "n_vocab",
                "n_ctx_train",
            ):
                if key == public_key:
                    model[key] = value
            return self._event("metadata", line, section="model")

        if match := CONTEXT_INFO.match(line):
            runtime[clean_key(match.group("key"))] = parse_scalar(match.group("value"))
            return self._event("metadata", line, section="runtime")

        if "n_ctx_seq" in line and "full capacity" in line:
            runtime["context_capacity_warning"] = line
            add_warning(meta, "context", line)
            return self._event("metadata", line, section="runtime")

        if match := OFFLOADED_LAYERS.search(line):
            runtime["offloaded_layers"] = int(match.group("offloaded"))
            runtime["total_layers"] = int(match.group("total"))
            return self._event("metadata", line, section="runtime")

        if match := GPU_LAYERS.search(line):
            runtime["gpu_repeating_layers"] = int(match.group("count"))
            return self._event("metadata", line, section="runtime")

        if line.startswith("load_tensors:") and (match := BUFFER_SIZE.search(line)):
            memory["buffers"][f"{clean_key(line.split(':', 1)[1].split('=', 1)[0])}_mib"] = float(match.group("mib"))
            return self._event("metadata", line, section="memory")

        if line.startswith("sched_reserve:") and (match := BUFFER_SIZE.search(line)):
            memory["buffers"][f"{clean_key(line.split(':', 1)[1].split('=', 1)[0])}_mib"] = float(match.group("mib"))
            return self._event("metadata", line, section="memory")

        if line.startswith("llama_kv_cache:") and "buffer size" in line and (match := BUFFER_SIZE.search(line)):
            memory["buffers"]["cuda0_kv_buffer_size_mib"] = float(match.group("mib"))
            return self._event("metadata", line, section="memory")

        if line.startswith("llama_memory_recurrent:") and "buffer size" in line and (match := BUFFER_SIZE.search(line)):
            memory["buffers"]["cuda0_recurrent_state_buffer_size_mib"] = float(match.group("mib"))
            return self._event("metadata", line, section="memory")

        if match := PROJECTION.search(line):
            memory["projected_device_use_mib"] = int(match.group("use"))
            memory["free_device_memory_mib"] = int(match.group("free"))
            return self._event("metadata", line, section="memory")

        if match := FREE_MEMORY_LEFT.search(line):
            memory["projected_free_after_fit_mib"] = int(match.group("free"))
            memory["minimum_free_target_mib"] = int(match.group("minimum"))
            return self._event("metadata", line, section="memory")

        if match := DEVICE_FREE.search(line):
            hardware["active_device"] = match.group("device")
            hardware["active_device_name"] = match.group("name")
            hardware["active_device_free_mib"] = int(match.group("free"))
            return self._event("metadata", line, section="hardware")

        if "memory breakdown" in line:
            memory["_next_breakdown"] = "final" if "total   free" in line else "initial"
            return self._event("metadata", line, section="memory")

        if line.startswith("common_memory_breakdown_print: |   - ") and (match := MEMORY_ROW.search(line)):
            phase = memory.get("_next_breakdown", "initial")
            numbers = [int(item) for item in re.findall(r"\d+", match.group("rest"))]
            memory[phase][match.group("target")] = {
                "total_mib": int(match.group("total")) if match.group("total") else None,
                "numbers": numbers,
                "raw": line.split("|", 1)[1].strip(),
            }
            return self._event("metadata", line, section="memory")

        if "has vision encoder" in line:
            vision["has_vision_encoder"] = True
            return self._event("metadata", line, section="vision")

        if line.startswith("clip_ctx:"):
            vision["backend"] = line.split(":", 1)[1].strip()
            return self._event("metadata", line, section="vision")

        if match := CLIP_INFO.match(line):
            vision[clean_key(match.group("key"))] = parse_scalar(match.group("value"))
            return self._event("metadata", line, section="vision")

        if match := VISION_HPARAM.match(line):
            vision[clean_key(match.group("key"))] = parse_scalar(match.group("value"))
            return self._event("metadata", line, section="vision")

        if match := WARMUP_IMAGE.search(line):
            vision["warmup_image_width"] = int(match.group("width"))
            vision["warmup_image_height"] = int(match.group("height"))
            return self._event("metadata", line, section="vision")

        if line.startswith("warmup: flash attention"):
            runtime["vision_flash_attention"] = "enabled" in line
            return self._event("metadata", line, section="runtime")

        if match := MM_PROJ_PATH.search(line):
            path = match.group("path")
            vision["mmproj_path"] = path
            vision["mmproj_filename"] = path.rsplit("/", 1)[-1]
            model["multimodal"] = True
            return self._event("metadata", line, section="vision")

        if match := SLOTS.search(line):
            runtime["n_slots"] = int(match.group("slots"))
            return self._event("metadata", line, section="runtime")

        if match := SLOT_CONTEXT.search(line):
            runtime["slot_context"] = int(match.group("context"))
            return self._event("metadata", line, section="runtime")

        if match := PROMPT_CACHE.search(line):
            runtime["prompt_cache_enabled"] = True
            runtime["prompt_cache_limit_mib"] = float(match.group("mib"))
            return self._event("metadata", line, section="runtime")

        if "chat template, thinking =" in line:
            runtime["chat_template_thinking"] = line.rsplit("=", 1)[-1].strip() == "1"
            return self._event("metadata", line, section="runtime")

        if match := SERVER_LISTEN.search(line):
            url = match.group("url")
            server["llama_url"] = url
            if host_port := re.search(r"://(?P<host>[^:/]+):(?P<port>\d+)", url):
                server["llama_host"] = host_port.group("host")
                server["llama_port"] = int(host_port.group("port"))
            return self._event("metadata", line, section="server")

        warning_markers = (
            "HEAD failed",
            "no remote preset found",
            "consider using --no-mmap",
            "does not support partial sequence removal",
            "no implementations specified for speculative decoding",
            "require at minimum 1024 image tokens",
        )
        if any(marker in line for marker in warning_markers):
            add_warning(meta, "startup", line)
            return self._event("metadata", line, section="warnings")

        return None

    def _parse_task_line(self, line: str, match: re.Match[str]) -> dict[str, Any]:
        event = match.group("event")
        slot_id = int(match.group("slot"))
        task_id = int(match.group("task"))
        body = match.group("body")
        self.state.slots[str(slot_id)] = {"slot_id": slot_id, "task_id": task_id if task_id >= 0 else None, "updated_at": now_iso()}

        if task_id < 0:
            return self._event(event, line, slot_id=slot_id)

        task = self._get_task(slot_id, task_id)
        if task is None:
            return self._event(event, line, slot_id=slot_id, terminated=True)
        task.touch(line, event)

        if "processing task" in body:
            task.status = "processing"
        elif prompt := NEW_PROMPT.search(body):
            task.status = "prompt"
            task.context_size = int(prompt.group("context"))
            task.prompt_tokens = int(prompt.group("tokens"))
            task.prompt_progress = 0.0
        elif progress := PROMPT_PROGRESS.search(body):
            task.status = "prompt"
            task.current_tokens = int(progress.group("n_tokens"))
            task.prompt_batch_tokens = int(progress.group("batch"))
            task.prompt_progress = float(progress.group("progress"))
        elif done := PROMPT_DONE.search(body):
            task.status = "generating"
            task.current_tokens = int(done.group("n_tokens"))
            task.prompt_batch_tokens = int(done.group("batch"))
            task.prompt_progress = 1.0
        elif sampler := SAMPLER_INIT.search(body):
            task.sampler_init_ms = float(sampler.group("ms"))
            task.prompt_tokens = int(sampler.group("total"))
            task.status = "generating"
        elif checkpoint := CHECKPOINT_CREATED.search(body):
            task.checkpoints_created += 1
            task.checkpoint_mib = float(checkpoint.group("size"))
        elif checkpoint := CHECKPOINT_RESTORED.search(body):
            task.checkpoints_restored += 1
            task.checkpoint_mib = float(checkpoint.group("size"))
        elif release := RELEASE.search(body):
            task.final_tokens = int(release.group("n_tokens"))
            task.truncated = release.group("truncated") == "1"
            task.status = "completed" if task.status != "cancelled" else task.status
            task.completed_at = now_iso()
            if task.prompt_tokens is not None:
                task.generated_tokens = max(0, task.final_tokens - task.prompt_tokens)
            self._complete_task(task)

        return self._event(event, line, task=task.to_dict())

    def _parse_slot_line(self, line: str, match: re.Match[str]) -> dict[str, Any]:
        event = match.group("event")
        slot_id = int(match.group("slot"))
        body = match.group("body")
        self.state.slots[str(slot_id)] = {"slot_id": slot_id, "task_id": None, "updated_at": now_iso(), "last_event": event}
        return self._event(event, line, slot_id=slot_id, body=body)

    def _parse_request(self, line: str, match: re.Match[str]) -> dict[str, Any]:
        request = {
            "method": match.group("method"),
            "path": match.group("path"),
            "client": match.group("client"),
            "status": int(match.group("status")),
            "at": now_iso(),
        }
        self.state.requests.append(request)
        if len(self.state.requests) > 200:
            del self.state.requests[:-200]

        task = self._last_active_task()
        if task:
            task.request = request
            task.touch(line, "request_done")

        return self._event("request", line, request=request, task=task.to_dict() if task else None)

    def _parse_timing_line(self, line: str) -> dict[str, Any] | None:
        task = self._last_active_task() or self._task_by_key(self._timing_task_key)
        if not task:
            return None

        if prompt := PROMPT_EVAL.search(line):
            task.prompt_eval_ms = float(prompt.group("ms"))
            task.prompt_eval_tokens = int(prompt.group("tokens"))
            task.prompt_eval_tps = float(prompt.group("tps"))
        elif eval_match := EVAL.search(line):
            task.eval_ms = float(eval_match.group("ms"))
            task.eval_tokens = int(eval_match.group("tokens"))
            task.eval_tps = float(eval_match.group("tps"))
        elif total := TOTAL.search(line):
            task.total_ms = float(total.group("ms"))
            task.total_tokens = int(total.group("tokens"))
        else:
            return None

        task.touch(line, "timing")
        return self._event("timing", line, task=task.to_dict())

    def _get_task(self, slot_id: int, task_id: int) -> Task | None:
        key = self._key(slot_id, task_id)
        task = self.state.active.get(key)
        if task is None:
            for completed in self.state.completed:
                if completed.slot_id == slot_id and completed.task_id == task_id:
                    return None
            task = Task(slot_id=slot_id, task_id=task_id)
            self.state.active[key] = task
            self.state.counters["tasks"] += 1
        self._last_task_key = key
        self._timing_task_key = key
        return task

    def _complete_task(self, task: Task) -> None:
        key = self._key(task.slot_id, task.task_id)
        self.state.active.pop(key, None)
        self.state.completed.append(task)
        if len(self.state.completed) > 500:
            del self.state.completed[:-500]
        self.state.counters["completed"] += 1

    def _find_task_by_task_id(self, task_id: int) -> Task | None:
        for task in self.state.active.values():
            if task.task_id == task_id:
                return task
        return None

    def _last_active_task(self) -> Task | None:
        return self._task_by_key(self._last_task_key) or self._task_by_key(self._timing_task_key)

    def _task_by_key(self, key: str | None) -> Task | None:
        return self.state.active.get(key) if key else None

    def _mark_last_task(self, line: str, event: str, attrs: dict[str, Any]) -> dict[str, Any]:
        task = self._last_active_task()
        if task:
            for key, value in attrs.items():
                setattr(task, key, value)
            task.touch(line, event)
            return self._event(event, line, task=task.to_dict())
        return self._event(event, line)

    def _event(self, kind: str, line: str, **payload: Any) -> dict[str, Any]:
        return {"kind": kind, "at": now_iso(), "line": line, **payload}

    @staticmethod
    def _key(slot_id: int, task_id: int) -> str:
        return f"{slot_id}:{task_id}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"enabled", "disabled"}:
        return value
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+(e[+-]?\d+)?", value, re.IGNORECASE):
        return float(value)
    return value


def add_warning(metadata: dict[str, Any], category: str, message: str) -> None:
    warning = {"category": category, "message": message}
    if warning not in metadata["warnings"]:
        metadata["warnings"].append(warning)
