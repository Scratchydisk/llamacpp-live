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
                    task.touch(line, "cancelled")
                    self.state.counters["cancelled"] += 1
                    return self._event("cancelled", line, task=task.to_dict())

        return self._event("line", line)

    def _parse_task_line(self, line: str, match: re.Match[str]) -> dict[str, Any]:
        event = match.group("event")
        slot_id = int(match.group("slot"))
        task_id = int(match.group("task"))
        body = match.group("body")
        self.state.slots[str(slot_id)] = {"slot_id": slot_id, "task_id": task_id if task_id >= 0 else None, "updated_at": now_iso()}

        if task_id < 0:
            return self._event(event, line, slot_id=slot_id)

        task = self._get_task(slot_id, task_id)
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

    def _get_task(self, slot_id: int, task_id: int) -> Task:
        key = self._key(slot_id, task_id)
        task = self.state.active.get(key)
        if task is None:
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

