"""Regression checks for the always-on file monitor."""

from pathlib import Path
import tempfile
import time

from llama_tracker.server import TrackerApp


def wait_for_task(app: TrackerApp, task_id: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with app._state_lock:
            if f"0:{task_id}" in app.parser.state.active:
                return
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} was not read from the followed log")


def demo_waits_for_and_reopens_log_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "llama-server.log"
        app = TrackerApp(log_path, follow=True)
        app.start()
        try:
            log_path.write_text("slot launch_slot_: id  0 | task 1 | processing task, is_child = 0\n")
            wait_for_task(app, "1")

            log_path.rename(Path(directory) / "llama-server.log.1")
            log_path.write_text("slot launch_slot_: id  0 | task 2 | processing task, is_child = 0\n")
            wait_for_task(app, "2")
        finally:
            app.stop()


if __name__ == "__main__":
    demo_waits_for_and_reopens_log_file()
    print("ok")
