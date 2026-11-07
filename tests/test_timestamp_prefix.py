from llama_tracker.parser import LlamaLogParser

PLAIN = "slot launch_slot_: id  0 | task 4430 | processing task, is_child = 0"
PREFIXED = "352.53.225.930 I slot launch_slot_: id  0 | task 4430 | processing task, is_child = 0"


def demo() -> None:
    for line in (PLAIN, PREFIXED):
        parser = LlamaLogParser()
        event = parser.parse_line(line)
        assert event is not None and event["kind"] == "launch_slot_", (line, event)
        task = parser.state.active.get("0:4430")
        assert task is not None, f"no task tracked for: {line!r}"


def demo_live_metrics_while_task_prefixed() -> None:
    parser = LlamaLogParser()
    parser.parse_line("8.40.674.366 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0")
    parser.parse_line(
        "8.44.220.334 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   5832, "
        "progress = 1.00, t =   3.55 s / 1644.70 tokens per second"
    )
    task = parser.state.active["0:0"]
    assert task.current_tokens == 5832, task.current_tokens
    assert task.prompt_tokens == 5832, task.prompt_tokens
    assert task.prompt_eval_tps == 1644.70, task.prompt_eval_tps

    parser.parse_line(
        "8.47.583.592 I slot print_timing: id  0 | task 0 | n_decoded =    198, tg =  65.77 t/s, tg_3s =  65.77 t/s"
    )
    assert task.status == "generating", task.status
    assert task.generated_tokens == 198, task.generated_tokens
    assert task.eval_tps == 65.77, task.eval_tps

    parser.parse_line(
        "9.17.002.174 I slot print_timing: id  0 | task 0 | prompt eval time =    3898.64 ms /  5858 tokens "
        "(    0.67 ms per token,  1502.58 tokens per second)"
    )
    assert task.prompt_tokens == 5858, task.prompt_tokens
    assert task.prompt_eval_tps == 1502.58, task.prompt_eval_tps


def demo_live_generation_without_tg_3s() -> None:
    parser = LlamaLogParser()
    parser.parse_line("5.00.000.000 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0")
    parser.parse_line("5.06.493.843 I slot print_timing: id  0 | task 0 | n_decoded =    100, tg =  18.06 t/s")
    task = parser.state.active["0:0"]
    assert task.status == "generating", task.status
    assert task.generated_tokens == 100, task.generated_tokens
    assert task.eval_tps == 18.06, task.eval_tps


def demo_cached_token_updates_drive_prompt_progress() -> None:
    parser = LlamaLogParser()
    parser.parse_line("slot launch_slot_: id  0 | task 8 | processing task, is_child = 0")
    parser.parse_line("slot update_slots: id  0 | task 8 | new prompt, n_ctx_slot = 192000, n_keep = 0, task.n_tokens = 695")
    parser.parse_line("slot update_slots: id  0 | task 8 | cached n_tokens = 673, memory_seq_rm [673, end)")
    task = parser.state.active["0:8"]
    assert task.current_tokens == 673, task.current_tokens
    assert round(task.prompt_progress * 100) == 97, task.prompt_progress

    parser.parse_line("slot init_sampler: id  0 | task 8 | init sampler, took 0.08 ms, tokens: text = 695, total = 695")
    assert task.prompt_progress == 1.0, task.prompt_progress


if __name__ == "__main__":
    demo()
    demo_live_metrics_while_task_prefixed()
    demo_live_generation_without_tg_3s()
    demo_cached_token_updates_drive_prompt_progress()
    print("ok")
