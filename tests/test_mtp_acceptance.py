from llama_tracker.parser import LlamaLogParser


def demo_records_mtp_acceptance_for_the_task() -> None:
    parser = LlamaLogParser()
    parser.parse_line("slot launch_slot_: id  3 | task 44 | processing task, is_child = 0")
    event = parser.parse_line(
        "slot print_timing: id  3 | task 44 | draft acceptance = 0.85741 "
        "( 451 accepted / 526 generated), mean len = 2.71"
    )

    assert event is not None
    task = parser.state.active["3:44"]
    assert task.draft_acceptance == 0.85741
    assert task.draft_accepted_tokens == 451
    assert task.draft_generated_tokens == 526
    assert task.draft_mean_length == 2.71


def demo_accepts_rate_wording_without_mean_length() -> None:
    parser = LlamaLogParser()
    parser.parse_line("slot launch_slot_: id  0 | task 1 | processing task, is_child = 0")
    parser.parse_line("slot print_timing: id  0 | task 1 | draft acceptance rate = 0.5 ( 1 accepted / 2 generated)")

    task = parser.state.active["0:1"]
    assert task.draft_acceptance == 0.5
    assert task.draft_mean_length is None


if __name__ == "__main__":
    demo_records_mtp_acceptance_for_the_task()
    demo_accepts_rate_wording_without_mean_length()
    print("ok")
