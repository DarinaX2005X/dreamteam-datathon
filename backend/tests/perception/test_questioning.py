import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from navidu.perception.questioning import (
    QuestionAnswer,
    parse_answer,
    unattended_bag_intervals,
    verify_bag_questions,
)


def test_question_support_has_no_arbitrary_five_frame_limit():
    answer = parse_answer(json.dumps({"answer": "yes", "frames": list(range(21))}), 21)
    assert answer == QuestionAnswer("yes", tuple(range(21)))


@pytest.mark.parametrize("value", [
    {"answer": "yes", "frames": [0, 21]},
    {"answer": "yes", "frames": [0, True]},
    {"answer": "yes", "frames": [0, 0]},
    {"answer": "no", "frames": [0, 1]},
    {"answer": "unknown", "frames": [0, 1]},
    {"answer": "certain", "frames": []},
    {"answer": [], "frames": []},
])
def test_invalid_or_contradictory_question_support_is_rejected(value):
    with pytest.raises(ValueError):
        parse_answer(json.dumps(value), 21)


def test_bag_after_departure_needs_joint_visible_support_and_prior_bag():
    bag = QuestionAnswer("yes", (0, 1, 2, 3, 4, 5))
    departure = QuestionAnswer("yes", (3, 4, 5))
    after = QuestionAnswer("yes", (2, 3, 4, 5))
    clear = QuestionAnswer("yes", (4, 5))
    assert unattended_bag_intervals(bag, departure, after, clear, list(range(6))) == [(4, 5)]
    for missing in range(4):
        for value in ("no", "unknown"):
            answers = [bag, departure, after, clear]
            answers[missing] = QuestionAnswer(value, ())
            assert unattended_bag_intervals(*answers, list(range(6))) == []
    assert unattended_bag_intervals(QuestionAnswer("yes", (4, 5)), departure,
                                    after, clear, list(range(6))) == []


def test_unattended_support_never_bridges_a_long_unobserved_gap():
    bag = QuestionAnswer("yes", (0, 1, 2, 6))
    after = QuestionAnswer("yes", (2, 6))
    assert unattended_bag_intervals(bag, after, after, after, list(range(7))) == []


def test_question_clock_matches_video_but_result_keeps_absolute_source_time(monkeypatch):
    """A window starting at 12 s has a 0..8 s model clock, not two timebases."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list))
    prompts, metadata = [], []
    replies = iter([
        {"answer": "yes", "frames": list(range(9))},
        *[{"answer": "yes", "frames": [4, 5]}] * 3,
    ])

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, *args, **kwargs):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            prompt = messages[0]["content"][1]["text"]
            prompts.append(prompt)
            return prompt

        def __call__(self, **kwargs):
            metadata.append(kwargs["videos_kwargs"]["video_metadata"][0])
            return Inputs(input_ids=np.zeros((1, 1), dtype=np.int64))

        def batch_decode(self, *args, **kwargs):
            return [json.dumps(next(replies))]

    model = SimpleNamespace(generate=lambda **kwargs: np.zeros((1, 2), dtype=np.int64))
    frames = [(12. + i, np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(9)]
    events, metrics = verify_bag_questions(model, Processor(), frames,
        device="cpu", dtype="unused", should_stop=lambda: False)

    assert metrics["question_calls"] == 4
    for prompt, clock in zip(prompts, metadata):
        assert "Video timestamps are relative to this window" in prompt
        assert "0=0.000s" in prompt and "8=8.000s" in prompt
        assert "0=12.000s" not in prompt
        assert clock["fps"] == 1000.
        assert clock["frames_indices"] == list(range(0, 8001, 1000))
    unattended = [event for event in events if event["kind"] == "bag_left_unattended"]
    assert len(unattended) == 1
    assert (unattended[0]["start_sec"], unattended[0]["end_sec"]) == (16., 17.)
