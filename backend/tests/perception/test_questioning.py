import json

import pytest

from navidu.perception.questioning import (
    QuestionAnswer,
    parse_answer,
    unattended_bag_intervals,
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
