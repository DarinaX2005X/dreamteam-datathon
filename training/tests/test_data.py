"""Point supervision never supplies negative labels for unannotated windows."""

from dataclasses import replace
from pathlib import Path

import pytest

from eval.training import TrainingSample
from training.data import make_inference_windows, make_training_windows, supervised_target


def _sample(positive=True, *, time_sec=17.0):
    return TrainingSample("sample", "a" * 64, Path("private-name.mov"), 30.0,
                          positive, time_sec if positive else None, positive)


def test_positive_training_crops_contain_the_known_point_and_are_repeatable():
    sample = _sample()
    windows = make_training_windows([sample])
    assert windows == make_training_windows([sample])
    assert len(windows) == 6
    assert all(window.start_sec <= 17 <= window.end_sec for window in windows)
    assert len({window.start_sec for window in windows}) == 6


def test_unknown_positive_windows_are_never_negative_training_targets():
    windows = make_inference_windows([_sample()])
    targets = [supervised_target(window, [(window.start_sec + index / 2, None) for index in range(16)])
               for window in windows]
    assert any(target is None for target in targets)
    assert any(target is not None for target in targets)
    assert all(target is None or target["candidate"] == 1 for target in targets)


def test_clean_sources_supply_only_clean_targets():
    for window in make_training_windows([_sample(False)]):
        frames = [(window.start_sec, None), (window.end_sec, None)]
        assert supervised_target(window, frames) == {"candidate": 0, "frame": None}


def test_validation_windows_are_independent_of_targets_and_cover_the_source():
    positive = _sample()
    negative = replace(positive, has_violation=False, time_sec=None, time_supervision=False)
    positive_windows = make_inference_windows([positive])
    negative_windows = make_inference_windows([negative])
    assert [(window.start_sec, window.end_sec) for window in positive_windows] == \
        [(window.start_sec, window.end_sec) for window in negative_windows]
    assert positive_windows[0].start_sec == 0
    assert positive_windows[-1].end_sec == positive.duration_sec
    assert all(left.end_sec >= right.start_sec for left, right in zip(positive_windows, positive_windows[1:]))


def test_ambiguous_positive_time_cannot_be_used_for_point_crop_training():
    with pytest.raises(ValueError, match="reliable organizer timestamp"):
        make_training_windows([replace(_sample(), time_sec=None, time_supervision=False)])


def test_duplicate_source_identity_cannot_inflate_the_training_set():
    with pytest.raises(ValueError, match="Repeated source hashes"):
        make_training_windows([_sample(), _sample()])
