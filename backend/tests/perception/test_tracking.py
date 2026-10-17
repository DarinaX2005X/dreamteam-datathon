"""Object presence and track identity do not bridge missing observations."""

import importlib.util

import pytest
from navidu.contracts.domain import Observation, Track, TrackSample
from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.runtime import LocalPerceptionRuntime, _object_observations


def test_presence_is_split_at_source_clock_gaps():
    samples = [TrackSample(t=t, bbox=(0.1, 0.1, 0.2, 0.2), score=0.9)
               for t in (0, 0.333, 4, 4.333)]
    track = Track(track_id="o1", kind="object", object_class="phone",
                  first_sec=0, last_sec=4.333, samples=samples)
    observations = _object_observations([track], "test-model", start_index=1)
    assert [(item.start_sec, item.end_sec) for item in observations] == [(0, 0.333), (4, 4.333)]


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None or importlib.util.find_spec("supervision") is None,
                    reason="The optional inference environment is required")
def test_class_tracker_ages_through_empty_frames(monkeypatch):
    import numpy as np

    class Detector:
        labels = {"77": "cell phone"}
        model_id = "test-model"
        device = "cpu"
        index = 0

        def predict(self, rgb):
            visible = self.index < 3 or self.index >= 14
            self.index += 1
            return (np.array([[.1, .1, .2, .2]]) if visible else np.empty((0, 4)),
                    np.array([77], dtype=int) if visible else np.array([], dtype=int),
                    np.array([.9]) if visible else np.array([]))

    runtime = LocalPerceptionRuntime()
    runtime._detector = Detector()
    monkeypatch.setattr(runtime, "prepare", lambda config: None)
    frames = [(index / 3, np.zeros((100, 100, 3), dtype=np.uint8)) for index in range(18)]
    result = runtime.perceive(frames, config=PipelineConfig(check_id="0123456789ab", analysis_fps=3),
                             emit=lambda event: None, should_stop=lambda: False)
    assert len(result.tracks) == 2
    assert result.tracks[0].last_sec < result.tracks[1].first_sec
    assert any(item.track_id is not None for item in result.raw_detections)
    assert all(item.end_sec - item.start_sec < 2 for item in result.observations)


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None or importlib.util.find_spec("supervision") is None,
                    reason="The optional inference environment is required")
def test_late_person_is_confirmed_on_two_detections_before_moderate_motion(monkeypatch):
    import numpy as np

    class Detector:
        labels = {"1": "person"}
        model_id = "test-model"
        device = "cpu"
        index = 0

        def predict(self, rgb):
            boxes, scores = [[.1, .1, .3, .8]], [.9]
            if self.index >= 3:
                # A synthetic second person appears after the initial frame.
                # Three-detection confirmation loses this moderate-confidence
                # track at the third box; two-detection confirmation retains it.
                offset = self.index - 3
                left = [.600, .606, .660, .676][offset]
                boxes.append([left, .2, left + .2, .8])
                scores.append([.488, .516, .505, .520][offset])
            self.index += 1
            return np.asarray(boxes), np.ones(len(boxes), dtype=int), np.asarray(scores)

    runtime = LocalPerceptionRuntime()
    runtime._detector = Detector()
    monkeypatch.setattr(runtime, "prepare", lambda config: None)
    frames = [(index / 3, np.zeros((100, 100, 3), dtype=np.uint8)) for index in range(7)]
    result = runtime.perceive(frames, config=PipelineConfig(check_id="0123456789ab", analysis_fps=3),
                             emit=lambda event: None, should_stop=lambda: False)
    people = [track for track in result.tracks if track.kind == "person"]
    assert len(people) == 2
    late = next(track for track in people if track.samples[0].bbox[0] > .5)
    assert len(late.samples) == 3
    assert late.first_sec == pytest.approx(4 / 3, abs=.001)
    late_raw = [item for item in result.raw_detections if item.bbox[0] > .5]
    assert late_raw[0].track_id is None
    assert all(item.track_id == late.track_id for item in late_raw[1:])


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None or importlib.util.find_spec("supervision") is None,
                    reason="The optional inference environment is required")
@pytest.mark.parametrize("frame_count, expected_windows", [(4, 1), (5, 2), (6, 2)])
def test_semantic_tail_requires_new_frames_and_emits_real_partial_results(monkeypatch, frame_count, expected_windows):
    import numpy as np

    class Detector:
        labels = {}
        model_id = "test-detector"
        device = "cpu"

        def predict(self, rgb):
            return np.empty((0, 4)), np.array([], dtype=int), np.array([])

    class Semantic:
        sample_fps = 1
        window_frames = 4

        def __init__(self):
            self.windows = []
            self.last_metrics = {}

        def observe(self, frames, tracks, artifacts, *, should_stop):
            self.windows.append([timestamp for timestamp, _ in frames])
            artifacts.observations.append(Observation(obs_id=f"ob-{len(self.windows)}",
                kind="phone_visible", start_sec=frames[0][0], end_sec=frames[-1][0],
                provenance="temporal_model", model_id="test-semantic", score=None))

    runtime = LocalPerceptionRuntime()
    runtime._detector, runtime._semantic = Detector(), Semantic()
    monkeypatch.setattr(runtime, "prepare", lambda config: None)
    events = []
    frames = [(float(index), np.zeros((32, 32, 3), dtype=np.uint8)) for index in range(frame_count)]
    result = runtime.perceive(frames, config=PipelineConfig(check_id="0123456789ab", analysis_fps=1),
                             emit=lambda event: events.append(event.model_copy(deep=True)), should_stop=lambda: False)
    assert len(runtime._semantic.windows) == expected_windows
    assert runtime._semantic.windows[-1][-1] == frame_count - 1
    assert len(result.observations) == expected_windows
    partial = [event for event in events[:-1] if event.observations_delta]
    assert len(partial) == expected_windows
    assert partial[0].processed_until_sec == 3
    assert partial[0].observations_delta[0].obs_id == "ob-1"
