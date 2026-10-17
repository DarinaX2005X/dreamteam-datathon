"""Object presence and track identity do not bridge missing observations."""

import importlib.util

import pytest
from navidu.contracts.domain import Track, TrackSample
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
