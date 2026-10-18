import numpy as np
import pytest
from navidu.contracts.domain import Track, TrackSample
from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.grounding import bind_grounded_events, ground_objects
from navidu.perception.object_detector import (
    CombinedDetector,
    identity_views,
    object_views,
    suppress_duplicates,
)
from navidu.perception.qwen import spatial_bindings, spatial_context
from navidu.perception.runtime import LocalPerceptionRuntime
from navidu.pipeline.interfaces import PerceptionArtifacts


def test_tiled_duplicates_collapse_but_distinct_hands_and_stacks_survive():
    detections = [
        {"kind": "cash", "bbox_xyxy": [1, 1, 10, 10], "score": .9},
        {"kind": "cash", "bbox_xyxy": [2, 2, 8, 8], "score": .8},
        {"kind": "cash", "bbox_xyxy": [12, 1, 21, 10], "score": .7},
        {"kind": "hand", "bbox_xyxy": [1, 1, 10, 10], "score": .7}]
    assert len(suppress_duplicates(detections)) == 3
    assert object_views(464, 848)[0] == (0, 0, 464, 848)


def test_spatial_crop_uses_only_actual_current_window_boxes_and_preserves_ambiguity():
    track = Track(track_id="o3", kind="object", object_class="cash", first_sec=0, last_sec=99,
        samples=[TrackSample(t=t, bbox=box, score=.8) for t, box in
                 [(0, (.1, .4, .2, .45)), (1, (.11, .4, .21, .45)), (99, (.7, .8, .9, .95))]])
    crop, hints = spatial_context([track], [0., 1.])
    assert crop == pytest.approx((.05, .36, .26, .49))
    assert '"frame":99' not in hints and '"box":[700' not in hints
    assert "not verified facts" in hints and "does not by itself prove" in hints
    assert spatial_context([track], [30., 31.]) == (None, "")
    assert spatial_context([track], [0.])[0] is None


def test_identity_search_uses_observed_desk_objects_and_bounds_its_work():
    objects = [{"kind": kind, "bbox_xyxy": box, "score": .8} for kind, box in [
        ("money_counter", [70, 350, 140, 430]), ("hand", [170, 390, 190, 410]),
        ("hand", [430, 10, 450, 30])]]
    selected = identity_views(464, 848, objects)
    assert 1 <= len(selected) <= 4
    assert all(view in object_views(464, 848) for view in selected)
    assert any(x1 <= 180 <= x2 and y1 <= 430 <= y2 for x1, y1, x2, y2 in selected)
    assert identity_views(464, 848, []) == object_views(464, 848)


def test_distant_hand_does_not_remove_the_money_detail_but_remains_a_candidate():
    cash = Track(track_id="o1", kind="object", object_class="cash", first_sec=0, last_sec=1,
        samples=[TrackSample(t=t, bbox=(.1, .4, .2, .45), score=.8) for t in (0, 1)])
    hand = Track(track_id="o2", kind="object", object_class="hand", first_sec=0, last_sec=1,
        samples=[TrackSample(t=t, bbox=(.8, .01, .9, .05), score=.8) for t in (0, 1)])
    crop, hints = spatial_context([cash, hand], [0., 1.])
    assert crop == pytest.approx((.05, .36, .25, .49))
    assert '"kind":"hand"' in hints and '"box":[800,10,900,50]' in hints


def test_fragmented_hands_cannot_hide_the_counter_from_cash_context():
    def track(index, kind, box):
        return Track(track_id=f"o{index}", kind="object", object_class=kind, first_sec=0, last_sec=1,
            samples=[TrackSample(t=t, bbox=box, score=.8) for t in (0, 1)])

    tracks = [track(index, "hand", (.1, .4, .2, .45)) for index in range(20)]
    tracks.extend([track(20, "cash", (.1, .4, .2, .45)),
                   track(21, "money_counter", (.5, .3, .65, .5))])
    crop, hints = spatial_context(tracks, [0., 1.])
    assert crop == pytest.approx((.05, .26, .7, .54))
    assert hints.count('"kind":"hand"') == 4
    assert '"kind":"money_counter"' in hints


def test_combined_detectors_keep_people_and_objects_in_separate_namespaces(monkeypatch):
    class General:
        labels, model_id, device = {"1": "person"}, "general", "cpu"

        def predict(self, rgb):
            return np.array([[.1, .1, .3, .8]]), np.array([1]), np.array([.9])

    class Objects:
        labels, model_id, device = {"1000": "hand", "1001": "cash"}, "reviewed", "cuda"

        def predict(self, rgb):
            return np.array([[.2, .4, .3, .5], [.24, .45, .28, .49]]), np.array([1000, 1001]), np.array([.12, .11])

    detector = CombinedDetector(General(), Objects())
    runtime = LocalPerceptionRuntime()
    runtime._detector = detector
    monkeypatch.setattr(runtime, "prepare", lambda config: None)
    result = runtime.perceive([(i / 3, np.zeros((32, 32, 3), dtype=np.uint8)) for i in range(3)],
        config=PipelineConfig(check_id="0123456789ab", analysis_fps=3), emit=lambda event: None,
        should_stop=lambda: False)
    assert len([track for track in result.tracks if track.kind == "person"]) == 1
    assert {track.object_class for track in result.tracks if track.kind == "object"} == {"cash", "hand"}
    assert {item.model_id for item in result.raw_detections if item.class_name == "cash"} == {"reviewed"}
    cash = next(track for track in result.tracks if track.object_class == "cash")
    assert all(sample.score == pytest.approx(.11) for sample in cash.samples)
    assert all(item.score == pytest.approx(.11) for item in result.raw_detections if item.class_name == "cash")
    assert not spatial_bindings([cash], [sample.t for sample in cash.samples])
    assert spatial_bindings([cash], [sample.t for sample in cash.samples], {"cash": .1})
    # Geometric overlap never becomes a cash handover or another semantic action.
    assert not result.observations


def test_detector_boxes_can_ground_verified_events_without_model_relocalization():
    track = Track(track_id="o9", kind="object", object_class="cash", first_sec=0, last_sec=1,
        samples=[TrackSample(t=t, bbox=(.1, .4, .2, .45), score=.8) for t in (0, 1)])
    bindings = spatial_bindings([track], [0., 1.])
    payload = {"objects": [], "events": [{"objects": ["c1"]}]}
    artifacts = PerceptionArtifacts()
    objects = ground_objects(payload, [0., 1.], artifacts, "m", "v", detected_objects=bindings)
    assert objects["c1"][0] is track and not artifacts.observations
    event = {"_source_index": 0, "start_sec": 0., "end_sec": 1., "subjects": ["p1"], "kind": "cash_in_hands"}
    bind_grounded_events(payload, [event], objects)
    assert event["relation_id"] == "cash:o9" and event["subjects"] == ["p1", "o9"]
    relabeled = {"objects": [{"id": "c4", "kind": "cash", "samples": [
        {"frame": index, "box": [100, 400, 200, 450]} for index in (0, 1)]}]}
    remapped = ground_objects(relabeled, [0., 1.], artifacts, "m", "v", detected_objects=bindings)
    assert remapped["c4"][0] is track and not artifacts.tracks
    assert all(sample.score == .8 for sample in track.samples)
    relabeled["events"] = [{"objects": ["c1", "c4"]}]
    alias_event = {"_source_index": 0, "start_sec": 0., "end_sec": 1., "subjects": ["p1"], "kind": "cash_in_hands"}
    bind_grounded_events(relabeled, [alias_event], remapped)
    assert alias_event["relation_id"] == "cash:o9" and alias_event["subjects"] == ["p1", "o9"]
    # A contradictory model box invalidates the proposed object identity.
    payload["objects"] = [{"id": "c1", "kind": "cash", "samples": [
        {"frame": index, "box": [700, 700, 800, 800]} for index in (0, 1)]}]
    assert not ground_objects(payload, [0., 1.], artifacts, "m", "v", detected_objects=bindings)
