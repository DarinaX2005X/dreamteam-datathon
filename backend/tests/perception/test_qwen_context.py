import pytest
from navidu.perception.grounding import ground_objects, visible_people
from navidu.perception.qwen import detail_bounds
from navidu.perception.structured_output import build_schema
from navidu.pipeline.interfaces import PerceptionArtifacts


def test_crop_is_derived_from_persistent_observed_geometry_not_person_number():
    box = (.4, .1, .6, .8)
    frames = [{"p7": box}] * 4 + [{"p1": (.0, .0, .1, .1), "p7": box}]
    result = detail_bounds(frames)
    assert result is not None and result[0] < box[0] and result[2] > box[2]
    assert result[1] > box[1] and result[3] >= box[3]
    assert all(0 <= value <= 1 for value in result)
    assert detail_bounds([{}, {}]) is None
    assert detail_bounds([{"p1": box}, {}, {}]) is None


def test_crop_object_coordinates_map_back_to_full_source_frame():
    payload = {"_detail_bounds": (.2, .3, .8, .9), "objects": [{"id": "c1", "kind": "cash",
        "samples": [{"frame": index, "box": [100, 200, 300, 400], "view": "detail"} for index in (0, 1)]}]}
    artifacts = PerceptionArtifacts()
    result = ground_objects(payload, [0., 1.], artifacts, "m", "v7")
    assert result["c1"][0].samples[0].bbox == pytest.approx((.26, .42, .38, .54))
    del payload["_detail_bounds"]
    assert ground_objects(payload, [0., 1.], PerceptionArtifacts(), "m", "v7") == {}


def test_visibility_retains_only_actual_supported_frames_and_never_fills_detector_gap():
    payload = {"visible_people": [{"track": "p1", "clear": True, "physical": True, "frames": [0, 1, 2]}]}
    artifacts = PerceptionArtifacts()
    visible_people(payload, [{}, {"p1": ()}, {"p1": ()}], [0., 1., 2.], artifacts, "m", "v7")
    assert [signal.start_sec for signal in artifacts.semantic_signals] == [1., 2.]
    artifacts = PerceptionArtifacts()
    visible_people(payload, [{}, {}, {"p1": ()}], [0., 1., 2.], artifacts, "m", "v7")
    assert artifacts.semantic_signals == []


def test_context_schema_has_explicit_bounded_objects_and_visibility():
    schema = build_schema(64, ["p1", "p2"], "v7-context")
    assert {"objects", "visible_people"}.issubset(schema["required"])
    assert schema["properties"]["events"]["maxItems"] == 10
    sample = schema["properties"]["objects"]["items"]["properties"]["samples"]["items"]
    assert sample["properties"]["view"]["enum"] == ["full", "detail"]


def test_compact_visible_intervals_still_require_two_actual_frames():
    payload = {"visible_people": [{"track": "p1", "clear": True, "physical": True,
                                   "start_frame": 0, "end_frame": 2}]}
    artifacts = PerceptionArtifacts()
    visible_people(payload, [{}, {"p1": ()}, {"p1": ()}], [0., 1., 2.], artifacts, "m", "v8")
    assert [signal.start_sec for signal in artifacts.semantic_signals] == [1., 2.]
    payload["visible_people"][0]["end_frame"] = 99
    invalid = PerceptionArtifacts()
    visible_people(payload, [{}, {}, {}], [0., 1., 2.], invalid, "m", "v8")
    assert invalid.semantic_signals == []
