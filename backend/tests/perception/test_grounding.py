from navidu.contracts.domain import Track, TrackSample
from navidu.perception.grounding import (
    bind_grounded_events,
    derive_simultaneous_people,
    ground_objects,
    visible_people,
)
from navidu.perception.reflections import apply_reflection_filters
from navidu.pipeline.interfaces import PerceptionArtifacts, SemanticSignal


def people_artifacts(ids=("p1", "p2", "p3"), times=(0., 1., 2.)):
    return PerceptionArtifacts(tracks=[Track(track_id=person, kind="person", first_sec=min(times),
        last_sec=max(times), samples=[TrackSample(t=t, bbox=(.1, .1, .3, .8), score=.9)
                                     for t in times]) for person in ids])


def claims(ids, frames=(0, 1, 2)):
    return {"visible_people": [{"track": person, "clear": True, "physical": True,
                                "frames": list(frames)} for person in ids]}


def test_count_uses_simultaneous_clear_physical_people_and_excludes_unclear():
    artifacts = people_artifacts()
    payload = claims(("p1", "p2", "p3"))
    payload["visible_people"][2]["clear"] = False
    visible_people(payload, [dict.fromkeys(("p1", "p2", "p3"))] * 3, [0., 1., 2.], artifacts, "m", "v7")
    derive_simultaneous_people(artifacts)
    assert not artifacts.observations
    payload["visible_people"][2]["clear"] = True
    visible_people(payload, [dict.fromkeys(("p1", "p2", "p3"))] * 3, [0., 1., 2.], artifacts, "m", "v7")
    derive_simultaneous_people(artifacts)
    assert len(artifacts.observations) == 1
    assert artifacts.observations[0].subjects == ["p1", "p2", "p3"]
    assert artifacts.observations[0].start_sec == 0 and artifacts.observations[0].end_sec == 2


def test_a_new_client_is_not_an_extra_person_and_detector_boxes_are_insufficient():
    artifacts = people_artifacts(times=(0., 1., 2., 3.))
    frame_tracks = [{"p1": (), "p2": ()}] * 2 + [{"p1": (), "p3": ()}] * 2
    payload = claims(("p1",), (0, 1, 2, 3))
    payload["visible_people"] += claims(("p2",), (0, 1))["visible_people"]
    payload["visible_people"] += claims(("p3",), (2, 3))["visible_people"]
    visible_people(payload, frame_tracks, [0., 1., 2., 3.], artifacts, "m", "v7")
    derive_simultaneous_people(artifacts)
    assert not artifacts.observations
    assert artifacts.diagnostics["simultaneous_people"]["maximum_clear_people"] == 2
    untouched = people_artifacts()
    derive_simultaneous_people(untouched)
    assert not untouched.observations


def test_reflection_is_removed_before_counting_even_with_conflicting_clear_claim():
    artifacts = people_artifacts()
    visible_people(claims(("p1", "p2", "p3")), [dict.fromkeys(("p1", "p2", "p3"))] * 3,
                   [0., 1., 2.], artifacts, "m", "v7")
    artifacts.semantic_signals.append(SemanticSignal(kind="reflection", start_sec=0, end_sec=2,
                                                     subjects=("p3", "p1")))
    apply_reflection_filters(artifacts)
    derive_simultaneous_people(artifacts)
    assert not artifacts.observations


def test_unverified_reflection_claim_blocks_clear_count_across_overlapping_windows():
    artifacts = people_artifacts()
    tracks = [dict.fromkeys(("p1", "p2", "p3"))] * 3
    visible_people(claims(("p1", "p2", "p3")), tracks, [0., 1., 2.], artifacts, "m", "v8")
    payload = {"reflections": [{"reflected_track": "p3", "source_track": "p99", "frames": [0, 1, 2]}]}
    visible_people(payload, tracks, [0., 1., 2.], artifacts, "m", "v8")
    apply_reflection_filters(artifacts)
    derive_simultaneous_people(artifacts)
    assert not artifacts.observations
    assert len(artifacts.tracks) == 3 and all(len(track.samples) == 3 for track in artifacts.tracks)
    assert artifacts.diagnostics["simultaneous_people"]["maximum_clear_people"] == 2


def test_reflection_uncertainty_is_limited_to_claimed_frames():
    artifacts = people_artifacts()
    payload = claims(("p1", "p2", "p3"))
    payload["reflections"] = [{"reflected_track": "p3", "source_track": "p99", "frames": [0]}]
    visible_people(payload, [dict.fromkeys(("p1", "p2", "p3"))] * 3,
                   [0., 1., 2.], artifacts, "m", "v8")
    derive_simultaneous_people(artifacts)
    assert len(artifacts.observations) == 1
    assert artifacts.observations[0].start_sec == 1 and artifacts.observations[0].end_sec == 2


def test_reflection_uncertainty_does_not_transfer_to_physical_source():
    ids = ("p1", "p2", "p3", "p4")
    artifacts = people_artifacts(ids=ids)
    payload = claims(ids)
    payload["reflections"] = [{"reflected_track": "p3", "source_track": "p1", "frames": [0, 1, 2]}]
    visible_people(payload, [dict.fromkeys(ids)] * 3, [0., 1., 2.], artifacts, "m", "v8")
    artifacts.semantic_signals.append(SemanticSignal(kind="reflection", start_sec=0, end_sec=2,
                                                     subjects=("p3", "p1")))
    apply_reflection_filters(artifacts)
    derive_simultaneous_people(artifacts)
    assert len(artifacts.observations) == 1
    assert artifacts.observations[0].subjects == ["p1", "p2", "p4"]


def object_payload(local="c1", frames=(0, 1), box=(100, 200, 200, 300)):
    return {"objects": [{"id": local, "kind": "cash", "samples": [
        {"frame": index, "box": list(box)} for index in frames]}]}


def test_cash_memory_requires_two_shared_frames_not_reused_desk_position():
    artifacts = PerceptionArtifacts()
    first = ground_objects(object_payload(), [0., 1.], artifacts, "m", "v7")
    same = ground_objects(object_payload("c2"), [0., 1.], artifacts, "m", "v7")
    next_client = ground_objects(object_payload("c3"), [4., 5.], artifacts, "m", "v7")
    one_common = ground_objects(object_payload("c4"), [5., 6.], artifacts, "m", "v7")
    assert first["c1"][0].track_id == same["c2"][0].track_id
    assert next_client["c3"][0].track_id != same["c2"][0].track_id
    assert one_common["c4"][0].track_id != next_client["c3"][0].track_id
    assert all(sample.score is None for track in artifacts.tracks for sample in track.samples)


def test_grounding_rejects_invented_frames_large_boxes_and_one_frame_objects():
    for payload in (object_payload(frames=(0, 99)), object_payload(frames=(0, 0)),
                    object_payload(box=(0, 0, 1000, 1000)), object_payload(box=(100, 200, 50, 300))):
        artifacts = PerceptionArtifacts()
        assert not ground_objects(payload, [0., 1.], artifacts, "m", "v7")
        assert not artifacts.tracks


def test_two_batches_in_one_window_never_merge_from_overlapping_boxes():
    artifacts = PerceptionArtifacts()
    payload = object_payload()
    payload["objects"] += object_payload("c2")["objects"]
    result = ground_objects(payload, [0., 1.], artifacts, "m", "v7")
    assert result["c1"][0].track_id != result["c2"][0].track_id


def test_sparse_object_box_does_not_draw_on_a_distant_frame():
    from navidu.evidence.frames import _nearest_box

    artifacts = PerceptionArtifacts()
    track = ground_objects(object_payload(), [1., 8.], artifacts, "m", "v7")["c1"][0]
    assert _nearest_box(track, 1.02) is not None
    assert _nearest_box(track, 4.) is None


def test_cash_link_needs_grounding_inside_event_interval():
    artifacts = PerceptionArtifacts()
    objects = ground_objects(object_payload(), [1., 2.], artifacts, "m", "v7")
    payload = {"events": [{"objects": ["c1"]}]}
    event = {"_source_index": 0, "kind": "cash_on_desk", "subjects": [], "start_sec": 1., "end_sec": 2.}
    bind_grounded_events(payload, [event], objects)
    assert event["relation_id"] == f"cash:{objects['c1'][0].track_id}"
    assert event["subjects"] == [objects["c1"][0].track_id]
    outside = {"_source_index": 0, "kind": "cash_on_desk", "subjects": [], "start_sec": 2., "end_sec": 3.}
    bind_grounded_events(payload, [outside], objects)
    assert "relation_id" not in outside
