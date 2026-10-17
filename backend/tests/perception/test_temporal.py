"""A structured model response cannot invent time, actors, or direction."""

import json

import pytest
from navidu.perception.temporal import _decode_payload, bind_cash_relations, validate_events


def _event(**updates):
    return {"kind": "cash_handover", "start_frame": 0, "end_frame": 1,
            "subjects": ["p1", "p2"], "actor": "p1", "recipient": "p2", **updates}


def _validate(event, frames=None):
    return validate_events({"events": [event], "roles": []},
        frames or [{"p1": (), "p2": ()}, {"p1": (), "p2": ()}], [17.333, 18.367])[0]


def test_handover_preserves_pts_and_explicit_direction():
    accepted = _validate(_event())
    assert len(accepted) == 1
    assert accepted[0]["start_sec"] == 17.333
    assert accepted[0]["end_sec"] == 18.367
    assert accepted[0]["actor_id"] == "p1"
    assert accepted[0]["recipient_id"] == "p2"


@pytest.mark.parametrize("changes", [
    {"end_frame": 999}, {"start_frame": -1}, {"start_frame": True},
    {"start_frame": 1, "end_frame": 0}, {"start_frame": 0, "end_frame": 0},
    {"subjects": ["p99"]}, {"actor": "p99"}, {"recipient": "p1"},
    {"recipient": None}, {"actor": None}, {"kind": "fraud"},
])
def test_invented_event_fields_are_discarded(changes):
    assert _validate(_event(**changes)) == []


def test_a_single_detection_does_not_bind_a_temporal_actor():
    assert _validate(_event(), [{"p1": (), "p2": ()}, {"p2": ()}]) == []


def test_roles_are_limited_to_visible_tracks_and_supported_categories():
    events, roles = validate_events({"events": [], "roles": [
        {"track": "p1", "role": "cashier"},
        {"track": "p8", "role": "client"},
        {"track": "p1", "role": "outsider"},
    ]}, [{"p1": ()}, {"p1": ()}], [0, 1])
    assert events == []
    assert roles == [{"track": "p1", "role": "cashier"}]


def test_invalid_json_does_not_become_an_empty_clean_result():
    with pytest.raises(RuntimeError, match="MODEL_OUTPUT_INVALID"):
        _decode_payload("The room looks fine.")
    assert _decode_payload('```json\n{"events":[],"roles":[]}\n```') == {"events": [], "roles": []}


def test_only_explicit_supported_cash_batch_binds_two_actions():
    payload = {"events": [_event(), _event(kind="counting_motion", actor="p2", recipient=None)],
               "roles": [], "same_cash_batches": [{"events": [0, 1], "frames": [0, 1]}]}
    frames = [{"p1": (), "p2": ()}, {"p1": (), "p2": ()}]
    events, _ = validate_events(payload, frames, [17.333, 18.367])
    bind_cash_relations(payload, events, frames, [17.333, 18.367], "test-model")
    assert events[0]["relation_id"] == events[1]["relation_id"]
    assert "000017333-000018367" in events[0]["relation_id"]


@pytest.mark.parametrize("batch", [
    {"events": [0, 1], "frames": [0]},
    {"events": [0, 99], "frames": [0, 1]},
    {"events": [0, 1], "frames": [0, 99]},
    {"events": [0, True], "frames": [0, 1]},
    {"events": [0, 0], "frames": [0, 1]},
    {"events": [0, 1], "frames": [0, 0]},
])
def test_invented_or_single_frame_relation_is_rejected(batch):
    payload = {"events": [_event(), _event(kind="counting_motion", actor="p2", recipient=None)],
               "roles": [], "same_cash_batches": [batch]}
    frames = [{"p1": (), "p2": ()}, {"p1": (), "p2": ()}]
    events, _ = validate_events(payload, frames, [17.333, 18.367])
    bind_cash_relations(payload, events, frames, [17.333, 18.367], "test-model")
    assert all("relation_id" not in event for event in events)


@pytest.mark.parametrize("field", ["events", "roles", "same_cash_batches"])
def test_malformed_top_level_collection_is_not_a_clean_output(field):
    with pytest.raises(RuntimeError, match="MODEL_OUTPUT_INVALID"):
        _decode_payload(json.dumps({"events": [], "roles": [], "same_cash_batches": [], field: {}}))
