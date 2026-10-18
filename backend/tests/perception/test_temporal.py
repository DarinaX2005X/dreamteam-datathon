"""A structured model response cannot invent time, actors, or direction."""

import json

import pytest

from navidu.perception.temporal import (
    TemporalVerifier,
    _decode_payload,
    bind_cash_relations,
    validate_events,
)
from navidu.pipeline.interfaces import PerceptionArtifacts


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
    {"recipient": None}, {"actor": None}, {"kind": "fraud"}, {"kind": "bag_left_unattended"},
])
def test_invented_event_fields_are_discarded(changes):
    assert _validate(_event(**changes)) == []


def test_a_single_detection_does_not_bind_a_temporal_actor():
    assert _validate(_event(), [{"p1": (), "p2": ()}, {"p2": ()}]) == []


@pytest.mark.parametrize("variant", ["v9-questions", "v10-questions"])
def test_question_profile_preserves_unattended_object_state_without_inventing_a_person(variant):
    verifier = TemporalVerifier.__new__(TemporalVerifier)
    verifier.model_id = "test"
    verifier.prompt_variant = variant
    verifier.prompt_version = "test-questions"
    verifier.last_metrics = {}
    verifier.capabilities = {"clear_person_count": True}
    payload = {"events": [{"kind": "bag_present", "start_frame": 0, "end_frame": 1,
        "subjects": [], "actor": None, "recipient": None, "objects": []}],
        "roles": [], "visible_people": [], "objects": [], "reflections": [],
        "same_cash_batches": []}
    verifier._infer = lambda *args: payload
    artifacts = PerceptionArtifacts()
    verifier.observe([(17., None), (18., None)], [], artifacts)
    assert [(event.kind, event.subjects, event.actor_id) for event in artifacts.observations] == [
        ("bag_present", [], None)]
    assert artifacts.capabilities["clear_person_count"] is True
    assert artifacts.semantic_windows_completed == 1
    if variant == "v10-questions":
        payload["_question_events"] = [{"kind": "bag_left_unattended", "start_sec": 17.,
            "end_sec": 18., "subjects": [], "actor_id": None, "recipient_id": None}]
        checked = PerceptionArtifacts()
        verifier.observe([(17., None), (18., None)], [], checked)
        event = next(item for item in checked.observations if item.kind == "bag_left_unattended")
        assert event.subjects == [] and event.actor_id is None
        assert len(event.frame_ids) >= 2
    del payload["visible_people"]
    with pytest.raises(RuntimeError, match="MODEL_OUTPUT_INVALID"):
        verifier.observe([(17., None), (18., None)], [], PerceptionArtifacts())


@pytest.mark.parametrize("kind", ["cash_in_both_hands", "cash_in_counter"])
def test_specific_money_states_remain_states_without_inventing_counting(kind):
    accepted = _validate(_event(kind=kind, actor="p1", recipient=None, subjects=["p1"]))
    assert len(accepted) == 1 and accepted[0]["kind"] == kind
    assert accepted[0]["recipient_id"] is None
    assert all(event["kind"] != "counting_motion" for event in accepted)
    assert not _validate(_event(kind=kind, actor="p1", recipient="p2"))


def test_event_diagnostics_distinguish_index_and_actor_errors_without_private_values():
    private = "private-generated-value"
    payload = {"events": [_event(), _event(start_frame=17, end_frame=18),
        _event(end_frame=0), _event(start_frame="0"), _event(subjects=[private]),
        _event(actor=private), _event(kind=private), _event(actor=None),
        _event(kind="cash_sorting", recipient="p2"), _event(recipient="p1")], "roles": []}
    frames = [{"p1": (), "p2": ()}] * 2
    counters = {}
    result = validate_events(payload, frames, [17.333, 18.367], diagnostics=counters)
    assert result == validate_events(payload, frames, [17.333, 18.367])
    assert len(result[0]) == 1
    assert counters["raw_kind_cash_handover"] == 8
    assert counters["accepted_kind_cash_handover"] == 1
    for reason in ("frame_range", "frame_order_or_single", "frame_type", "subject_support",
                   "actor_support", "kind_or_shape", "action_without_actor",
                   "unexpected_recipient", "handover_recipient"):
        assert counters[f"rejected_{reason}"] == 1
    assert private not in json.dumps(counters)
    assert all(type(value) is int for value in counters.values())


def test_event_diagnostics_count_unprocessed_overflow_without_loosening_limit():
    counters = {}
    accepted, _ = validate_events({"events": [_event()] * 26},
        [{"p1": (), "p2": ()}] * 2, [0., 1.], diagnostics=counters)
    assert len(accepted) == 24
    assert counters["events_over_processing_limit"] == 2


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


def test_cash_relations_do_not_collide_when_dense_context_has_same_endpoints():
    relations = []
    for times in ([0., 1., 2.], [0., .5, 2.]):
        payload = {"events": [_event(end_frame=2),
            _event(kind="counting_motion", actor="p2", recipient=None, end_frame=2)],
            "same_cash_batches": [{"events": [0, 1], "frames": [0, 2]}]}
        frames = [{"p1": (), "p2": ()}] * 3
        events, _ = validate_events(payload, frames, times)
        bind_cash_relations(payload, events, frames, times, "base")
        relations.append(events[0]["relation_id"])
    assert relations[0] != relations[1]
