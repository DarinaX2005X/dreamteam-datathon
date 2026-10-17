import json

import pytest
from navidu.perception.structured_output import EVENT_KINDS, ROLES, build_prefix, build_schema
from navidu.perception.temporal import _ROLES as VALIDATED_ROLES
from navidu.perception.temporal import EVENT_KINDS as VALIDATED_EVENT_KINDS

JsonSchemaParser = pytest.importorskip("lmformatenforcer").JsonSchemaParser


def _accepted(schema, value) -> bool:
    parser = JsonSchemaParser(schema)
    for character in json.dumps(value, separators=(",", ":")):
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def _payload(events=None):
    return {"events": events or [], "roles": [], "same_cash_batches": [], "reflections": []}


def _handover(**updates):
    return {"kind": "cash_handover", "start_frame": 0, "end_frame": 2,
            "subjects": ["p1", "p2"], "actor": "p1", "recipient": "p2", **updates}


def test_empty_payload_is_valid_even_without_known_people():
    assert _accepted(build_schema(4, []), _payload())


def test_no_known_people_never_claims_unsupported_lmfe_zero_array_guarantee():
    schema = build_schema(2, [])
    event = {"kind": "cash_on_desk", "start_frame": 0, "end_frame": 1,
             "subjects": [], "actor": None, "recipient": None}
    assert _accepted(schema, _payload([event]))
    assert not _accepted(schema, _payload([{**event, "actor": "p1"}]))
    assert schema["properties"]["roles"]["maxItems"] == 0
    assert schema["properties"]["events"]["items"]["properties"]["subjects"]["maxItems"] == 0
    with pytest.raises(ValueError, match="empty set"):
        build_prefix(None, schema)


def test_schema_vocabulary_matches_post_validation():
    assert set(EVENT_KINDS) == VALIDATED_EVENT_KINDS
    assert set(ROLES) == VALIDATED_ROLES


def test_valid_handover_uses_window_ids_and_indices():
    assert _accepted(build_schema(4, ["p1", "p2"]), _payload([_handover()]))


def test_roles_relations_and_reflections_use_exact_current_keys():
    payload = _payload([_handover()])
    payload["roles"] = [{"track": "p1", "role": "cashier"}]
    payload["same_cash_batches"] = [{"events": [0, 1], "frames": [0, 2]}]
    payload["reflections"] = [{
        "reflected_track": "p2", "source_track": "p1", "frames": [0, 1, 2],
    }]
    assert _accepted(build_schema(4, ["p1", "p2"]), payload)


@pytest.mark.parametrize("event", [
    _handover(subjects=["p9"]),
    _handover(end_frame=4),
    _handover(kind="unsupported_action"),
])
def test_schema_rejects_unknown_ids_indices_and_kinds(event):
    assert not _accepted(build_schema(4, ["p1", "p2"]), _payload([event]))


def test_schema_rejects_more_than_six_events():
    assert not _accepted(build_schema(4, ["p1", "p2"]), _payload([_handover()] * 7))


def test_schema_rejects_unknown_keys_but_allows_nullable_actor_and_recipient():
    schema = build_schema(4, ["p1"])
    cash_on_desk = {"kind": "cash_on_desk", "start_frame": 0, "end_frame": 1,
                    "subjects": [], "actor": None, "recipient": None}
    assert _accepted(schema, _payload([cash_on_desk]))
    assert not _accepted(schema, {**_payload(), "commentary": "none"})
