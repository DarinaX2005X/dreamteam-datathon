"""Optional constrained-decoding helpers for the temporal model payload."""

from __future__ import annotations

from typing import Any

EVENT_KINDS = (
    "cash_in_hands", "cash_on_desk", "cash_handover", "counting_motion",
    "cash_sorting", "cash_packing", "bag_opened", "bag_integrity_damage",
    "personal_item_identified", "camera_device_visible", "phone_visible", "bag_present",
)
ROLES = ("cashier", "client", "unknown")


def _array(items: dict[str, Any], maximum: int, *, minimum: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def build_schema(frame_count: int, known_track_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Build the exact bounded JSON payload schema for one semantic window."""
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be positive")
    if any(not isinstance(value, str) or not value for value in known_track_ids):
        raise ValueError("known_track_ids must contain non-empty strings")
    track_ids = list(dict.fromkeys(known_track_ids))
    frame_index = {"type": "integer", "enum": list(range(frame_count))}
    event_index = {"type": "integer", "enum": list(range(6))}
    # An empty enum is not a valid JSON Schema enum and some parsers treat it
    # as unrestricted. Without people, arrays of track IDs must stay empty.
    track_id = {"type": "string", **({"enum": track_ids} if track_ids else {})}
    nullable_track_id = ({"anyOf": [track_id, {"type": "null"}]}
                         if track_ids else {"type": "null"})
    event = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(EVENT_KINDS)},
            "start_frame": frame_index,
            "end_frame": frame_index,
            "subjects": _array(track_id, len(track_ids)),
            "actor": nullable_track_id,
            "recipient": nullable_track_id,
        },
        "required": ["kind", "start_frame", "end_frame", "subjects", "actor", "recipient"],
        "additionalProperties": False,
    }
    role = {
        "type": "object",
        "properties": {"track": track_id, "role": {"type": "string", "enum": list(ROLES)}},
        "required": ["track", "role"],
        "additionalProperties": False,
    }
    batch = {
        "type": "object",
        "properties": {
            "events": _array(event_index, 6, minimum=2),
            "frames": _array(frame_index, frame_count, minimum=2),
        },
        "required": ["events", "frames"],
        "additionalProperties": False,
    }
    reflection = {
        "type": "object",
        "properties": {
            "reflected_track": track_id,
            "source_track": track_id,
            # Distinctness remains a semantic validator concern. Tail windows
            # with two frames must not create an impossible min>max branch.
            "frames": _array(frame_index, max(3, frame_count), minimum=3),
        },
        "required": ["reflected_track", "source_track", "frames"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "events": _array(event, 6),
            "roles": _array(role, 24 if track_ids else 0),
            "same_cash_batches": _array(batch, 12),
            "reflections": _array(reflection, 8 if track_ids else 0),
        },
        "required": ["events", "roles", "same_cash_batches", "reflections"],
        "additionalProperties": False,
    }


def build_prefix(tokenizer, schema: dict[str, Any], *, tokenizer_data=None):
    """Build a Transformers prefix function; pass warmed tokenizer_data to reuse its trie."""
    if schema["properties"]["roles"]["maxItems"] == 0:
        # LMFE 0.11.3 treats maxItems=0 as its default maximum. Do not claim
        # constrained IDs in a window without any known people; the caller
        # must use the ordinary verifier and retain its post-validation.
        raise ValueError("LMFE cannot enforce an empty set of known track IDs")
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer_data if tokenizer_data is not None else tokenizer,
        JsonSchemaParser(schema),
    )
