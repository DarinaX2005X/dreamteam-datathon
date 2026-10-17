"""Learned proposals are constrained to the real input window and private metadata stays out."""

import json

import pytest
from navidu.perception.proposal_task import decode_proposal, proposal_messages


def test_model_prompt_is_independent_of_absolute_source_time():
    first = proposal_messages([(0.0, None), (.5, None), (1.0, None)])
    shifted = proposal_messages([(100.0, None), (100.5, None), (101.0, None)])
    assert first == shifted
    assert "100.500" not in json.dumps(shifted)


def test_candidate_maps_only_to_an_actual_source_frame():
    result = decode_proposal('{"candidate":1,"frame":1}', [8.333, 8.667, 9.0])
    assert result.candidate
    assert result.frame_index == 1
    assert result.time_sec == 8.667
    clean = decode_proposal('{"candidate":0,"frame":null}', [8.333, 8.667, 9.0])
    assert not clean.candidate
    assert clean.time_sec is None


@pytest.mark.parametrize("output", [
    '{"candidate":true,"frame":1}', '{"candidate":1,"frame":3}',
    '{"candidate":1,"frame":null}', '{"candidate":0,"frame":1}',
    '{"candidate":0,"frame":null,"reason":"private text"}',
    '{"candidate":1,"frame":1.5}', 'private unstructured generated text',
])
def test_invalid_proposals_are_rejected_without_echoing_model_output(output):
    with pytest.raises(RuntimeError, match="MODEL_OUTPUT_INVALID") as error:
        decode_proposal(output, [0.0, .5, 1.0])
    assert output not in str(error.value)


def test_nonincreasing_frames_cannot_become_a_model_sequence():
    with pytest.raises(ValueError, match="increasing"):
        proposal_messages([(1.0, None), (1.0, None)])
