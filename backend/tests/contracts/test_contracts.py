import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image
from pydantic import TypeAdapter, ValidationError

from navidu.contracts.api_v1 import JobResponse, JuryViolation
from navidu.contracts.domain import Decision, Finding, TimeWindow, verdict_of
from navidu.contracts.generate import generate
from navidu.contracts.rules import RulesResponse, SourceClauseView
from navidu.contracts.ui_api import CheckDetailResponse, ReevaluateResponse
from navidu.contracts.units import finding_id

FIXTURES = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["v1_done_violation", "v1_done_clean", "v1_processing", "v1_failed"])
def test_jury_fixtures_and_exact_keys(name):
    result = TypeAdapter(JobResponse).validate_python(load(name))
    value = result.model_dump(mode="json")
    expected = {"job_id", "status"}
    if result.status == "done":
        expected |= {"verdict", "violations"}
        for item in value["violations"]:
            assert set(item) == {"type", "timecode_sec", "description", "evidence_frames"}
            for frame in item["evidence_frames"]:
                assert set(frame) == {"timestamp_sec", "image_base64"}
                image = Image.open(io.BytesIO(base64.b64decode(frame["image_base64"])))
                image.load()
                assert image.format == "JPEG" and image.size == (64, 36)
    elif result.status == "failed":
        expected.add("error")
    assert set(value) == expected


@pytest.mark.parametrize("name", ["ui_check_done", "ui_check_processing", "ui_check_failed"])
def test_ui_fixtures(name):
    CheckDetailResponse.model_validate(load(name))


def test_other_fixtures():
    ReevaluateResponse.model_validate(load("ui_reevaluate_diff"))
    RulesResponse.model_validate(load("ui_rules"))


def test_verdict_decisions_and_reset():
    finding_set = CheckDetailResponse.model_validate(load("ui_check_done")).check.finding_set
    assert verdict_of(finding_set) == "violation"
    for finding in finding_set.findings:
        finding.decision = Decision(value="rejected", note="", decided_at=datetime.now(timezone.utc))
    assert verdict_of(finding_set) == "no_violation"
    assert verdict_of(finding_set, use_decisions=False) == "violation"
    finding_set.findings[0].decision.value = "not_visible"
    assert verdict_of(finding_set) == "violation"
    finding_set.findings[0].decision = None
    assert verdict_of(finding_set) == "violation"


def test_ids_stable_on_equivalent_window():
    assert finding_id("V-COUNT-ABSENT", 41.501, 47.501) == finding_id("V-COUNT-ABSENT", 41.5, 47.5)
    assert finding_id("V-COUNT-ABSENT", 41.5, 47.5) != finding_id("V-COUNT-ABSENT", 42, 47.5)


@pytest.mark.parametrize("count", [0, 4])
def test_evidence_count_bounded(count):
    value = load("v1_done_violation")["violations"][0]
    value["evidence_frames"] = value["evidence_frames"] * count
    with pytest.raises(ValidationError):
        JuryViolation.model_validate(value)
    finding = load("ui_check_done")["check"]["finding_set"]["findings"][0]
    finding["evidence"] = finding["evidence"] * count
    with pytest.raises(ValidationError):
        Finding.model_validate(finding)


def test_evidence_time_and_no_extra_success_fields():
    value = load("v1_done_violation")
    value["violations"][0]["timecode_sec"] = 44.3
    with pytest.raises(ValidationError):
        TypeAdapter(JobResponse).validate_python(value)
    value = load("v1_done_clean")
    value["confidence"] = 0.99
    with pytest.raises(ValidationError):
        TypeAdapter(JobResponse).validate_python(value)


def test_invalid_time_and_unverified_quote():
    with pytest.raises(ValidationError):
        TimeWindow(start_sec=4, end_sec=3)
    with pytest.raises(ValidationError):
        TimeWindow(start_sec=float("nan"), end_sec=3)
    source = load("ui_rules")["rules"][0]["sources"][0]
    source["quote"] = "This was never verified."
    assert SourceClauseView.model_validate(source).quote is None


def test_generation_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr("navidu.contracts.generate.ROOT", tmp_path)
    first = generate()
    assert generate() == first
    assert first["paths"]["/api/v1/analyze"]["post"]["security"] == [{"ApiKey": []}]
