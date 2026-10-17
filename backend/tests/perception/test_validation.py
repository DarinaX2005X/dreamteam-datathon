"""An installed model and a successful call are not feature-quality evidence."""

import hashlib
import json

import pytest
from navidu.perception.validation import validated_capabilities, validation_binding


def _evidence(tmp_path, *, snapshot=None):
    snapshot = snapshot or {"model": "test-pinned", "options": {"fps": 1}}
    binding = validation_binding(snapshot)
    evidence = {"schema_version": 1, "binding": binding, "capability": "cash_detection",
        "examples": [
            {"sample_id": "development-positive", "expected": True, "predicted": True,
             "label_source": "human_feature_label"},
            {"sample_id": "development-negative", "expected": False, "predicted": False,
             "label_source": "human_feature_label"},
        ]}
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    report = {"passed": True, "true_positive": 1, "false_positive": 0,
        "false_negative": 0, "true_negative": 1,
        "evidence": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    profile = {"binding": binding, "capabilities": {"cash_detection": report}}
    (tmp_path / "validation.json").write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
    return snapshot, profile, evidence


def test_no_evidence_never_enables_a_feature(tmp_path):
    capabilities, limitations = validated_capabilities(tmp_path, {})
    assert not any(capabilities.values())
    assert limitations


def test_only_matching_measured_feature_is_enabled(tmp_path):
    snapshot, _, _ = _evidence(tmp_path)
    capabilities, limitations = validated_capabilities(tmp_path, snapshot)
    assert capabilities["cash_detection"] is True
    assert all(not value for key, value in capabilities.items() if key != "cash_detection")
    assert limitations == []
    assert not any(validated_capabilities(tmp_path, {**snapshot, "model": "different"})[0].values())
    assert not any(validated_capabilities(tmp_path, snapshot, end_to_end_only=True)[0].values())


def test_changed_evidence_cannot_enable_a_feature(tmp_path):
    snapshot, _, _ = _evidence(tmp_path)
    (tmp_path / "measured.json").write_text("{}", encoding="utf-8")
    assert not any(validated_capabilities(tmp_path, snapshot)[0].values())


@pytest.mark.parametrize("change", [
    {"true_positive": 0}, {"true_negative": 0}, {"false_positive": 1},
    {"false_negative": 1}, {"true_positive": True}, {"passed": False},
    {"true_positive": 99},
])
def test_unmeasured_or_poor_feature_cannot_be_enabled(tmp_path, change):
    snapshot, profile, _ = _evidence(tmp_path)
    profile["capabilities"]["cash_detection"].update(change)
    (tmp_path / "validation.json").write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
    assert not any(validated_capabilities(tmp_path, snapshot)[0].values())


def test_validation_artifact_does_not_recursively_change_its_binding():
    assert validation_binding({"model": "x", "validation": {"sha256": "first"}}) == \
        validation_binding({"model": "x", "validation": {"sha256": "second"}})


def _end_to_end_evidence(tmp_path):
    snapshot = {"model": "test-pinned", "diagnostic_only": True}
    binding = validation_binding(snapshot)
    report = {"schema_version": 1, "algorithm_fingerprint": binding,
        "candidate_semantic_capabilities": ["cash_detection", "counting_action"],
        "metrics": {"mode": "semantic_pipeline_evaluation",
            "counts": {"selected_unique_sources": 5, "completed_unique_sources": 5,
                "failed_unique_sources": 0, "missing_unique_sources": 0,
                "true_negatives_with_checkable_rules": 3,
                "completed_sources_with_all_rules_not_checkable": 0},
            "verdict": {"tp": 2, "fp": 0, "tn": 3, "fn": 0},
            "timing_error_sec": {"count": 2, "p90": 1.0},
            "latency_sec": {"count": 5, "p90": 90.0},
            "classification": {"available": False, "accuracy": None}}}
    path = tmp_path / "end-to-end.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    profile = {"binding": binding, "evidence_level": "end_to_end", "accepted": True,
        "acceptance": {"min_precision": 0.8, "min_recall": 0.8,
            "max_false_positive_rate_clean": 0.2, "max_p90_time_error_sec": 2.0},
        "evidence": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path / "validation.json").write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
    return snapshot, profile, report


def test_end_to_end_profile_enables_implemented_features_without_inventing_feature_labels(tmp_path):
    snapshot, _, _ = _end_to_end_evidence(tmp_path)
    # Publication purpose and the validation document cannot invalidate their
    # own measured algorithm; actual inference inputs still do.
    capabilities, _ = validated_capabilities(tmp_path, {**snapshot, "diagnostic_only": False})
    assert capabilities["cash_detection"]
    assert capabilities["counting_action"]
    assert not capabilities["bag_damage"]
    assert not any(validated_capabilities(tmp_path, {**snapshot, "model": "changed"})[0].values())


def test_end_to_end_profile_requires_explicit_acceptance(tmp_path):
    snapshot, profile, _ = _end_to_end_evidence(tmp_path)
    profile["accepted"] = False
    (tmp_path / "validation.json").write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
    assert not any(validated_capabilities(tmp_path, snapshot)[0].values())


@pytest.mark.parametrize("metric_changes", [
    {"verdict": {"tp": 0, "fp": 0, "tn": 3, "fn": 2}},
    {"verdict": {"tp": 2, "fp": 2, "tn": 1, "fn": 0}},
    {"timing_error_sec": {"count": 0, "p90": None}},
    {"latency_sec": {"count": 0, "p90": None}},
    {"mode": "detector_only_diagnostic"},
])
def test_empty_or_failed_end_to_end_evidence_cannot_enable_analysis(tmp_path, metric_changes):
    snapshot, profile, report = _end_to_end_evidence(tmp_path)
    report["metrics"].update(metric_changes)
    path = tmp_path / "end-to-end.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    profile["evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "validation.json").write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
    assert not any(validated_capabilities(tmp_path, snapshot)[0].values())


def test_diagnostic_measurement_binding_matches_production_configuration():
    production = {"model": "x", "diagnostic_only": False}
    diagnostic = {"model": "x", "diagnostic_only": True}
    assert validation_binding(production) == validation_binding(diagnostic)
