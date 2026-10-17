"""Measured analysis gates, bound to the exact local inference configuration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .configuration import snapshot_hash

SEMANTIC_CAPABILITIES = frozenset({
    "cash_detection", "handover_direction", "counting_action", "bag_damage",
    "bag_opening", "personal_item_semantics", "cash_on_desk", "clear_person_count",
})


def validation_binding(snapshot: dict, *, devices: dict[str, str] | None = None) -> str:
    # Evidence affects result reuse, but cannot be part of its own key. The
    # publication flags change access rather than inference, so a measured
    # experimental/diagnostic setup can promote the identical production setup.
    algorithm = {key: value for key, value in snapshot.items()
                 if key not in {"validation", "diagnostic_only", "experimental_analysis"}}
    algorithm["implemented_semantic_capabilities"] = sorted(SEMANTIC_CAPABILITIES)
    if devices is not None:
        algorithm["devices"] = devices
    return snapshot_hash(algorithm)


def _read_evidence(directory: Path, evidence: dict) -> dict | None:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
        return None
    path = (directory / evidence["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != evidence.get("sha256"):
        return None
    value = json.loads(data)
    return value if isinstance(value, dict) else None


def _end_to_end_capabilities(directory: Path, profile: dict, binding: str) -> dict[str, bool]:
    disabled = dict.fromkeys(SEMANTIC_CAPABILITIES, False)
    if profile.get("accepted") is not True:
        return disabled
    report = _read_evidence(directory, profile.get("evidence"))
    if report is None or report.get("schema_version") != 1 \
            or report.get("algorithm_fingerprint") != binding:
        return disabled
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict) or metrics.get("mode") != "semantic_pipeline_evaluation":
        return disabled
    counts, verdict = metrics.get("counts", {}), metrics.get("verdict", {})
    if not isinstance(counts, dict) or not isinstance(verdict, dict):
        return disabled
    selected = counts.get("selected_unique_sources")
    if type(selected) is not int or selected < 5 \
            or counts.get("completed_unique_sources") != selected \
            or counts.get("failed_unique_sources") != 0 \
            or counts.get("missing_unique_sources") != 0 \
            or type(counts.get("true_negatives_with_checkable_rules")) is not int \
            or counts["true_negatives_with_checkable_rules"] < 1 \
            or counts.get("completed_sources_with_all_rules_not_checkable", selected) == selected:
        return disabled
    confusion = [verdict.get(key) for key in ("tp", "fp", "tn", "fn")]
    if any(type(value) is not int or value < 0 for value in confusion):
        return disabled
    tp, fp, tn, fn = confusion
    if tp < 1 or tn < 1 or tp + fp + tn + fn != selected:
        return disabled
    acceptance = profile.get("acceptance", {})
    if not isinstance(acceptance, dict):
        return disabled
    minimum_precision, minimum_recall = acceptance.get("min_precision"), acceptance.get("min_recall")
    maximum_false_positive = acceptance.get("max_false_positive_rate_clean")
    maximum_timing_error = acceptance.get("max_p90_time_error_sec")
    limits = (minimum_precision, minimum_recall, maximum_false_positive, maximum_timing_error)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in limits) \
            or not 0 < minimum_precision <= 1 or not 0 < minimum_recall <= 1 \
            or not 0 <= maximum_false_positive < 1 or maximum_timing_error < 0:
        return disabled
    timing, latency = metrics.get("timing_error_sec", {}), metrics.get("latency_sec", {})
    if not isinstance(timing, dict) or not isinstance(latency, dict) \
            or type(timing.get("count")) is not int or timing["count"] < 1 \
            or type(timing.get("p90")) not in (int, float) \
            or not math.isfinite(timing["p90"]) or timing["p90"] > maximum_timing_error \
            or latency.get("count") != selected \
            or type(latency.get("p90")) not in (int, float) \
            or not math.isfinite(latency["p90"]) or latency["p90"] <= 0:
        return disabled
    if tp / (tp + fp) < minimum_precision or tp / (tp + fn) < minimum_recall \
            or fp / (tn + fp) > maximum_false_positive:
        return disabled
    implemented = report.get("candidate_semantic_capabilities")
    if not isinstance(implemented, list) or any(not isinstance(item, str) for item in implemented):
        return disabled
    # Binary/timestamp evaluation establishes the end-to-end operating profile,
    # not individual feature accuracy. Each enabled flag still only means that
    # the matching measured pipeline implements this visible observation type.
    return {name: name in implemented for name in SEMANTIC_CAPABILITIES}


def validated_capabilities(directory: Path, snapshot: dict, *,
                           devices: dict[str, str] | None = None,
                           end_to_end_only: bool = False) -> tuple[dict[str, bool], list[str]]:
    capabilities = dict.fromkeys(SEMANTIC_CAPABILITIES, False)
    path = directory / "validation.json"
    if not path.is_file():
        return capabilities, ["Проверка качества анализа действий ещё не завершена."]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or not isinstance(value.get("profiles"), list):
            raise ValueError("invalid schema")
        binding = validation_binding(snapshot, devices=devices)
        profiles = [item for item in value["profiles"] if isinstance(item, dict)
                    and item.get("binding") == binding]
        if len(profiles) != 1:
            return capabilities, ["Для текущей версии модели ещё нет проверки качества действий."]
        if profiles[0].get("evidence_level") == "end_to_end":
            capabilities = _end_to_end_capabilities(directory, profiles[0], binding)
            limitations = [] if any(capabilities.values()) else ["Проверка качества анализа действий ещё не завершена."]
            return capabilities, limitations
        if end_to_end_only:
            return capabilities, ["Для этой модели нужна проверка полного анализа записей."]
        reports = profiles[0].get("capabilities")
        if not isinstance(reports, dict):
            raise ValueError("invalid capability report")
        for name in SEMANTIC_CAPABILITIES:
            report = reports.get(name)
            if not isinstance(report, dict) or report.get("passed") is not True:
                continue
            # A successful invocation or an empty-output smoke test cannot enable
            # a semantic feature. Both positive and negative labelled examples
            # must have actual measured counts in a retained evidence artifact.
            counts = [report.get(key) for key in ("true_positive", "false_positive", "false_negative", "true_negative")]
            if any(type(count) is not int or count < 0 for count in counts):
                continue
            tp, fp, fn, tn = counts
            if tp < 1 or tn < 1 or tp / (tp + fp) < 0.8 or tp / (tp + fn) < 0.8:
                continue
            evidence = report.get("evidence")
            if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
                continue
            evidence_path = (directory / evidence["path"]).resolve()
            # Evidence is a local structured artifact, never a remote resource.
            if not evidence_path.is_relative_to(directory.resolve()) or not evidence_path.is_file():
                continue
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence.get("sha256"):
                continue
            measured = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(measured, dict) or measured.get("schema_version") != 1 \
                    or measured.get("binding") != binding \
                    or measured.get("capability") != name:
                continue
            examples = measured.get("examples")
            if not isinstance(examples, list) or not examples:
                continue
            measured_counts = [0, 0, 0, 0]
            identities = set()
            for example in examples:
                if not isinstance(example, dict) or type(example.get("expected")) is not bool \
                        or type(example.get("predicted")) is not bool \
                        or not isinstance(example.get("sample_id"), str) \
                        or not example["sample_id"] \
                        or example.get("label_source") not in {"human_feature_label", "provided_feature_label"} \
                        or example["sample_id"] in identities:
                    break
                identities.add(example["sample_id"])
                expected, predicted = example["expected"], example["predicted"]
                measured_counts[0 if expected and predicted else 1 if predicted else 2 if expected else 3] += 1
            else:
                if measured_counts == counts:
                    capabilities[name] = True
    except (OSError, ValueError, TypeError, AttributeError):
        return dict.fromkeys(SEMANTIC_CAPABILITIES, False), ["Файл проверки качества модели повреждён."]
    limitations = [] if any(capabilities.values()) else ["Проверка качества анализа действий ещё не завершена."]
    return capabilities, limitations
