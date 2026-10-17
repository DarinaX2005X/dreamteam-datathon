"""Aggregate actual pipeline outputs without exporting private clip observations."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from eval.dataset import EvaluationSource


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear interpolation at (n - 1) * q, matching NumPy's default method."""
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("Percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values), "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9), "max": max(values) if values else None,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def confusion_metrics(truths_and_predictions: list[tuple[int, int]]) -> dict[str, Any]:
    counts = Counter(truths_and_predictions)
    tp, fp, tn, fn = counts[(1, 1)], counts[(0, 1)], counts[(0, 0)], counts[(1, 0)]
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "evaluated": tp + fp + tn + fn,
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "precision": _ratio(tp, tp + fp), "recall": recall,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "specificity": specificity, "false_positive_rate_clean": _ratio(fp, fp + tn),
        "clean_evaluated": tn + fp, "positive_evaluated": tp + fn,
    }


def validate_prediction(record: dict, source: EvaluationSource, *, fingerprint: str | None = None) -> None:
    if record.get("schema_version") != 1 or record.get("source_sha256") != source.sha256:
        raise ValueError("Prediction identity/schema mismatch")
    if fingerprint is not None and record.get("fingerprint") != fingerprint:
        raise ValueError("Prediction belongs to a different effective pipeline")
    if record.get("status") not in {"done", "failed", "cancelled"}:
        raise ValueError("Only finalized attempts may enter an aggregate")
    attempts = record.get("previous_failed_attempts", 0)
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError("Previous failure count must be a nonnegative integer")
    elapsed = record.get("latency_sec")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Latency must be a finite nonnegative measurement")
    if record["status"] != "done":
        if not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]{0,80}", str(record.get("failure_code", record["status"]))):
            raise ValueError("Failure reports must use a code, never a private model description")
        return
    if record.get("verdict") not in {"violation", "no_violation"}:
        raise ValueError("Completed prediction lacks the required binary verdict")
    findings = record.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Completed prediction lacks findings")
    if (record["verdict"] == "violation") != bool(findings):
        raise ValueError("Verdict and actual automatic findings disagree")
    for finding in findings:
        timestamp = finding.get("timecode_sec")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or not 0 <= timestamp <= source.duration_sec:
            raise ValueError("A finding timestamp lies outside its source video")


def aggregate(
    sources: list[EvaluationSource], records: dict[str, dict], *,
    fingerprint: str | None = None, detector_only: bool = False,
) -> dict[str, Any]:
    """One vote per SHA. Row-level duplicate statistics are separate and labelled."""
    if len({source.sha256 for source in sources}) != len(sources):
        raise ValueError("Sources must already be deduplicated by SHA256")
    known = {source.sha256 for source in sources}
    if not set(records).issubset(known):
        raise ValueError("Prediction contains a source outside the selected evaluation partition")
    pairs, row_pairs = [], []
    latencies, failed_latencies, timing_errors, row_timing = [], [], [], []
    failures = Counter()
    failure_attempts = 0
    rule_counts = Counter()
    completed = 0
    ambiguous_detected = 0
    eligible_detected = 0
    missing = 0
    source_count = len(sources)
    not_checkable = 0
    analysis_modes = Counter()
    analysis_ready = 0
    versions: set[tuple[tuple[str, str], ...]] = set()
    for source in sources:
        record = records.get(source.sha256)
        if record is None:
            missing += 1
            continue
        validate_prediction(record, source, fingerprint=fingerprint)
        failure_attempts += int(record.get("previous_failed_attempts", 0))
        if record["status"] != "done":
            failures[str(record.get("failure_code", record["status"]))] += 1
            failed_latencies.append(float(record["latency_sec"]))
            failure_attempts += 1
            continue
        completed += 1
        analysis_modes.update([str(record.get("analysis_mode", "unknown"))])
        analysis_ready += int(record.get("analysis_ready") is True)
        versions.add(tuple(sorted(record.get("model_versions", {}).items())))
        latencies.append(float(record["latency_sec"]))
        predicted = int(record["verdict"] == "violation")
        pairs.append((source.has_violation, predicted))
        row_pairs.extend([(source.has_violation, predicted)] * len(source.rows))
        statuses = record.get("rule_statuses", [])
        rule_counts.update(statuses)
        if statuses and all(status == "not_checkable" for status in statuses):
            not_checkable += 1
        if source.has_violation and predicted:
            timestamps = [float(finding["timecode_sec"]) for finding in record["findings"]]
            for row in source.rows:
                if row.label_sec is not None:
                    row_timing.append(min(abs(timestamp - row.label_sec) for timestamp in timestamps))
            if source.timing_ambiguous:
                ambiguous_detected += 1
            elif len(source.label_secs) == 1:
                eligible_detected += 1
                timing_errors.append(min(abs(timestamp - source.label_secs[0]) for timestamp in timestamps))
    confusion = confusion_metrics(pairs)
    labelled_types = sum(bool(row.type_code) for source in sources for row in source.rows)
    result = {
        "schema_version": 1,
        "mode": "detector_only_diagnostic" if detector_only else "semantic_pipeline_evaluation",
        "counts": {
            "selected_unique_sources": source_count,
            "selected_source_rows": sum(len(source.rows) for source in sources),
            "selected_positive_sources": sum(source.has_violation for source in sources),
            "selected_clean_sources": sum(1 - source.has_violation for source in sources),
            "completed_unique_sources": completed,
            "failed_unique_sources": sum(failures.values()),
            "missing_unique_sources": missing,
            "previous_and_current_failed_attempts": failure_attempts,
            "sources_with_ambiguous_timing": sum(source.timing_ambiguous for source in sources),
            "completed_sources_with_all_rules_not_checkable": not_checkable,
            "completed_sources_marked_analysis_ready": analysis_ready,
        },
        "completion_rate": _ratio(completed, source_count),
        "failure_codes": dict(sorted(failures.items())),
        "latency_sec": {**distribution(latencies), "scope": "Actual per-video analyze wall time; warm model preparation measured separately; cache retrieval not counted."},
        "failed_attempt_latency_sec": distribution(failed_latencies),
        "rule_status_counts": dict(sorted(rule_counts.items())),
        "analysis_mode_counts": dict(sorted(analysis_modes.items())),
        "model_version_sets": [dict(values) for values in sorted(versions)],
        "classification": {
            "available": False, "accuracy": None, "rows_with_type_annotations": labelled_types,
            "reason": "No independently verified type annotations are provided." if not labelled_types else "Type matching is not implemented; existing labels must be independently verified first.",
        },
    }
    if detector_only:
        result.update({"verdict": None, "timing_error_sec": None, "row_level_diagnostic": None,
                       "quality_note": "Object detection only. Binary quality and event localization metrics are deliberately withheld."})
        return result
    result["verdict"] = {
        **confusion,
        "denominator": "Completed unique sources only; failures and missing inputs are reported separately.",
        "selected_accuracy_lower_bound": _ratio(confusion["tp"] + confusion["tn"], source_count),
        "positive_recall_with_failures_as_misses": _ratio(confusion["tp"], sum(source.has_violation for source in sources)),
    }
    result["timing_error_sec"] = {
        **distribution(timing_errors),
        "correctly_detected_unambiguous_positive_sources": eligible_detected,
        "excluded_detected_ambiguous_positive_sources": ambiguous_detected,
        "missed_or_failed_positive_sources": sum(source.has_violation for source in sources) - confusion["tp"],
        "method": "Nearest actual finding to the sole organizer timestamp; only correctly detected unique positive sources with one unambiguous label. This is not one-to-one event matching and may favour videos with more findings.",
        "ambiguous_policy": "Different timestamps on byte-identical copies are inconsistent annotations, not two confirmed events; exclude the entire source from headline timing.",
    }
    result["row_level_diagnostic"] = {
        "verdict": confusion_metrics(row_pairs),
        "timing_error_sec": distribution(row_timing),
        "note": "Original rows are counted separately, including duplicate files and their distinct timestamp annotations. These are diagnostics, not independent-sample headline metrics.",
    }
    return result


def markdown_report(report: dict) -> str:
    """Only whitelisted aggregate numeric/runtime fields enter a shareable report."""
    metrics = report["metrics"]
    counts = metrics["counts"]
    fingerprint = report.get("fingerprint", "unknown")
    lines = [
        "# Локальная оценка конвейера", "",
        f"Дата: {report.get('created_at', 'unknown')}. Раздел: `{report.get('split', 'unknown')}`.",
        f"Профиль: `{metrics['mode']}`. Версия: `{report.get('pipeline_version', 'unknown')}`.",
        f"Fingerprint: `{fingerprint}`.", "",
        f"Алгоритм: `{report.get('algorithm_fingerprint', 'unknown')}`.", "",
        "Запуски выполнены через явный диагностический режим. Эти артефакты не предназначены для публикации как готовые пользовательские проверки; численные метрики сами по себе не включают продуктовый статус готовности.", "",
        "## Охват", "",
        f"Выбрано {counts['selected_unique_sources']} уникальных источников / {counts['selected_source_rows']} исходных строк. Завершено {counts['completed_unique_sources']}; ошибок {counts['failed_unique_sources']}; без результата {counts['missing_unique_sources']}.",
        f"Положительных источников {counts['selected_positive_sources']}, отрицательных {counts['selected_clean_sources']}. Спорных временных аннотаций {counts['sources_with_ambiguous_timing']}.",
        f"Завершённых источников, у которых все правила помечены непроверяемыми: {counts['completed_sources_with_all_rules_not_checkable']}. Это не подтверждение корректности действий на записи.",
        "",
    ]
    if report.get("split") == "all":
        lines.extend(["## Разделы выборки", "",
                      "Объединённые числа описывают все обработанные источники. Независимой итоговой проверкой остаётся только holdout; development использовался при разработке.", "",
                      "| Раздел | Выбрано | Завершено | Ошибок | Без результата | TP | FP | TN | FN |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for partition, values in report.get("metrics_by_partition", {}).items():
            partition_counts, partition_verdict = values["counts"], values["verdict"]
            cells = [partition, *[str(partition_counts[key]) for key in (
                "selected_unique_sources", "completed_unique_sources", "failed_unique_sources", "missing_unique_sources")],
                *[str(partition_verdict[key]) for key in ("tp", "fp", "tn", "fn")]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    verdict = metrics["verdict"]
    if verdict is None:
        lines.extend(["## Диагностический запуск", "", "Временной анализ отключён. Качество обнаружения нарушений и локализация не оцениваются.", ""])
    else:
        lines.extend(["## Вердикт по уникальным источникам", "", "| Метрика | Значение |", "| --- | ---: |"])
        for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "false_positive_rate_clean", "selected_accuracy_lower_bound", "positive_recall_with_failures_as_misses"):
            value = verdict[key]
            text = "недоступно" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"| {key} | {text} |")
        lines.extend(["", "Основные precision/recall/F1/accuracy используют только завершённые уникальные источники. Дополнительно показаны нижняя граница accuracy по всем выбранным и recall с ошибками/пропусками как недетектированными положительными.", ""])
        timing = metrics["timing_error_sec"]
        lines.extend(["## Таймкод", "", f"Сопоставлено {timing['count']} корректно обнаруженных положительных источников с одной однозначной меткой. Исключено спорных обнаруженных источников: {timing['excluded_detected_ambiguous_positive_sources']}.", "", "| Ошибка, с | Значение |", "| --- | ---: |"])
        for key in ("median", "p90", "max"):
            value = timing[key]
            lines.append(f"| {key} | {'недоступно' if value is None else f'{value:.6f}'} |")
        lines.extend(["", "Для каждой исходной метки берётся ближайший фактический finding. Это не взаимно однозначное сопоставление событий; большое число findings может улучшать такую оценку. Различные метки одинаковых файлов не считаются двумя подтверждёнными событиями и исключены из основной временной метрики. Диагностика по исходным строкам хранится отдельно в JSON.", ""])
    lines.extend(["## Время обработки", "", "| Задержка, с | Значение |", "| --- | ---: |"])
    for key in ("median", "p90", "max"):
        value = metrics["latency_sec"][key]
        lines.append(f"| {key} | {'недоступно' if value is None else f'{value:.6f}'} |")
    lines.extend(["", "Измеряется полное время analyze для каждого видео при уже подготовленных весах. Время загрузки модели учитывается отдельно; чтение кэша не считается новым inference.", "", "## Ограничения", "", "Точность классификации недоступна: независимых меток типов нет. Разбиение группирует SHA256-копии, но не доказывает перенос на новых актёров, новую сцену или ракурс. Fold 0 предназначен для итоговой проверки после заморозки параметров; остальные фолды — разработка.", ""])
    runtime = report.get("runtime", {})
    lines.extend(["## Среда", "", f"Устройство: `{runtime.get('device', 'unknown')}`. Подготовка модели: `{report.get('prepare_sec', 'unknown')}` с."])
    versions = report.get("package_versions", {})
    for name, version in sorted(versions.items()):
        lines.append(f"- `{name}`: `{version}`")
    lines.append("")
    return "\n".join(lines)
