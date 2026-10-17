from __future__ import annotations

import unittest
from pathlib import Path

from eval.dataset import EvaluationSource, SourceRow, load_sources, select_sources
from eval.metrics import aggregate, markdown_report, percentile, validate_prediction

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "f" * 64


def source(key: str, truth: int, labels: tuple[float | None, ...], *, fold: int = 1) -> EvaluationSource:
    return EvaluationSource(key * 64, truth, fold, 100.0,
                            tuple(SourceRow(f"private-{key}-{index}", f"open/videos/{key}-{index}.mov", label, "") for index, label in enumerate(labels)))


def prediction(item: EvaluationSource, verdict: int, times: tuple[float, ...] = (), latency: float = 1.0) -> dict:
    return {"schema_version": 1, "source_sha256": item.sha256, "fingerprint": FINGERPRINT,
            "pipeline_version": "test-only", "status": "done", "latency_sec": latency,
            "analysis_ready": False, "analysis_mode": "diagnostic",
            "verdict": "violation" if verdict else "no_violation",
            "findings": [{"timecode_sec": time, "rule_id": "synthetic-test-rule"} for time in times],
            "model_versions": {"synthetic-unit-test": "0"}}


class MetricsTests(unittest.TestCase):
    def test_sha_headline_confusion_failures_and_rows_are_separate(self) -> None:
        a = source("a", 1, (57.0, 68.0))
        b = source("b", 1, (10.0,))
        c = source("c", 0, (None,))
        d = source("d", 0, (None, None))
        e = source("e", 1, (20.0,))
        f = source("f", 0, (None,))
        records = {a.sha256: prediction(a, 1, (58.0,), 1), b.sha256: prediction(b, 0, (), 2),
                   c.sha256: prediction(c, 1, (5.0,), 10), d.sha256: prediction(d, 0, (), 20),
                   e.sha256: {"schema_version": 1, "source_sha256": e.sha256, "fingerprint": FINGERPRINT,
                              "status": "failed", "failure_code": "MODEL_OOM", "latency_sec": 3.0}}
        result = aggregate([a, b, c, d, e, f], records, fingerprint=FINGERPRINT)
        verdict = result["verdict"]
        self.assertEqual([verdict[key] for key in ("tp", "fp", "tn", "fn")], [1, 1, 1, 1])
        for name in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "false_positive_rate_clean"):
            self.assertEqual(verdict[name], 0.5)
        self.assertAlmostEqual(verdict["selected_accuracy_lower_bound"], 1 / 3)
        self.assertAlmostEqual(verdict["positive_recall_with_failures_as_misses"], 1 / 3)
        self.assertEqual(result["counts"]["failed_unique_sources"], 1)
        self.assertEqual(result["counts"]["missing_unique_sources"], 1)
        self.assertEqual(result["counts"]["completed_sources_marked_analysis_ready"], 0)
        self.assertEqual(result["analysis_mode_counts"], {"diagnostic": 4})
        self.assertEqual(result["row_level_diagnostic"]["verdict"]["evaluated"], 6)
        self.assertEqual(result["timing_error_sec"]["count"], 0)
        self.assertEqual(result["timing_error_sec"]["excluded_detected_ambiguous_positive_sources"], 1)
        self.assertEqual(result["row_level_diagnostic"]["timing_error_sec"]["count"], 2)
        self.assertEqual(result["latency_sec"]["median"], 6)
        self.assertAlmostEqual(result["latency_sec"]["p90"], 17)
        self.assertEqual(result["classification"]["accuracy"], None)

    def test_timing_uses_nearest_actual_finding_only_on_true_positives(self) -> None:
        a, b, c = source("a", 1, (10.0,)), source("b", 1, (40.0,)), source("c", 0, (None,))
        records = {a.sha256: prediction(a, 1, (2.0, 11.5, 19.0)), b.sha256: prediction(b, 0), c.sha256: prediction(c, 1, (11.0,))}
        result = aggregate([a, b, c], records)
        self.assertEqual(result["timing_error_sec"]["count"], 1)
        self.assertEqual(result["timing_error_sec"]["median"], 1.5)
        self.assertEqual(result["timing_error_sec"]["missed_or_failed_positive_sources"], 1)

    def test_detector_only_does_not_publish_event_quality(self) -> None:
        a = source("a", 1, (10.0,))
        result = aggregate([a], {a.sha256: prediction(a, 0)}, detector_only=True)
        self.assertIsNone(result["verdict"])
        self.assertIsNone(result["timing_error_sec"])
        self.assertIsNone(result["row_level_diagnostic"])
        self.assertEqual(result["counts"]["completed_unique_sources"], 1)

    def test_invalid_or_nonfinal_predictions_cannot_enter_metrics(self) -> None:
        a = source("a", 1, (10.0,))
        variants = [prediction(a, 1), prediction(a, 1, (101.0,)), prediction(a, 0, (), float("nan"))]
        for value in variants:
            with self.assertRaises(ValueError):
                validate_prediction(value, a)
        value = prediction(a, 0)
        with self.assertRaises(ValueError):
            validate_prediction(value, a, fingerprint="0" * 64)
        with self.assertRaises(ValueError):
            aggregate([a, a], {})
        failure = {**value, "status": "failed", "failure_code": "Private model description must not be exported"}
        with self.assertRaises(ValueError):
            validate_prediction(failure, a)

    def test_empty_denominators_and_percentile_are_explicit(self) -> None:
        result = aggregate([], {})
        self.assertIsNone(result["verdict"]["accuracy"])
        self.assertIsNone(result["verdict"]["balanced_accuracy"])
        self.assertIsNone(result["latency_sec"]["p90"])
        self.assertEqual(percentile([5.0], 0.9), 5.0)

    def test_shareable_report_cannot_include_private_model_text_or_clip_ids(self) -> None:
        a = source("a", 1, (10.0,))
        record = prediction(a, 1, (11.0,))
        record["description"] = "PRIVATE_MODEL_DESCRIPTION"
        report = {"metrics": aggregate([a], {a.sha256: record}), "fingerprint": FINGERPRINT,
                  "pipeline_version": "test-only", "split": "development", "runtime": {"device": "test"},
                  "prepare_sec": 1, "package_versions": {}, "created_at": "2026-10-17"}
        output = markdown_report(report)
        self.assertNotIn("PRIVATE_MODEL_DESCRIPTION", output)
        self.assertNotIn("private-a-0", output)
        self.assertNotIn(a.sha256, output)


class ActualDatasetPartitionTests(unittest.TestCase):
    def test_fixed_partition_deduplicates_and_reserves_fold_zero(self) -> None:
        sources = load_sources(ROOT / "research/dataset/inventory.csv", ROOT / "eval/labels.csv", ROOT / "eval/folds.csv")
        self.assertEqual(len(sources), 40)
        self.assertEqual(len(select_sources(sources, "development")), 32)
        self.assertEqual(len(select_sources(sources, "holdout")), 8)
        ambiguous = [item for item in sources if item.timing_ambiguous]
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous[0].label_secs, (57.0, 68.0))
        self.assertEqual(ambiguous[0].fold, 0)
        self.assertEqual(len(select_sources(sources, "holdout", ["Х-4", "Х-47"])), 1)
        with self.assertRaises(ValueError):
            select_sources(sources, "development", ["Х-4"])
        with self.assertRaises(ValueError):
            select_sources(sources, "development", ["Х-1"])


if __name__ == "__main__":
    unittest.main()
