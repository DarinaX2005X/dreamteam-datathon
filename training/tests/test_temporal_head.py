from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("temporal_head", ROOT / "training" / "temporal_head.py")
temporal_head = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = temporal_head
SPEC.loader.exec_module(temporal_head)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "train_temporal", ROOT / "training" / "train_temporal.py")
train_temporal = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
sys.modules[RUNNER_SPEC.name] = train_temporal
RUNNER_SPEC.loader.exec_module(train_temporal)


class TemporalHeadTests(unittest.TestCase):
    def measurement(self, correctness, errors=None, truths=None, scores=None):
        errors = errors or [None] * len(correctness)
        truths = truths or [True] * len(correctness)
        scores = scores or [None] * len(correctness)
        return {"invalid_predictions": 0,
                "per_source": [{"sample_id": str(i), "correct": correct,
                                "truth": truths[i], "score": scores[i],
                                "localization_error_sec": errors[i]}
                               for i, correct in enumerate(correctness)]}

    def test_paired_selection_rejects_trade_and_requires_half_second_timing_gain(self):
        baseline = self.measurement([True, False], [5.0, None])
        self.assertFalse(temporal_head.paired_improves(
            self.measurement([False, True], [1.0, None]), baseline))
        self.assertTrue(temporal_head.paired_improves(
            self.measurement([True, True], [5.0, None]), baseline))
        self.assertFalse(temporal_head.paired_improves(
            self.measurement([True, False], [4.51, None]), baseline))
        self.assertTrue(temporal_head.paired_improves(
            self.measurement([True, False], [4.5, None]), baseline))

    def test_duplicate_or_mismatched_sources_are_rejected(self):
        baseline = self.measurement([True])
        duplicate = {"per_source": [{"sample_id": "0", "correct": True,
                                      "truth": True, "score": 0.8,
                                      "localization_error_sec": None}] * 2}
        self.assertFalse(temporal_head.paired_improves(duplicate, baseline))
        self.assertFalse(temporal_head.paired_improves(baseline, duplicate))
        self.assertFalse(temporal_head.paired_improves(self.measurement([True, True]), baseline))

    def test_mean_timing_gain_and_invalid_values(self):
        baseline = self.measurement([True, True], [1.0, 9.0])
        sum_only_gain = self.measurement([True, True], [0.75, 8.75])
        mean_gain = self.measurement([True, True], [0.5, 8.5])
        self.assertFalse(temporal_head.paired_improves(sum_only_gain, baseline))
        self.assertTrue(temporal_head.paired_improves(mean_gain, baseline))
        invalid = self.measurement([True, True], [0.5, 8.5])
        invalid["invalid_predictions"] = 1
        self.assertFalse(temporal_head.paired_improves(invalid, baseline))
        self.assertFalse(temporal_head.paired_improves(
            self.measurement([True, True], [math.nan, 8.0]), baseline))
        self.assertFalse(temporal_head.paired_improves(
            self.measurement([True, True], [0.5, 8.5], scores=[math.inf, 0.5]), baseline))
        self.assertFalse(temporal_head.paired_improves(
            self.measurement([True, True], [0.5, 8.5], truths=[False, True]), baseline))

    def test_existing_output_is_not_claimed_or_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "sentinel.json"
            sentinel.write_text('{"state":"running"}\n', encoding="utf-8")
            with self.assertRaises(train_temporal.TemporalTrainingError):
                train_temporal.initialize_progress(output, 90)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"state":"running"}\n')

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is optional for metadata tests")
    def test_head_shapes_loss_and_positive_localization(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(1729)
        head = temporal_head.build_temporal_head()
        values = torch.randn(9, 256)
        times = torch.arange(9, dtype=torch.float32) / 2
        logits = head(values)
        self.assertEqual(tuple(logits.shape), (9,))
        loss, parts = temporal_head.mil_loss(logits, True, times, 2.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["localization"], 0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))
        head.zero_grad(set_to_none=True)
        negative_loss, negative_parts = temporal_head.mil_loss(head(values), False, times, None)
        self.assertEqual(negative_parts["localization"], 0.0)
        negative_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))


if __name__ == "__main__":
    unittest.main()
