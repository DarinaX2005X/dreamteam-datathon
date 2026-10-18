from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen_onset_probe", ROOT / "training" / "qwen_onset_probe.py")
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class QwenOnsetProbeTests(unittest.TestCase):
    def test_threshold_is_train_only_and_rejects_all_positive(self):
        rows = []
        for index in range(12):
            rows.append({"sample_id": f"p{index}", "truth": True, "score": 0.8 + index / 1000})
            rows.append({"sample_id": f"n{index}", "truth": False, "score": 0.1 + index / 1000})
        result = probe.calibrate_threshold(rows)
        self.assertEqual(result.metrics["tp"], 12)
        self.assertEqual(result.metrics["tn"], 12)
        collapsed = [{**row, "score": 0.9} for row in rows]
        with self.assertRaisesRegex(ValueError, "anti-all-positive"):
            probe.calibrate_threshold(collapsed)

    def test_strict_gate_preserves_baseline_and_recovers_one_miss(self):
        base_pos = {"x15", "x18"}
        negatives = {"n1", "n2", "n3", "n4"}
        misses = {"x13", "x46"}
        rows = [{"sample_id": key, "candidate": key in base_pos | {"x13"}, "score": 0.8}
                for key in base_pos | negatives | misses]
        self.assertTrue(probe.strict_validation_gate(rows, base_pos, negatives, misses))
        broken = [{**row, "candidate": True} if row["sample_id"] == "n1" else row for row in rows]
        self.assertFalse(probe.strict_validation_gate(broken, base_pos, negatives, misses))
        invalid = [{**row, "score": math.nan} if row["sample_id"] == "x13" else row for row in rows]
        self.assertFalse(probe.strict_validation_gate(invalid, base_pos, negatives, misses))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch runs only in Modal checks")
    def test_normalized_bag_score_is_length_invariant_and_loss_has_gradients(self):
        import torch

        torch.set_num_threads(2)
        one = probe.normalized_log_mean_exp(torch.tensor([2.0]))
        many = probe.normalized_log_mean_exp(torch.full((17,), 2.0))
        self.assertTrue(torch.allclose(one, many))
        features = torch.randn(8, 6)
        times = torch.arange(8, dtype=torch.float32) / 2
        pairs, endpoints = probe.transition_pairs(features, times)
        head = probe.build_probe(6)
        logits = head(pairs).flatten()
        positive, _ = probe.source_loss(logits, endpoints, True, 2.5)
        negative, parts = probe.source_loss(logits, endpoints, False, None)
        loss = probe.balanced_source_loss([(True, positive), (False, negative)])
        self.assertEqual(parts["localization"], 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch runs only in Modal checks")
    def test_far_positive_peak_does_not_change_onset_support_loss(self):
        import torch

        times = torch.tensor([0.0, 1.0, 2.0, 3.0, 8.0])
        first = torch.tensor([-2.0, 0.0, 1.0, 0.0, -5.0])
        second = first.clone()
        second[-1] = 50.0
        _, first_parts = probe.source_loss(first, times, True, 2.0)
        _, second_parts = probe.source_loss(second, times, True, 2.0)
        self.assertAlmostEqual(first_parts["localization"], second_parts["localization"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch runs only in Modal checks")
    def test_qwen_nested_return_pools_only_final_embeddings(self):
        import torch

        final = (torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
                 torch.tensor([[5.0, 7.0]]))
        deepstack = ([torch.full((3, 2), 1000.0)],)
        pooled = probe.pool_qwen_image_features((final, deepstack))
        self.assertTrue(torch.equal(pooled, torch.tensor([3.0, 5.0])))
        with self.assertRaisesRegex(ValueError, "unexpected Qwen"):
            probe.pool_qwen_image_features(final)


if __name__ == "__main__":
    unittest.main()
