from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "refine_qwen_onset_probe", ROOT / "training/refine_qwen_onset_probe.py")
refine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = refine
SPEC.loader.exec_module(refine)


class RefineQwenOnsetProbeTests(unittest.TestCase):
    def test_variant_selection_prefers_balanced_accuracy_and_a_on_tie(self):
        a = {"variant": "delta_only_linear", "feasible": True,
             "oof_metrics": {"balanced_accuracy": 0.75, "specificity": 0.75, "recall": 0.75}}
        b = {"variant": "full_transition_linear", "feasible": True,
             "oof_metrics": {"balanced_accuracy": 0.75, "specificity": 0.75, "recall": 0.75}}
        self.assertEqual(refine.choose_variant([a, b])["variant"], "delta_only_linear")
        b["oof_metrics"]["balanced_accuracy"] = 0.8
        self.assertEqual(refine.choose_variant([a, b])["variant"], "full_transition_linear")
        self.assertIsNone(refine.choose_variant([{**a, "feasible": False}]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch runs only in Modal")
    def test_delta_variant_excludes_absolute_pre_post_blocks(self):
        import torch

        pairs = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        delta = refine.variant_features(pairs, "delta_only_linear")
        full = refine.variant_features(pairs, "full_transition_linear")
        self.assertTrue(torch.equal(delta, pairs[:, 6:]))
        self.assertTrue(torch.equal(full, pairs))


if __name__ == "__main__":
    unittest.main()
