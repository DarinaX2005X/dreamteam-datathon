from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "videoprism_probe", ROOT / "training/videoprism_probe.py")
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class VideoPrismProbeTests(unittest.TestCase):
    def test_coverage_includes_tail_without_labels(self):
        self.assertEqual(probe.coverage_starts(7.5), [0.0])
        self.assertEqual(probe.coverage_starts(18.0), [0.0, 4.0, 8.0, 10.5])

    def test_onset_is_fit_only_centered_and_clamped(self):
        self.assertEqual(probe.onset_start(20.0, 1.0), 0.0)
        self.assertEqual(probe.onset_start(20.0, 19.0), 12.5)
        self.assertAlmostEqual(probe.onset_start(20.0, 10.0), 6.25)

    def test_clip_pts_are_strict_and_complete(self):
        frames = [(index / 2, index) for index in range(40)]
        clip = probe.select_clip(frames, 4.1)
        self.assertEqual(len(clip), 16)
        self.assertEqual(clip[0][0], 4.0)
        with self.assertRaisesRegex(ValueError, "increase"):
            probe.select_clip([(0.0, 0)] * 16, 0.0)

    def test_calibration_guard(self):
        rows = []
        for index in range(12):
            rows.extend([{"sample_id": f"p{index}", "truth": True, "score": 0.8},
                         {"sample_id": f"n{index}", "truth": False, "score": 0.2}])
        _, metrics = probe.calibrate(rows)
        self.assertEqual((metrics["tp"], metrics["tn"]), (12, 12))

    def test_letterbox_preserves_wide_frame_edges(self):
        frame = np.zeros((100, 400, 3), dtype=np.uint8)
        frame[:, :4] = 255
        frame[:, -4:] = 127
        result = probe.letterbox_rgb(frame)
        self.assertEqual(result.shape, (288, 288, 3))
        content = result[108:180]
        self.assertGreater(float(content[:, 0].mean()), 0.9)
        self.assertGreater(float(content[:, -1].mean()), 0.45)


if __name__ == "__main__":
    unittest.main()
