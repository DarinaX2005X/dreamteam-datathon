from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.dataset import dataset_fingerprint, load_sources
from eval.test_metrics import source
from eval.training import build_training_manifest, load_training_partition, targets_for

ROOT = Path(__file__).resolve().parents[1]
TABLES = [ROOT / "research/dataset/inventory.csv", ROOT / "eval/labels.csv", ROOT / "eval/folds.csv"]


class TrainingDataTests(unittest.TestCase):
    def test_duplicate_conflicting_times_keep_binary_but_never_create_two_events(self) -> None:
        targets = targets_for(source("a", 1, (57.0, 68.0)))
        self.assertEqual(targets, {"has_violation": True, "time_sec": None, "time_supervision": False})
        self.assertEqual(targets_for(source("b", 0, (None,))),
                         {"has_violation": False, "time_sec": None, "time_supervision": False})

    def test_actual_inner_split_is_disjoint_and_omits_all_holdout_examples(self) -> None:
        sources = load_sources(*TABLES)
        manifest = build_training_manifest(sources, dataset_fingerprint(TABLES))
        samples = manifest["samples"]
        train = {sample["source_sha256"] for sample in samples if sample["partition"] == "train"}
        validation = {sample["source_sha256"] for sample in samples if sample["partition"] == "validation"}
        holdout = {item.sha256 for item in sources if item.fold == 0}
        self.assertEqual((len(train), len(validation), len(holdout)), (24, 8, 8))
        self.assertFalse(train & validation or (train | validation) & holdout)
        self.assertEqual(len(samples), len(train | validation))
        self.assertEqual(manifest["counts"]["train"]["positive_sources"], 12)
        self.assertEqual(manifest["counts"]["train"]["timestamp_supervised_sources"], 12)
        self.assertEqual(manifest["counts"]["validation"]["positive_sources"], 4)
        self.assertEqual(manifest["counts"]["reserved_final_holdout"]["ambiguous_timing_sources"], 1)
        self.assertTrue(all(set(sample["targets"]) == {"has_violation", "time_sec", "time_supervision"} for sample in samples))

    def test_reader_rejects_holdout_and_tampered_supervision_before_opening_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            args = {"manifest_path": manifest_path, "inventory": TABLES[0], "labels": TABLES[1],
                    "folds": TABLES[2], "dataset_dir": Path(directory), "verify_hashes": False}
            with self.assertRaisesRegex(ValueError, "never holdout"):
                load_training_partition(**args, partition="holdout")
            manifest = build_training_manifest(load_sources(*TABLES), dataset_fingerprint(TABLES))
            manifest["samples"][0]["targets"]["has_violation"] = not manifest["samples"][0]["targets"]["has_violation"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the fixed organizer tables"):
                load_training_partition(**args, partition="train")


if __name__ == "__main__":
    unittest.main()
