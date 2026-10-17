from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_proposal", ROOT / "training" / "train_proposal.py"
)
train_proposal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_proposal
SPEC.loader.exec_module(train_proposal)


class TrainingRunnerTests(unittest.TestCase):
    def test_fixed_recipe_rejects_unapproved_epoch_or_lora_shape(self) -> None:
        train_proposal.Recipe().validate()
        with self.assertRaisesRegex(train_proposal.TrainingError, "epochs"):
            train_proposal.Recipe(epochs=4).validate()
        with self.assertRaisesRegex(train_proposal.TrainingError, "rank 8"):
            train_proposal.Recipe(rank=16).validate()
        with self.assertRaisesRegex(train_proposal.TrainingError, "16 frames"):
            train_proposal.Recipe(frame_count=8).validate()

    def test_binary_metrics_are_balanced_and_label_blind_aggregation_is_earliest(self) -> None:
        metrics = train_proposal.binary_metrics(
            [True, True, False, False], [True, False, True, False]
        )
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        proposals = [
            SimpleNamespace(candidate=False, time_sec=None),
            SimpleNamespace(candidate=True, time_sec=9.5),
            SimpleNamespace(candidate=True, time_sec=4.0),
        ]
        self.assertEqual(train_proposal.choose_video_proposal(proposals), (True, 4.0))

    def test_host_guard_rejects_local_and_non_l40s_gpu(self) -> None:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="RTX 2050", total_memory=4 * 1024**3),
        )
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="11.8"))
        with (
            mock.patch.object(train_proposal.platform, "system", return_value="Windows"),
            self.assertRaisesRegex(train_proposal.TrainingError, "Linux"),
        ):
            train_proposal.enforce_training_host(fake_torch)
        with (
            mock.patch.object(train_proposal.platform, "system", return_value="Linux"),
            self.assertRaisesRegex(train_proposal.TrainingError, "L40S"),
        ):
            train_proposal.enforce_training_host(fake_torch)

    def test_host_guard_accepts_authorized_l40s_class(self) -> None:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(
                name="NVIDIA L40S", total_memory=48 * 1024**3
            ),
        )
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.4"))
        with mock.patch.object(train_proposal.platform, "system", return_value="Linux"):
            result = train_proposal.enforce_training_host(fake_torch)
        self.assertEqual(result["gpu_name"], "NVIDIA L40S")
        self.assertGreaterEqual(result["gpu_memory_bytes"], train_proposal.MIN_CUDA_MEMORY_BYTES)

    def test_artifact_hashes_are_deterministic_and_include_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
            first = train_proposal.artifact_hashes(root)
            second = train_proposal.artifact_hashes(root)
        self.assertEqual(first, second)
        self.assertEqual([item["path"] for item in first], ["adapter_config.json", "adapter_model.safetensors"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in first))

    def test_cli_requires_explicit_dataset_base_model_and_output(self) -> None:
        with self.assertRaises(SystemExit):
            train_proposal.parse_args([])
        args = train_proposal.parse_args(
            [
                "--base-model",
                "models/internvl3_5-4b",
                "--dataset-dir",
                "dataset",
                "--output-dir",
                "runs/lora",
            ]
        )
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.seed, 1729)
        self.assertEqual(args.cuda_memory_fraction, 0.70)
        self.assertEqual(args.max_wall_minutes, 165)

    def test_dependency_versions_are_pinned(self) -> None:
        versions = {
            "torch": "2.6.0+cu124",
            **train_proposal.REQUIRED_DEPENDENCIES,
        }
        train_proposal.validate_versions(versions)
        with self.assertRaisesRegex(train_proposal.TrainingError, "transformers 4.57.6"):
            train_proposal.validate_versions({**versions, "transformers": "4.58.0"})

    def test_progress_reporter_atomically_writes_sanitized_numeric_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = train_proposal.ProgressReporter(root)
            reporter.begin("training")
            reporter.write(
                epoch=1,
                completed=2,
                total=5,
                mean_loss=1.25,
                latest_loss=1.0,
            )
            current = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            history = (root / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(current["stage"], "training")
        self.assertEqual((current["completed"], current["total"]), (2, 5))
        self.assertGreaterEqual(current["stage_eta_seconds"], 0)
        self.assertEqual(len(history), 1)
        self.assertNotIn("sample", current)

    def test_failure_status_records_only_exception_type_and_preserves_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = train_proposal.ProgressReporter(root)
            reporter.write(completed=0, total=1)
            train_proposal.mark_failed_status(root, RuntimeError("private/path/video.mp4"))
            failed = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["failure_type"], "RuntimeError")
            self.assertNotIn("private", json.dumps(failed))
            failed["state"] = "interrupted"
            train_proposal.write_json_atomic(root / "progress.json", failed)
            train_proposal.mark_failed_status(root, ValueError("must remain interrupted"))
            preserved = json.loads((root / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(preserved["state"], "interrupted")

    def test_startup_failure_is_durable_but_existing_run_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new-run"
            argv = [
                "--base-model",
                str(root / "models" / "internvl3_5-4b"),
                "--dataset-dir",
                str(root / "dataset"),
                "--output-dir",
                str(output),
            ]
            with (
                mock.patch.object(
                    train_proposal, "package_versions", side_effect=RuntimeError("startup")
                ),
                self.assertRaisesRegex(RuntimeError, "startup"),
            ):
                train_proposal.main(argv)
            failed = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["failure_type"], "RuntimeError")

            existing = root / "existing-run"
            existing.mkdir()
            sentinel = '{"state":"running","sentinel":true}\n'
            (existing / "progress.json").write_text(sentinel, encoding="utf-8")
            argv[-1] = str(existing)
            with self.assertRaisesRegex(train_proposal.TrainingError, "must be empty"):
                train_proposal.main(argv)
            self.assertEqual(
                (existing / "progress.json").read_text(encoding="utf-8"), sentinel
            )

    def test_partial_accumulation_restores_mean_gradient_scale(self) -> None:
        class FakeGradient:
            def __init__(self, value: float) -> None:
                self.value = value

            def mul_(self, factor: float) -> None:
                self.value *= factor

        gradient = FakeGradient(0.25)
        model = SimpleNamespace(parameters=lambda: [SimpleNamespace(grad=gradient)])
        train_proposal.rescale_partial_gradients(model, microbatch_count=2, target_count=8)
        self.assertEqual(gradient.value, 1.0)

    def test_checkpoint_selection_rejects_traded_errors_and_unparseable_output(self) -> None:
        def measurement(correct, errors=(None, None)):
            return {"invalid_output_count": 0, "no_valid_source_output_count": 0,
                    "per_source": [{"sample_id": str(index), "correct": value,
                                    "localization_error_sec": errors[index]}
                                   for index, value in enumerate(correct)]}

        baseline = measurement([True, False])
        self.assertFalse(train_proposal.checkpoint_improves(measurement([False, True]), baseline))
        self.assertFalse(train_proposal.checkpoint_improves(measurement([True, False]), baseline))
        improved = measurement([True, True])
        self.assertTrue(train_proposal.checkpoint_improves(improved, baseline))
        self.assertFalse(train_proposal.checkpoint_improves(
            {**improved, "invalid_output_count": 1}, baseline))
        self.assertFalse(train_proposal.checkpoint_improves({"per_source": []}, baseline))

    def test_time_improvement_requires_the_same_paired_sources(self) -> None:
        baseline = {"per_source": [{"sample_id": "a", "correct": True,
                                    "localization_error_sec": 5.0}]}
        improved = {"per_source": [{"sample_id": "a", "correct": True,
                                    "localization_error_sec": 3.0}]}
        different = {"per_source": [{"sample_id": "b", "correct": True,
                                     "localization_error_sec": 1.0}]}
        self.assertTrue(train_proposal.checkpoint_improves(improved, baseline))
        self.assertFalse(train_proposal.checkpoint_improves(different, baseline))
        negligible = {"per_source": [{"sample_id": "a", "correct": True,
                                      "localization_error_sec": 4.99}]}
        self.assertFalse(train_proposal.checkpoint_improves(negligible, baseline))

    def test_budget_preserves_checkpoint_time_and_is_capped_at_three_hours(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(train_proposal.time, "time", return_value=100):
                reporter = train_proposal.ProgressReporter(Path(directory), max_wall_minutes=3)
                self.assertFalse(reporter.should_stop())
            with mock.patch.object(train_proposal.time, "time", return_value=221):
                self.assertTrue(reporter.should_stop())
            with self.assertRaisesRegex(train_proposal.TrainingError, "180"):
                train_proposal.ProgressReporter(Path(directory), max_wall_minutes=181)


if __name__ == "__main__":
    unittest.main()
