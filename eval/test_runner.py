from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.dataset import file_sha256
from eval.run_pipeline import (
    RunStore,
    _artifact_projection,
    atomic_json,
    exclusive_lock,
    freeze_profile,
    guard_holdout,
    hardware_fingerprint,
    hardware_snapshot,
    require_combined_report_scope,
    save_profile,
)
from eval.test_metrics import FINGERPRINT, prediction, source


class CacheTests(unittest.TestCase):
    def test_cpu_metadata_does_not_touch_cuda_and_onnx_metadata_needs_no_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}), patch("eval.run_pipeline.cuda_driver_metadata") as probe:
            cpu = hardware_snapshot("cpu")
            self.assertEqual(cpu["accelerators"], [])
            probe.assert_not_called()
            accelerator = {"type": "cuda", "index": 0, "name": "test-only-gpu",
                           "memory_bytes": 1024, "compute_capability": "0.0"}
            probe.return_value = (accelerator, 12000)
            onnx = hardware_snapshot("cuda")
            self.assertEqual(onnx["accelerators"], [accelerator])
            self.assertIsNone(onnx["cuda_runtime"])
            self.assertEqual(onnx["cuda_driver_api_version"], 12000)
            probe.assert_called_once_with()

    def test_cache_profile_rejects_different_or_unknown_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            hardware = {"cpu": "synthetic-cpu", "accelerators": [{"name": "synthetic-gpu"}]}
            profile = {"fingerprint": FINGERPRINT, "algorithm_fingerprint": "a" * 64,
                       "pipeline_version": "test-only", "hardware": hardware,
                       "hardware_fingerprint": hardware_fingerprint(hardware), "runtime_label": "test-only"}
            save_profile(path, profile)
            save_profile(path, {**profile, "prepare_sec": 3})
            for change in ({"hardware_fingerprint": "b" * 64}, {"runtime_label": "another-machine"}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "HARDWARE_PROFILE_MISMATCH"):
                    save_profile(path, {**profile, **change})
            atomic_json(path, {key: value for key, value in profile.items() if key != "hardware_fingerprint"})
            with self.assertRaisesRegex(ValueError, "HARDWARE_PROFILE_MISMATCH"):
                save_profile(path, profile)

    def test_evaluation_cannot_accept_publishable_artifacts(self) -> None:
        item = source("a", 0, (None,))
        for ready, mode in ((True, "diagnostic"), (False, "validated"), (True, "validated")):
            artifacts = SimpleNamespace(run=SimpleNamespace(
                pipeline_version="test-only", analysis_ready=ready, analysis_mode=mode))
            with self.subTest(ready=ready, mode=mode), self.assertRaisesRegex(ValueError, "EVALUATION_MODE_MISMATCH"):
                _artifact_projection(artifacts, item, fingerprint=FINGERPRINT,
                                     pipeline_version="test-only", latency=1, attempt=Path("unused"))

    def test_only_complete_hash_verified_artifacts_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory), FINGERPRINT)
            item = source("a", 0, (None,))
            attempt = store.new_attempt(item)
            self.assertIsNone(store.cached(item))
            payloads = {"run.json": {"pipeline_version": "test-only"}, "finding_set.json": {},
                        "engine_manifest.json": {"complete": True}, "video.json": {}}
            for filename, value in payloads.items():
                atomic_json(attempt / filename, value)
            record = prediction(item, 0)
            record["artifact_hashes"] = {filename: file_sha256(attempt / filename) for filename in payloads}
            store.save(item, attempt, record)
            self.assertEqual(store.cached(item)["status"], "done")
            atomic_json(attempt / "finding_set.json", {"changed": True})
            with self.assertRaises(ValueError):
                store.cached(item)

    def test_projection_verifies_pipeline_frames_and_rejects_missing_evidence(self) -> None:
        from PIL import Image

        item = source("a", 1, (20.0,))
        artifacts = SimpleNamespace(
            run=SimpleNamespace(pipeline_version="test-only", analysis_ready=False,
                analysis_mode="diagnostic", phase_timings_sec={}, model_versions={"test": "1"}),
            video=SimpleNamespace(duration_sec=100.0),
            finding_set=SimpleNamespace(verdict="violation", rules=[SimpleNamespace(status="violation")],
                findings=[SimpleNamespace(timecode_sec=20.0, rule_id="synthetic-test-rule",
                    evidence=[SimpleNamespace(frame_id="fr-20000")])]))
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            for filename in ("run.json", "finding_set.json", "video.json"):
                atomic_json(attempt / filename, {})
            atomic_json(attempt / "engine_manifest.json", {"complete": True})
            frame = attempt / "frames/original/fr-20000.jpg"
            frame.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(frame)
            record = _artifact_projection(artifacts, item, fingerprint=FINGERPRINT,
                pipeline_version="test-only", latency=1.0, attempt=attempt)
            self.assertEqual(record["status"], "done")
            self.assertEqual(record["artifact_hashes"]["frames/original/fr-20000.jpg"], file_sha256(frame))
            frame.unlink()
            with self.assertRaises(FileNotFoundError):
                _artifact_projection(artifacts, item, fingerprint=FINGERPRINT,
                    pipeline_version="test-only", latency=1.0, attempt=attempt)

    def test_failed_and_partial_attempts_never_become_completed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory), FINGERPRINT)
            item = source("a", 1, (20.0,))
            first = store.new_attempt(item)
            (first / "partial.txt").write_text("partial", encoding="utf-8")
            self.assertIsNone(store.cached(item))
            failure = {"schema_version": 1, "source_sha256": item.sha256, "fingerprint": FINGERPRINT,
                       "status": "failed", "failure_code": "TEST_FAILURE", "latency_sec": 2}
            store.save(item, first, failure)
            self.assertIsNone(store.cached(item))
            self.assertEqual(store.latest(item)["failure_code"], "TEST_FAILURE")
            second = store.new_attempt(item)
            self.assertNotEqual(first, second)
            self.assertTrue((first / "partial.txt").is_file())
            wrong = {**failure, "fingerprint": "0" * 64}
            with self.assertRaises(ValueError):
                store.save(item, second, wrong)

    def test_os_lock_prevents_concurrent_runs_and_releases_after_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_lock(root):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(root):
                        self.fail("Concurrent lock acquired")
            with exclusive_lock(root):
                pass


class HoldoutTests(unittest.TestCase):
    def test_combined_report_cannot_initialize_or_change_holdout_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = {"fingerprint": FINGERPRINT, "algorithm_fingerprint": "a" * 64,
                       "pipeline_version": "test-only", "detector_only": False}
            holdout = source("a", 1, (10.0,), fold=0)
            sources = [holdout, source("b", 0, (None,), fold=1)]
            with self.assertRaisesRegex(ValueError, "HOLDOUT_NOT_EVALUATED"):
                require_combined_report_scope(root, profile, "dataset-v1", sources)
            self.assertFalse((root / "holdout-session.json").exists())
            freeze_profile(root, profile, "dataset-v1")
            guard_holdout(root, profile, "dataset-v1", [holdout])
            require_combined_report_scope(root, profile, "dataset-v1", sources)
            with self.assertRaisesRegex(ValueError, "HOLDOUT_SCOPE_MISMATCH"):
                require_combined_report_scope(root, profile, "dataset-v1", [*sources, source("c", 0, (None,), fold=0)])
            with self.assertRaisesRegex(ValueError, "HOLDOUT_PROFILE_MISMATCH"):
                require_combined_report_scope(root, profile, "dataset-v2", sources)

    def test_holdout_requires_single_frozen_profile_and_allows_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = {"fingerprint": FINGERPRINT, "algorithm_fingerprint": "a" * 64,
                       "pipeline_version": "test-only", "detector_only": False}
            selected = [source("a", 1, (10.0,), fold=0)]
            with self.assertRaises(ValueError):
                guard_holdout(root, profile, "dataset-v1", selected)
            first = freeze_profile(root, profile, "dataset-v1")
            self.assertEqual(first, freeze_profile(root, profile, "dataset-v1"))
            guard_holdout(root, profile, "dataset-v1", selected)
            guard_holdout(root, profile, "dataset-v1", selected)
            with self.assertRaises(ValueError):
                guard_holdout(root, profile, "dataset-v2", selected)
            with self.assertRaises(ValueError):
                guard_holdout(root, {**profile, "fingerprint": "0" * 64}, "dataset-v1", selected)
            with self.assertRaises(ValueError):
                guard_holdout(root, {**profile, "algorithm_fingerprint": "0" * 64}, "dataset-v1", selected)
            with self.assertRaises(ValueError):
                guard_holdout(root, {**profile, "hardware_fingerprint": "0" * 64}, "dataset-v1", selected)
            with self.assertRaises(ValueError):
                freeze_profile(root, {**profile, "fingerprint": "0" * 64}, "dataset-v1")
            with self.assertRaises(ValueError):
                guard_holdout(root, profile, "dataset-v1", [])

    def test_detector_only_profile_cannot_be_frozen_for_quality_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_profile(Path(directory), {"fingerprint": FINGERPRINT, "pipeline_version": "test", "detector_only": True}, "dataset-v1")


if __name__ == "__main__":
    unittest.main()
