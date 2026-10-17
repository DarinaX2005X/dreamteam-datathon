"""A selected adapter can be registered without touching its base or readiness."""

import hashlib
import json

import pytest
from navidu.perception.assets import ModelAssetsError, verify_model
from navidu.perception.package_adapter import package_adapter
from navidu.perception.proposal_task import LORA_TARGET_MODULES, PROMPT_VERSION


def _file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _run(tmp_path):
    models = tmp_path / "models"
    base = {"id": "internvl3_5-4b", "repo": "public/base", "revision": "pinned",
        "files": [{"path": "base.weights", **_file(models / "base.weights", b"unchanged-base")}]}
    (models / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": [base]}), encoding="utf-8")
    (models / "validation.json").write_bytes(b"existing-validation-must-stay")
    run_dir = tmp_path / "run"
    config = {"peft_type": "LORA", "target_modules": LORA_TARGET_MODULES, "bias": "none",
              "r": 8, "lora_alpha": 16, "lora_dropout": .05}
    artifacts = [{"path": name, **_file(run_dir / "epoch-1" / name, content)} for name, content in (
        ("adapter_config.json", json.dumps(config).encode("utf-8")),
        ("adapter_model.safetensors", b"synthetic-test-placeholder"))]
    run = {"state": "complete", "holdout_used": False, "folds": {"train": [1, 2, 3], "validation": [4]},
        "base_model_id": "public/base@pinned", "base_model_revision": "pinned",
        "base_model_aggregate_sha256": verify_model(models, base), "prompt_version": PROMPT_VERSION,
        "selected_epoch": 1, "dataset_fingerprint": "test-dataset",
        "epochs": [{"epoch": 1, "checkpoint": "epoch-1", "validation": {"balanced_accuracy": .75},
                    "changed_trainable_tensors": 2, "training": {"optimizer_steps": 18}, "artifacts": artifacts}]}
    (run_dir / "run-manifest.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir, models, run


def test_selected_artifact_is_registered_idempotently_without_changing_base_or_quality(tmp_path):
    run_dir, models, _ = _run(tmp_path)
    result = package_adapter(run_dir, models, "test-proposal")
    assert result == {"state": "packaged_unvalidated", "model_id": "test-proposal", "reused": False}
    assert (models / "base.weights").read_bytes() == b"unchanged-base"
    assert (models / "validation.json").read_bytes() == b"existing-validation-must-stay"
    entry = json.loads((models / "manifest.json").read_text("utf-8"))["models"][-1]
    assert entry["runtime"] == "peft-lora-proposal" and entry["task"]["fps"] == 2.
    assert package_adapter(run_dir, models, "test-proposal")["reused"] is True


@pytest.mark.parametrize("change", [
    {"state": "validation_pending"}, {"holdout_used": True}, {"selected_epoch": 0},
    {"folds": {"train": [0, 1, 2], "validation": [4]}}, {"base_model_revision": "other"},
])
def test_unfinished_or_different_training_cannot_be_registered(tmp_path, change):
    run_dir, models, run = _run(tmp_path)
    original = (models / "manifest.json").read_bytes()
    (run_dir / "run-manifest.json").write_text(json.dumps({**run, **change}), encoding="utf-8")
    with pytest.raises(ModelAssetsError):
        package_adapter(run_dir, models, "test-proposal")
    assert (models / "manifest.json").read_bytes() == original
    assert not (models / "test-proposal").exists()


def test_changed_checkpoint_is_not_silently_registered(tmp_path):
    run_dir, models, _ = _run(tmp_path)
    (run_dir / "epoch-1/adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(ModelAssetsError, match="изменился"):
        package_adapter(run_dir, models, "test-proposal")
    assert not (models / "test-proposal").exists()
