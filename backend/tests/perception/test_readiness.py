"""An asset existing on disk is not evidence that inference is ready."""

import hashlib
import json
import sys
from types import SimpleNamespace

from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.runtime import LocalPerceptionRuntime


def _config(tmp_path, monkeypatch, *, semantic=False):
    monkeypatch.setenv("NAVIDU_REQUIRE_SEMANTIC", "1" if semantic else "0")
    monkeypatch.setenv("NAVIDU_DETECTOR_MODEL", "test-detector")
    monkeypatch.setenv("NAVIDU_SEMANTIC_MODEL", "test-semantic")
    models = []
    for name in ("test-detector", "test-semantic"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(b"test fixture")
        models.append({"id": name, "files": [{"path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]})
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": models}), encoding="utf-8")
    return PipelineConfig(check_id="0123456789ab", models_dir=tmp_path, device="cpu")


def test_assets_present_does_not_report_ready(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    assert LocalPerceptionRuntime().runtime_status(config).model_status == "loading"


def test_successfully_loaded_detector_is_ready_and_reused(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    loads = []
    def detector(directory, entry, device):
        loads.append(entry["id"])
        return SimpleNamespace(device=device)
    monkeypatch.setitem(sys.modules, "navidu.perception.detector", SimpleNamespace(RfDetrOnnx=detector))
    runtime = LocalPerceptionRuntime()
    runtime.prepare(config)
    runtime.prepare(config)
    status = runtime.runtime_status(config)
    assert status.model_status == "ready"
    assert status.analysis_ready is False
    assert status.analysis_mode == "diagnostic"
    assert status.pipeline_version.startswith("0.1.0+runtime.")
    assert loads == ["test-detector"]
    rebound = config.model_copy(update={"pipeline_version": status.pipeline_version})
    runtime.prepare(rebound)
    assert runtime.runtime_status(rebound).pipeline_version == status.pipeline_version
    assert loads == ["test-detector"]
    changed = config.model_copy(update={"device": "cuda"})
    assert runtime.runtime_status(changed).model_status == "loading"


def test_required_semantic_must_be_loaded(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, semantic=True)
    runtime = LocalPerceptionRuntime()
    runtime._prepared_key = runtime._configuration_key(config)
    runtime._pipeline_version = "test+runtime.loaded"
    runtime._detector = SimpleNamespace(device="cpu")
    assert runtime.runtime_status(config).model_status == "loading"
    runtime._semantic = SimpleNamespace(device="cpu")
    status = runtime.runtime_status(config)
    assert status.model_status == "ready"
    assert status.analysis_ready is False
    assert status.analysis_mode == "unvalidated"


def test_public_readiness_requires_every_enabled_core_semantic_capability(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, semantic=True)
    runtime = LocalPerceptionRuntime()
    runtime._prepared_key = runtime._configuration_key(config)
    runtime._pipeline_version = "test+runtime.loaded"
    runtime._detector = SimpleNamespace(device="cpu")
    runtime._semantic = SimpleNamespace(device="cpu")
    runtime._required_semantic_capabilities = {"counting_action", "handover_direction"}
    runtime._validated_capabilities = {"cash_detection": True, "counting_action": True}
    assert runtime.runtime_status(config).analysis_ready is False
    runtime._validated_capabilities["handover_direction"] = True
    assert runtime.runtime_status(config).analysis_ready is True
    diagnostic = config.model_copy(update={"diagnostic_only": True})
    runtime._prepared_key = runtime._configuration_key(diagnostic)
    assert runtime.runtime_status(diagnostic).analysis_ready is False
    assert runtime.runtime_status(diagnostic).analysis_mode == "diagnostic"


def test_missing_required_semantic_has_actionable_status(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, semantic=True)
    (tmp_path / "test-semantic.bin").unlink()
    status = LocalPerceptionRuntime().runtime_status(config)
    assert status.model_status == "missing"
    assert "test-semantic.bin" in status.limitations[0]


def test_bad_checksum_never_becomes_ready(tmp_path, monkeypatch):
    import pytest

    config = _config(tmp_path, monkeypatch)
    (tmp_path / "test-detector.bin").write_bytes(b"bad! fixture")
    runtime = LocalPerceptionRuntime()
    with pytest.raises(RuntimeError, match="MODEL_CHECKSUM_MISMATCH"):
        runtime.prepare(config)
    assert runtime.runtime_status(config).model_status == "error"
