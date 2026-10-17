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
    assert runtime.runtime_status(config).model_status == "ready"
    assert loads == ["test-detector"]
    changed = config.model_copy(update={"device": "cuda"})
    assert runtime.runtime_status(changed).model_status == "loading"


def test_required_semantic_must_be_loaded(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, semantic=True)
    runtime = LocalPerceptionRuntime()
    runtime._prepared_key = runtime._configuration_key(config)
    runtime._detector = SimpleNamespace(device="cpu")
    assert runtime.runtime_status(config).model_status == "loading"
    runtime._semantic = object()
    assert runtime.runtime_status(config).model_status == "ready"


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
