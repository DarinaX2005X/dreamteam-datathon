import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load("lado_smoke_api", "scripts/smoke_api.py")
preflight = load("lado_preflight", "scripts/preflight.py")
run_local = load("lado_run_local", "scripts/run_local.py")


def jpeg_base64():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_validate_done_checks_timecode_and_real_jpeg():
    payload = {
        "job_id": "0123456789ab",
        "status": "done",
        "verdict": "violation",
        "violations": [
            {
                "type": "V-TEST",
                "timecode_sec": 1.25,
                "description": "Наблюдение и ссылка на норму",
                "evidence_frames": [
                    {"timestamp_sec": 1.25, "image_base64": jpeg_base64()}
                ],
            }
        ],
    }
    assert smoke.validate_done(payload) == ("violation", 1)
    payload["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(
        b"not jpeg"
    ).decode()
    with pytest.raises(smoke.SmokeFailure, match="JPEG"):
        smoke.validate_done(payload)


def test_manifest_files_accepts_versioned_nested_schema(tmp_path):
    manifest = {
        "schema_version": 1,
        "models": [{"files": [{"path": "detector/model.onnx", "sha256": "a" * 64}]}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    model_dir = tmp_path / "detector"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"model")
    loaded, _ = preflight.load_manifest(tmp_path)
    assert preflight.manifest_files(loaded) == [
        (Path("detector/model.onnx"), "a" * 64, None)
    ]


def test_processing_job_id_is_strict():
    assert (
        smoke.validate_processing({"job_id": "0123456789ab", "status": "processing"})
        == "0123456789ab"
    )
    with pytest.raises(smoke.SmokeFailure, match="job_id"):
        smoke.validate_processing({"job_id": "../../unsafe", "status": "processing"})


def test_dotenv_is_literal_and_existing_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("NAVIDU_PORT", "9000")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NAVIDU_PORT=8000\nNAVIDU_MODELS_DIR='C:\\models folder\\v1#fixed'\n",
        encoding="utf-8",
    )
    run_local.load_dotenv(env_file)
    assert os.environ["NAVIDU_PORT"] == "9000"
    assert os.environ["NAVIDU_MODELS_DIR"] == r"C:\models folder\v1#fixed"
