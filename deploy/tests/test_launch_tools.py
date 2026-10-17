import base64
import hashlib
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
fetch_models = load("lado_fetch_models", "scripts/fetch_models.py")


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


def test_preflight_requires_only_selected_non_rejected_model_files(
    tmp_path, monkeypatch
):
    content = b"selected model"
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "models": [
            {
                "id": "selected-detector",
                "files": [
                    {
                        "path": "selected/model.onnx",
                        "sha256": digest,
                        "size_bytes": len(content),
                    }
                ],
            },
            {
                "id": "unused-alternate",
                "files": [{"path": "unused/missing.onnx", "sha256": "0" * 64}],
            },
            {
                "id": "rejected-model",
                "status": "rejected",
                "files": [{"path": "rejected/missing.onnx", "sha256": "0" * 64}],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / "selected" / "model.onnx"
    target.parent.mkdir()
    target.write_bytes(content)
    monkeypatch.setenv("NAVIDU_DETECTOR_MODEL", "selected-detector")
    monkeypatch.setenv("NAVIDU_REQUIRE_SEMANTIC", "false")
    args = type(
        "Args",
        (),
        {
            "models_dir": str(tmp_path),
            "data_dir": str(tmp_path / "data"),
            "host_only": False,
            "min_ram_gib": 0,
            "min_disk_gib": 0,
            "min_vram_gib": 0,
            "device": "cpu",
            "require_models": True,
        },
    )()
    checks = preflight.run_checks(args)
    by_name = {item.name: item for item in checks}
    assert by_name["configured_models"].status == "ok"
    assert by_name["model_files"].status == "ok"
    assert by_name["model_files"].detail.startswith("1 listed")
    assert by_name["model_integrity"].status == "ok"


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


@pytest.mark.parametrize(
    ("memory_mib", "expected_device"), [(4096, "cpu"), (8192, "cuda"), (None, "cpu")]
)
def test_auto_device_runtime_and_download_profile_match(
    memory_mib, expected_device, monkeypatch
):
    monkeypatch.delenv("NAVIDU_DETECTOR_MODEL", raising=False)
    monkeypatch.setenv("NAVIDU_REQUIRE_SEMANTIC", "true")
    device = run_local.select_device("auto", memory_mib)
    assert device == expected_device
    selected = fetch_models.PROFILE_MODELS[device]
    assert run_local.default_detector(device) == selected[0]
    assert preflight.configured_model_ids(device) == list(selected)
