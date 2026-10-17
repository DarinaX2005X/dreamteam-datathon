"""Model artifacts are explicit local files, never runtime download URLs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ModelAssetsError(RuntimeError):
    pass


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ModelAssetsError("MODEL_MISSING: отсутствует models/manifest.json. Выполните scripts/fetch_models.py.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
            raise ValueError("unsupported manifest schema")
    except (ValueError, TypeError) as error:
        raise ModelAssetsError("MODEL_MANIFEST_INVALID: некорректный манифест моделей.") from error
    return manifest


def artifact_path(directory: Path, relative: str) -> Path:
    root = directory.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ModelAssetsError("MODEL_MANIFEST_INVALID: путь весов выходит за каталог моделей.")
    return path


def model_entry(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in manifest["models"]:
        if model.get("id") == model_id:
            if not isinstance(model.get("files"), list) or not model["files"]:
                raise ModelAssetsError(f"MODEL_MANIFEST_INVALID: нет файлов модели {model_id}.")
            return model
    raise ModelAssetsError(f"MODEL_MISSING: модель {model_id} не подготовлена. Выполните scripts/fetch_models.py.")


def missing_files(directory: Path, entry: dict[str, Any]) -> list[str]:
    missing = []
    for item in entry["files"]:
        path = artifact_path(directory, item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            missing.append(item["path"])
    return missing


def verify_model(directory: Path, entry: dict[str, Any]) -> str:
    """Check content once on prepare; return a stable hash for run provenance."""
    aggregate = hashlib.sha256()
    for item in sorted(entry["files"], key=lambda value: value["path"]):
        path = artifact_path(directory, item["path"])
        if not path.is_file():
            raise ModelAssetsError(f"MODEL_MISSING: {item['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ModelAssetsError(f"MODEL_CHECKSUM_MISMATCH: повреждены веса {item['path']}.")
        aggregate.update(item["path"].encode("utf-8") + b"\0" + actual.encode("ascii"))
    return aggregate.hexdigest()

