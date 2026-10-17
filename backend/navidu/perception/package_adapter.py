"""Register a completed, selected local adapter; this never enables readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from .assets import ModelAssetsError, artifact_path, model_entry, read_manifest, verify_model
from .proposal import TASK, validate_adapter
from .proposal_task import PROMPT_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_adapter(run_dir: Path, models_dir: Path, model_id: str) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", model_id):
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: неверный идентификатор адаптера.")
    run_path = run_dir / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("state") != "complete" or run.get("holdout_used") is not False \
            or run.get("folds") != {"train": [1, 2, 3], "validation": [4]} \
            or run.get("prompt_version") != PROMPT_VERSION:
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: нужен завершённый запуск с внутренней проверкой без holdout.")
    selected = run.get("selected_epoch")
    epochs = run.get("epochs", [])
    if type(selected) is not int or not 1 <= selected <= 3 or not isinstance(epochs, list):
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: не выбран проверенный checkpoint.")
    candidates = [item for item in epochs if isinstance(item, dict) and item.get("epoch") == selected]
    if len(candidates) != 1:
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: неоднозначный checkpoint.")
    epoch = candidates[0]
    if not isinstance(epoch.get("validation"), dict) \
            or type(epoch.get("changed_trainable_tensors")) is not int \
            or epoch["changed_trainable_tensors"] < 1 \
            or not isinstance(epoch.get("training"), dict) \
            or type(epoch["training"].get("optimizer_steps")) is not int \
            or epoch["training"]["optimizer_steps"] < 1:
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: нет подтверждения обучения и измерения checkpoint.")
    manifest = read_manifest(models_dir)
    base_entry = model_entry(manifest, "internvl3_5-4b")
    base_hash = verify_model(models_dir, base_entry)
    if run.get("base_model_revision") != base_entry.get("revision") \
            or run.get("base_model_aggregate_sha256") != base_hash \
            or run.get("base_model_id") != f"{base_entry['repo']}@{base_entry['revision']}":
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: обучающий запуск использовал другую базовую модель.")
    checkpoint = artifact_path(run_dir, epoch["checkpoint"])
    source_files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = artifact_path(checkpoint, name)
        claims = [item for item in epoch.get("artifacts", []) if item.get("path") == name]
        if len(claims) != 1 or not source.is_file() \
                or source.stat().st_size != claims[0].get("size_bytes") \
                or _sha256(source) != claims[0].get("sha256"):
            raise ModelAssetsError("MODEL_ADAPTER_INVALID: checkpoint изменился после измерения.")
        source_files[name] = (source, claims[0])
    run_hash = _sha256(run_path)
    entry = {"id": model_id, "name": "Supervised cash episode proposal LoRA",
        "runtime": "peft-lora-proposal", "local_dir": model_id, "revision": run_hash,
        "base_model": {"id": base_entry["id"], "revision": base_entry.get("revision"), "sha256": base_hash},
        "task": dict(TASK),
        "training": {"run_manifest_sha256": run_hash, "selected_epoch": selected,
                     "dataset_fingerprint": run.get("dataset_fingerprint"), "holdout_used": False},
        "files": [{"path": f"{model_id}/{name}", "size_bytes": claim["size_bytes"], "sha256": claim["sha256"]}
                  for name, (_, claim) in source_files.items()]}
    source_entry = {**entry, "local_dir": checkpoint.relative_to(run_dir.resolve()).as_posix(),
        "files": [{**claim, "path": source.relative_to(run_dir.resolve()).as_posix()}
                  for source, claim in source_files.values()]}
    validate_adapter(run_dir, source_entry, base_entry, base_hash)
    existing = [item for item in manifest["models"] if item.get("id") == model_id]
    if existing:
        if len(existing) != 1 or existing[0] != entry:
            raise ModelAssetsError("MODEL_ADAPTER_INVALID: этот идентификатор уже занят другими весами.")
        verify_model(models_dir, entry)
        validate_adapter(models_dir, entry, base_entry, base_hash)
        return {"state": "packaged_unvalidated", "model_id": model_id, "reused": True}
    destination = artifact_path(models_dir, model_id)
    # Never replace base weights or an earlier adapter. Only a fresh directory
    # is populated, and registration happens after both hashes are verified.
    destination.mkdir(exist_ok=False)
    for name, (source, _) in source_files.items():
        shutil.copyfile(source, destination / name)
    verify_model(models_dir, entry)
    validate_adapter(models_dir, entry, base_entry, base_hash)
    manifest["models"].append(entry)
    temporary = models_dir / f".manifest-{uuid.uuid4().hex}.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(models_dir / "manifest.json")
    return {"state": "packaged_unvalidated", "model_id": model_id, "reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(json.dumps(package_adapter(args.run_dir, args.models_dir, args.model_id)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
