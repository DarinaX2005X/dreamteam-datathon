"""Small, content-based runtime snapshots for readiness and result reuse.

Weight bytes are deliberately not read here. Their pinned hashes are verified by
prepare once; health only reads the manifest, parameters and implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from navidu.contracts.pipeline import PipelineConfig

_OPTIONS = {
    "NAVIDU_DETECTION_THRESHOLD": "0.45",
    "NAVIDU_CPU_THREADS": "6",
    "NAVIDU_SEMANTIC_WINDOW_FRAMES": "8",
    "NAVIDU_SEMANTIC_FPS": "1",
    "NAVIDU_SEMANTIC_MAX_NEW_TOKENS": "512",
    "NAVIDU_SEMANTIC_MAX_PATCHES": "1",
    "NAVIDU_SEMANTIC_DTYPE": "auto",
    "NAVIDU_SEMANTIC_STRUCTURED_OUTPUT": "0",
    "NAVIDU_QWEN_MAX_PIXELS": "451584",
    "NAVIDU_QWEN_INPUT_MODE": "images",
    "NAVIDU_QWEN_DETAIL_VIEW": "0",
    "NAVIDU_SEMANTIC_PROMPT_VARIANT": "v5",
    "NAVIDU_PROPOSAL_ADAPTER": "",
}
_PACKAGES = ("numpy", "pillow", "av", "supervision", "onnxruntime", "onnxruntime-gpu",
             "torch", "torchvision", "transformers", "safetensors", "tokenizers", "peft", "accelerate",
             "lm-format-enforcer")
_IMPLEMENTATION = ("perception", "ingest", "scene", "facts", "rules", "evidence", "pipeline", "contracts")


def base_version(version: str) -> str:
    return version.split("+runtime.", 1)[0]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    code = {}
    for name in _IMPLEMENTATION:
        for path in sorted((package_root / name).rglob("*.py")):
            code[path.relative_to(package_root).as_posix()] = _file_hash(path)
    return code


_IMPORTED_CODE = _implementation_hashes()


def configuration_snapshot(config: PipelineConfig, entries: list[dict], *, require_semantic: bool) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    if _implementation_hashes() != _IMPORTED_CODE:
        raise RuntimeError("RUNTIME_RESTART_REQUIRED: код анализа изменён; перезапустите локальный worker.")
    parameters = {}
    for directory, prefix in ((config.rules_dir, "rules"), (package_root / "facts", "facts")):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}:
                    parameters[f"{prefix}/{path.relative_to(directory).as_posix()}"] = _file_hash(path)
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "base_version": base_version(config.pipeline_version),
        "models": [{"id": item["id"], "runtime": item.get("runtime"),
                    "revision": item.get("revision"),
                    "base_model": item.get("base_model"), "task": item.get("task"),
                    "files": [{key: file[key] for key in ("path", "size_bytes", "sha256")}
                              for file in sorted(item["files"], key=lambda file: file["path"])]}
                   for item in entries],
        "require_semantic": require_semantic,
        "analysis_fps": config.analysis_fps,
        "diagnostic_only": config.diagnostic_only,
        "experimental_analysis": config.experimental_analysis,
        "device_requested": config.device,
        "evidence_overlay": config.evidence_overlay,
        "max_duration_sec": config.max_duration_sec,
        "options": {name: os.getenv(name, default) for name, default in _OPTIONS.items()},
        "implementation": _IMPORTED_CODE,
        "parameters": parameters,
        "packages": packages,
        "validation": {"sha256": _file_hash(config.models_dir / "validation.json")
                       if (config.models_dir / "validation.json").is_file() else None},
    }


def snapshot_hash(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def effective_version(snapshot: dict, *, devices: dict[str, str]) -> str:
    return f"{snapshot['base_version']}+runtime.{snapshot_hash({**snapshot, 'devices': devices})[:20]}"
