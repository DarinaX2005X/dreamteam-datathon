#!/usr/bin/env python3
"""Fail-fast host and model checks for Lado local or container launches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check(name: str, ok: bool, detail: str, *, required: bool = True) -> Check:
    return Check(name, "ok" if ok else ("error" if required else "warning"), detail)


def command_output(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def memory_gib() -> float | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
            return state.total / 1024**3
        except (AttributeError, OSError):
            return None
    try:
        values = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        total_kib = int(
            next(line for line in values if line.startswith("MemTotal:")).split()[1]
        )
        return total_kib / 1024**2
    except (OSError, StopIteration, ValueError):
        return None


def load_manifest(models_dir: Path) -> tuple[dict | None, str]:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.is_file():
        return (
            None,
            f"missing {manifest_path}; run scripts/fetch_models.py after the model choice is pinned",
        )
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {manifest_path}: {exc}"


def manifest_files(
    manifest: dict, selected_ids: set[str] | None = None
) -> list[tuple[Path, str | None, int | None]]:
    found: list[tuple[Path, str | None, int | None]] = []
    entries = manifest.get("files") or manifest.get("artifacts") or []
    if not entries and isinstance(manifest.get("models"), list):
        entries = [
            file_entry
            for model in manifest["models"]
            if isinstance(model, dict)
            and model.get("status") != "rejected"
            and (selected_ids is None or model.get("id") in selected_ids)
            for file_entry in model.get("files", [])
        ]
    if isinstance(entries, dict):
        entries = [
            {"path": path, **(value if isinstance(value, dict) else {"sha256": value})}
            for path, value in entries.items()
        ]
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, str):
                found.append((Path(entry), None, None))
            elif isinstance(entry, dict) and entry.get("path"):
                found.append(
                    (Path(entry["path"]), entry.get("sha256"), entry.get("size_bytes"))
                )
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def configured_model_ids(device: str) -> list[str]:
    default_detector = (
        "rfdetr-nano-onnx-fp32" if device == "cuda" else "rfdetr-nano-onnx"
    )
    model_ids = [os.getenv("NAVIDU_DETECTOR_MODEL", default_detector)]
    if os.getenv("NAVIDU_REQUIRE_SEMANTIC", "1").lower() not in {
        "0",
        "false",
        "no",
    }:
        model_ids.append(os.getenv("NAVIDU_SEMANTIC_MODEL", "internvl3_5-1b"))
    return model_ids


def run_checks(args: argparse.Namespace) -> list[Check]:
    models_dir = Path(args.models_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    checks = [
        check(
            "architecture",
            platform.machine().lower() in {"amd64", "x86_64"},
            platform.machine(),
        ),
        check("python", sys.version_info >= (3, 10), platform.python_version()),
    ]
    if args.host_only:
        docker_ok, docker_text = command_output(
            ["docker", "version", "--format", "{{.Server.Version}}"]
        )
        compose_ok, compose_text = command_output(
            ["docker", "compose", "version", "--short"]
        )
        checks.extend(
            [
                check("docker", docker_ok, docker_text or "Docker Engine unavailable"),
                check(
                    "compose", compose_ok, compose_text or "Docker Compose unavailable"
                ),
            ]
        )
    else:
        checks.extend(
            [
                check(
                    "ffmpeg",
                    shutil.which("ffmpeg") is not None
                    and shutil.which("ffprobe") is not None,
                    f"ffmpeg={shutil.which('ffmpeg')}, ffprobe={shutil.which('ffprobe')}",
                ),
                check(
                    "backend",
                    importlib.util.find_spec("navidu") is not None,
                    "navidu importable"
                    if importlib.util.find_spec("navidu")
                    else "install backend with service/runtime extras",
                ),
            ]
        )
    total_memory = memory_gib()
    checks.append(
        check(
            "ram",
            total_memory is not None and total_memory >= args.min_ram_gib,
            "unknown"
            if total_memory is None
            else f"{total_memory:.1f} GiB; required {args.min_ram_gib:g} GiB",
        )
    )
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        free_disk = shutil.disk_usage(data_dir).free / 1024**3
        disk_ok = os.access(data_dir, os.W_OK) and free_disk >= args.min_disk_gib
        disk_detail = f"{free_disk:.1f} GiB free at {data_dir}; required {args.min_disk_gib:g} GiB"
    except OSError as exc:
        disk_ok, disk_detail = False, str(exc)
    checks.append(check("data_dir", disk_ok, disk_detail))

    gpu_ok, gpu_text = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_required = args.device == "cuda"
    if gpu_ok and args.min_vram_gib:
        try:
            memory_mib = max(
                float(line.split(",")[1]) for line in gpu_text.splitlines()
            )
            gpu_ok = memory_mib >= args.min_vram_gib * 1024
            gpu_text += f"; required VRAM {args.min_vram_gib:g} GiB"
        except (IndexError, ValueError):
            gpu_ok = False
            gpu_text += "; could not parse VRAM"
    checks.append(
        check(
            "gpu", gpu_ok, gpu_text or "nvidia-smi unavailable", required=gpu_required
        )
    )
    if (
        not args.host_only
        and args.device == "cuda"
        and importlib.util.find_spec("torch")
    ):
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
            detail = f"torch={torch.__version__}, torch CUDA={torch.version.cuda}, available={cuda_ok}"
        except (ImportError, OSError, RuntimeError) as exc:
            cuda_ok, detail = False, repr(exc)
        checks.append(check("torch_cuda", cuda_ok, detail))
    if not args.host_only and args.device == "cuda":
        if importlib.util.find_spec("onnxruntime"):
            try:
                import onnxruntime

                providers = onnxruntime.get_available_providers()
                ort_ok = "CUDAExecutionProvider" in providers
                ort_detail = (
                    f"onnxruntime={onnxruntime.__version__}, providers={providers}"
                )
            except (ImportError, OSError, RuntimeError) as exc:
                ort_ok, ort_detail = False, repr(exc)
        else:
            ort_ok, ort_detail = False, "onnxruntime-gpu is not installed"
        checks.append(check("onnxruntime_cuda", ort_ok, ort_detail))

    if args.host_only:
        return checks

    manifest, manifest_detail = load_manifest(models_dir)
    checks.append(
        check(
            "model_manifest",
            manifest is not None,
            manifest_detail,
            required=args.require_models,
        )
    )
    if manifest is not None:
        configured_ids = configured_model_ids(args.device)
        available_ids = {
            model.get("id")
            for model in manifest.get("models", [])
            if isinstance(model, dict)
            and model.get("status") != "rejected"
            and isinstance(model.get("id"), str)
        }
        missing_ids = [
            model_id for model_id in configured_ids if model_id not in available_ids
        ]
        checks.append(
            check(
                "configured_models",
                not missing_ids,
                "configured model IDs present"
                if not missing_ids
                else "missing manifest IDs: " + ", ".join(missing_ids),
                required=args.require_models,
            )
        )
        listed = manifest_files(manifest, set(configured_ids))
        missing = [
            str(path) for path, _, _ in listed if not (models_dir / path).is_file()
        ]
        checks.append(
            check(
                "model_files",
                bool(listed) and not missing,
                f"{len(listed)} listed; missing: {', '.join(missing[:5]) or 'none'}",
                required=args.require_models,
            )
        )
        if args.require_models and listed and not missing:
            invalid = []
            for relative, expected_hash, expected_size in listed:
                path = models_dir / relative
                if expected_size is not None and path.stat().st_size != expected_size:
                    invalid.append(f"{relative} (size)")
                elif expected_hash and sha256(path) != expected_hash.lower():
                    invalid.append(f"{relative} (sha256)")
            checks.append(
                check(
                    "model_integrity",
                    not invalid,
                    "SHA256/size verified" if not invalid else ", ".join(invalid[:5]),
                )
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Lado runtime prerequisites without changing the host."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=os.getenv("NAVIDU_DEVICE", "auto"),
    )
    parser.add_argument(
        "--models-dir", default=os.getenv("NAVIDU_MODELS_DIR", ROOT / "models")
    )
    parser.add_argument(
        "--data-dir", default=os.getenv("NAVIDU_DATA_DIR", ROOT / "data")
    )
    parser.add_argument("--min-ram-gib", type=float, default=8)
    parser.add_argument("--min-disk-gib", type=float, default=10)
    parser.add_argument("--min-vram-gib", type=float, default=0)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="Check the fresh Docker host; skip in-image backend, FFmpeg, and model checks.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = run_checks(args)
    if args.json:
        print(
            json.dumps([asdict(item) for item in checks], ensure_ascii=False, indent=2)
        )
    else:
        for item in checks:
            print(f"[{item.status.upper():7}] {item.name}: {item.detail}")
    return 1 if any(item.status == "error" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
