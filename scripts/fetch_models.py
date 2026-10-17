#!/usr/bin/env python3
"""Fetch and verify the public model files pinned in models/manifest.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "manifest.json"
PROFILE_MODELS = {
    "cpu": ("rfdetr-nano-onnx", "internvl3_5-1b"),
    "cuda": ("rfdetr-nano-onnx-fp32", "internvl3_5-1b"),
    "detector-only": ("rfdetr-nano-onnx",),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FetchError(RuntimeError):
    """Raised when a pinned artifact cannot be safely fetched or verified."""


@dataclass(frozen=True)
class FilePin:
    model_id: str
    path: str
    url: str
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, pin: FilePin) -> bool:
    return path.is_file() and path.stat().st_size == pin.size and sha256_file(path) == pin.sha256


def safe_target(models_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise FetchError(f"unsafe manifest path: {relative!r}")
    root = models_dir.resolve()
    target = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise FetchError(f"symlinked model directory is not allowed: {current}")
        current.mkdir(exist_ok=True)
    if target.is_symlink():
        raise FetchError(f"symlinked model target is not allowed: {target}")
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise FetchError(f"model path escapes models directory: {relative!r}") from error
    return target


def load_pins(manifest_path: Path, selected_ids: Iterable[str]) -> list[FilePin]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FetchError(f"cannot read model manifest {manifest_path}: {error}") from error
    models = {entry.get("id"): entry for entry in manifest.get("models", [])}
    pins: list[FilePin] = []
    for model_id in selected_ids:
        entry = models.get(model_id)
        if entry is None:
            raise FetchError(f"model ID is not pinned: {model_id}")
        if entry.get("status") == "rejected":
            raise FetchError(f"model is rejected by the manifest: {model_id}")
        local_dir = entry.get("local_dir")
        if not isinstance(local_dir, str):
            raise FetchError(f"model has no valid local_dir: {model_id}")
        for item in entry.get("files", []):
            path = item.get("path")
            url = item.get("url")
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if not isinstance(path, str) or PurePosixPath(path).parts[:1] != (local_dir,):
                raise FetchError(f"file path is outside model local_dir for {model_id}: {path!r}")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise FetchError(f"model URL must use HTTPS: {url!r}")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise FetchError(f"invalid SHA-256 pin for {path!r}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise FetchError(f"invalid size pin for {path!r}")
            pins.append(FilePin(model_id, path, url, digest, size))
    return pins


def _content_range(value: str | None) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else None


def _download_range(
    pin: FilePin,
    part: Path,
    start: int,
    end: int,
    *,
    timeout: float,
    retries: int,
    allow_insecure_http: bool,
) -> None:
    expected = end - start + 1
    current = part.stat().st_size if part.exists() else 0
    if current > expected:
        raise FetchError(f"partial range is oversized: {part}")
    failures = 0
    while current < expected:
        request_start = start + current
        scheme_ok = pin.url.startswith("https://") or (
            allow_insecure_http and pin.url.startswith("http://")
        )
        if not scheme_ok:
            raise FetchError(f"download URL must use HTTPS: {pin.url!r}")
        request = urllib.request.Request(
            pin.url,
            headers={"Range": f"bytes={request_start}-{end}", "User-Agent": "lado-model-fetch/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
                status = getattr(response, "status", response.getcode())
                received_range = _content_range(response.headers.get("Content-Range"))
                if status != 206 or received_range != (request_start, end, pin.size):
                    raise FetchError(
                        f"server returned invalid range for {pin.path}: "
                        f"status={status}, Content-Range={response.headers.get('Content-Range')!r}"
                    )
                with part.open("ab") as output:
                    while current < expected:
                        chunk = response.read(min(1024 * 1024, expected - current))
                        if not chunk:
                            raise OSError("response ended before the requested range")
                        output.write(chunk)
                        current += len(chunk)
            failures = 0
        except FetchError:
            raise
        except (OSError, urllib.error.URLError) as error:
            failures += 1
            if failures > retries:
                raise FetchError(f"download failed for {pin.path}: {error}") from error
            time.sleep(min(0.5 * (2 ** (failures - 1)), 8.0))


def download_file(
    pin: FilePin,
    models_dir: Path,
    *,
    workers: int = 8,
    timeout: float = 60.0,
    retries: int = 8,
    allow_insecure_http: bool = False,
) -> str:
    target = safe_target(models_dir, pin.path)
    if verified(target, pin):
        return "verified"
    part_dir = target.parent / f".{target.name}.{pin.sha256[:16]}.parts"
    if part_dir.is_symlink():
        raise FetchError(f"symlinked partial directory is not allowed: {part_dir}")
    part_dir.mkdir(exist_ok=True)
    count = min(max(1, workers), max(1, pin.size // (8 * 1024 * 1024) + 1))
    base, extra = divmod(pin.size, count)
    ranges: list[tuple[Path, int, int]] = []
    cursor = 0
    for index in range(count):
        length = base + (1 if index < extra else 0)
        ranges.append((part_dir / f"{index:03d}.part", cursor, cursor + length - 1))
        cursor += length
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(
                _download_range,
                pin,
                part,
                start,
                end,
                timeout=timeout,
                retries=retries,
                allow_insecure_http=allow_insecure_http,
            )
            for part, start, end in ranges
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    staging = target.parent / f".{target.name}.{pin.sha256[:16]}.staging"
    digest = hashlib.sha256()
    total = 0
    with staging.open("wb") as output:
        for part, start, end in ranges:
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise FetchError(f"partial range has incorrect size: {part}")
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    if total != pin.size or digest.hexdigest() != pin.sha256:
        raise FetchError(
            f"integrity check failed for {pin.path}: size={total}, sha256={digest.hexdigest()}"
        )
    os.replace(staging, target)
    return "downloaded"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_MODELS, default=None)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", action="append", dest="models", metavar="ID")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.workers > 32 or args.timeout <= 0 or args.retries < 0:
        raise FetchError("workers must be 1..32; timeout must be positive; retries cannot be negative")
    environment_profile = os.environ.get("NAVIDU_MODEL_PROFILE")
    if environment_profile is None:
        device = os.environ.get("NAVIDU_DEVICE")
        environment_profile = device if device in ("cpu", "cuda") else "cpu"
    profile = args.profile or environment_profile
    if profile not in PROFILE_MODELS:
        raise FetchError(f"invalid NAVIDU_MODEL_PROFILE: {profile!r}")
    selected = tuple(dict.fromkeys(args.models or PROFILE_MODELS[profile]))
    pins = load_pins(args.manifest.resolve(), selected)
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for pin in pins:
        target = safe_target(models_dir, pin.path)
        if args.verify_only:
            ok = verified(target, pin)
            print(f"{'OK' if ok else 'MISSING_OR_INVALID'} {pin.model_id} {pin.path}")
            failures += not ok
            continue
        result = download_file(
            pin,
            models_dir,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
        )
        print(f"{result.upper()} {pin.model_id} {pin.path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
