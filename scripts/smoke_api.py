#!/usr/bin/env python3
"""Submit real videos to the jury API and validate terminal responses."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import secrets
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JPEG_MARKERS = ((b"\xff\xd8\xff", b"\xff\xd9"),)


class SmokeFailure(RuntimeError):
    pass


@dataclass
class Result:
    path: Path
    job_id: str
    elapsed: float
    verdict: str
    violations: int


def request_json(request: urllib.request.Request, timeout: float) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.load(exc)
        except ValueError:
            payload = exc.read().decode("utf-8", "replace")
        raise SmokeFailure(f"HTTP {exc.code}: {payload}") from exc
    except (OSError, ValueError) as exc:
        raise SmokeFailure(str(exc)) from exc


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----lado-" + secrets.token_hex(16)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode()
    body = header + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def validate_processing(payload: Any) -> str:
    if not isinstance(payload, dict) or payload.get("status") != "processing":
        raise SmokeFailure(f"expected processing response, got {payload!r}")
    job_id = payload.get("job_id")
    if (
        not isinstance(job_id, str)
        or len(job_id) != 12
        or any(c not in "0123456789abcdef" for c in job_id)
    ):
        raise SmokeFailure(f"invalid job_id: {job_id!r}")
    return job_id


def validate_done(payload: Any) -> tuple[str, int]:
    if not isinstance(payload, dict):
        raise SmokeFailure("terminal response is not an object")
    if payload.get("status") == "failed":
        raise SmokeFailure(f"analysis failed: {payload.get('error')!r}")
    if payload.get("status") != "done":
        raise SmokeFailure(f"unexpected status: {payload.get('status')!r}")
    verdict = payload.get("verdict")
    violations = payload.get("violations")
    if verdict not in {"violation", "no_violation"} or not isinstance(violations, list):
        raise SmokeFailure("done response has invalid verdict or violations")
    if verdict == "no_violation" and violations:
        raise SmokeFailure("no_violation response contains violations")
    if verdict == "violation" and not violations:
        raise SmokeFailure("violation response has no violations")
    for index, violation in enumerate(violations):
        prefix = f"violations[{index}]"
        if not isinstance(violation, dict):
            raise SmokeFailure(f"{prefix} is not an object")
        if not isinstance(violation.get("type"), str) or not violation["type"]:
            raise SmokeFailure(f"{prefix}.type is empty")
        if (
            not isinstance(violation.get("description"), str)
            or not violation["description"]
        ):
            raise SmokeFailure(f"{prefix}.description is empty")
        timecode = violation.get("timecode_sec")
        if (
            not isinstance(timecode, (int, float))
            or isinstance(timecode, bool)
            or timecode < 0
        ):
            raise SmokeFailure(f"{prefix}.timecode_sec is invalid")
        frames = violation.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise SmokeFailure(f"{prefix}.evidence_frames must contain 1..3 frames")
        for frame_index, frame in enumerate(frames):
            frame_prefix = f"{prefix}.evidence_frames[{frame_index}]"
            if not isinstance(frame, dict):
                raise SmokeFailure(f"{frame_prefix} is not an object")
            timestamp = frame.get("timestamp_sec")
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or timestamp < 0
            ):
                raise SmokeFailure(f"{frame_prefix}.timestamp_sec is invalid")
            try:
                image = base64.b64decode(frame.get("image_base64", ""), validate=True)
            except (ValueError, TypeError) as exc:
                raise SmokeFailure(f"{frame_prefix}.image_base64 is invalid") from exc
            if not any(
                image.startswith(start) and image.endswith(end)
                for start, end in JPEG_MARKERS
            ):
                raise SmokeFailure(f"{frame_prefix} is not a complete JPEG")
    return verdict, len(violations)


def analyze(base: str, key: str, path: Path, timeout: float, poll: float) -> Result:
    started = time.monotonic()
    body, content_type = multipart(path)
    request = urllib.request.Request(
        base.rstrip("/") + "/api/v1/analyze",
        data=body,
        method="POST",
        headers={
            "X-API-Key": key,
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )
    status, payload = request_json(request, min(timeout, 120))
    if status != 200:
        raise SmokeFailure(f"POST returned {status}")
    job_id = validate_processing(payload)
    deadline = started + timeout
    while time.monotonic() < deadline:
        get_request = urllib.request.Request(
            base.rstrip("/") + f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": key, "Accept": "application/json"},
        )
        status, payload = request_json(
            get_request, min(15, max(1, deadline - time.monotonic()))
        )
        if status != 200:
            raise SmokeFailure(f"GET returned {status}")
        if payload.get("status") != "processing":
            verdict, count = validate_done(payload)
            return Result(path, job_id, time.monotonic() - started, verdict, count)
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    raise SmokeFailure(f"job {job_id} exceeded {timeout:g}s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Lado jury API smoke test on real local video files."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Service origin, for example http://127.0.0.1:8000",
    )
    parser.add_argument("--key", required=True, help="X-API-Key value")
    parser.add_argument("--files", nargs="+", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Maximum seconds per file, including upload",
    )
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if args.parallel < 1 or args.timeout <= 0 or args.poll <= 0:
        parser.error("--parallel, --timeout, and --poll must be positive")
    files = [path.expanduser().resolve() for path in args.files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        parser.error("missing input file(s): " + ", ".join(missing))

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                analyze, args.base, args.key, path, args.timeout, args.poll
            ): path
            for path in files
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                print(
                    f"PASS {result.path} job={result.job_id} status=done elapsed={result.elapsed:.1f}s "
                    f"verdict={result.verdict} violations={result.violations}"
                )
            except (OSError, SmokeFailure) as exc:
                failures += 1
                print(f"FAIL {path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
