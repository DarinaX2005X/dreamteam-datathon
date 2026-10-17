#!/usr/bin/env python3
"""Queue an explicitly attested team-owned recording as a read-only example.

The running service worker owns inference. This command never supplies findings,
substitutes a pipeline, warms a second model, or changes the readiness contract.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi import UploadFile  # noqa: E402
from navidu.service.config import Settings  # noqa: E402
from navidu.service.exceptions import ServiceError  # noqa: E402
from navidu.service.readiness import PREPARING_MESSAGE, published_analysis_valid  # noqa: E402
from navidu.service.store import Store  # noqa: E402
from navidu.service.uploads import persist_upload, probe_video  # noqa: E402

from scripts.run_local import load_dotenv  # noqa: E402

ORGANIZER_INVENTORY = ROOT / "research" / "dataset" / "inventory.csv"
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def organizer_hashes(extra_inventories: tuple[Path, ...] = ()) -> set[str]:
    """Always include the repository's known source hashes when it is present."""
    paths = [ORGANIZER_INVENTORY] if ORGANIZER_INVENTORY.is_file() else []
    paths.extend(extra_inventories)
    result = set()
    for path in dict.fromkeys(path.resolve() for path in paths):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "sha256" not in reader.fieldnames:
                raise ValueError("Organizer inventory must contain a sha256 column.")
            for row in reader:
                value = (row.get("sha256") or "").strip()
                if not SHA256.fullmatch(value):
                    raise ValueError("Organizer inventory contains an invalid SHA256.")
                result.add(value.lower())
    return result


def source_digest(source: Path, settings: Settings) -> str:
    if not source.is_file():
        raise ValueError("The own-recording path must name a local video file.")
    size = source.stat().st_size
    if not size:
        raise ValueError("The own recording is empty.")
    if size > settings.max_upload_bytes:
        raise ValueError("The own recording exceeds NAVIDU_MAX_UPLOAD_MB.")
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise ValueError("The own recording exceeds NAVIDU_MAX_UPLOAD_MB.")
            digest.update(chunk)
    return digest.hexdigest()


def enqueue_example(
    source: Path,
    settings: Settings,
    *,
    own_recording: bool,
    extra_inventories: tuple[Path, ...] = (),
) -> str:
    """Use the same streamed copy, media checks and durable queue as UI uploads."""
    if own_recording is not True:
        raise ValueError("Explicit --own-recording attestation is required.")
    settings.validate()
    source = source.expanduser().resolve()
    digest = source_digest(source, settings)
    if digest in organizer_hashes(extra_inventories):
        raise ValueError("This recording matches an organizer source and cannot become an example.")
    store = Store(settings)
    if not store.health()["ready"]:
        raise ServiceError(503, "NOT_READY", PREPARING_MESSAGE)

    # persist_upload closes its input and removes failed partial copies itself.
    item = UploadFile(filename=source.name, file=source.open("rb"))
    check_id, name, path, copied_digest, meta = asyncio.run(
        persist_upload(item, settings, probe=probe_video)
    )
    attestation = path.parent / "example.json"
    try:
        if copied_digest != digest:
            raise ValueError("The recording changed while being copied; submit it again.")
        health = store.health()
        if not health["ready"]:
            raise ServiceError(503, "NOT_READY", PREPARING_MESSAGE)
        attestation.write_text(json.dumps({
            "kind": "team-owned-recording",
            "ownership_attested": True,
            "source_sha256": copied_digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder": "scripts/build_example.py",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        store.enqueue(check_id=check_id, file_name=name, sha256=copied_digest,
            priority="ui", client_ip_hash=store.ip_hash("local-example-builder"),
            video_file=path.name, video_meta=meta, pipeline_version=health["pipeline_version"],
            is_example=True)
    except BaseException:
        # An enqueue outcome may be uncertain after its DB transaction committed.
        # Never delete an input which a worker may already own.
        with store.connection() as db:
            enqueued = db.execute("SELECT 1 FROM checks WHERE id=?", (check_id,)).fetchone()
        if enqueued is None:
            attestation.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        raise
    return check_id


def wait_for_example(store: Store, check_id: str, *, wait_seconds: float) -> None:
    """Do not emit deployment configuration for failed or unvalidated work."""
    deadline = time.monotonic() + wait_seconds
    while True:
        row = store.get(check_id)
        if not row["is_example"]:
            raise ValueError("The check is not a read-only example.")
        if row["status"] == "done":
            if not published_analysis_valid(row):
                raise ValueError("The example does not contain a validated automatic analysis.")
            return
        if row["status"] in {"failed", "cancelled"}:
            error = json.loads(row["error_json"] or "{}")
            code = error.get("code", "ANALYSIS_FAILED")
            raise ValueError(f"Example analysis did not complete ({code}). No example was enabled.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The example is still queued or processing. No example was enabled.")
        time.sleep(min(0.5, remaining))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True,
                        help="Local path to the user's or team's own recording.")
    parser.add_argument("--own-recording", action="store_true", required=True,
                        help="Attest that this is an own recording, not organizer dataset material.")
    parser.add_argument("--organizer-inventory", action="append", type=Path, default=[],
                        help="Additional CSV with a sha256 column; the repository inventory always applies.")
    parser.add_argument("--wait-seconds", type=float, default=1800,
                        help="Wait for the existing local worker (default: 1800 seconds).")
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        parser.error("--wait-seconds must be a finite positive number.")
    try:
        # Match the local launcher without overwriting already exported settings.
        load_dotenv(ROOT / ".env")
        settings = Settings.from_env()
        check_id = enqueue_example(args.video, settings, own_recording=args.own_recording,
                                   extra_inventories=tuple(args.organizer_inventory))
        print(f"Queued read-only example: {check_id}. Waiting for the local service worker.", file=sys.stderr)
        wait_for_example(Store(settings), check_id, wait_seconds=args.wait_seconds)
        print(f"NAVIDU_EXAMPLE_CHECK_ID={check_id}")
        return 0
    except KeyboardInterrupt:
        print("Stopped waiting. The queued check remains in the local service; no example was enabled.", file=sys.stderr)
        return 130
    except ServiceError as exc:
        print(f"Example not enabled: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TimeoutError, RuntimeError) as exc:
        print(f"Example not enabled: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
