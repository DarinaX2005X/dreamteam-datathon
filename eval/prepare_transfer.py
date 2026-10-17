"""Prepare or verify a bounded private archive of unique organizer video bytes.

The archive contains only representative MOV sources, the three evaluation
tables, the development training manifest, and its own checksum manifest.
It never includes decoded frames, model output, weights, caches or user uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import dataset_fingerprint, file_sha256, load_sources, select_sources  # noqa: E402
from eval.training import build_training_manifest  # noqa: E402

MANIFEST_MEMBER = "eval/runs/transfer-manifest.json"


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Transfer member must be a plain relative path")
    return path


def _sha_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o600
    archive.addfile(info, io.BytesIO(data))


def prepare(output: Path | None, dataset_dir: Path, video_ids: list[str] | None = None) -> dict:
    tables = [ROOT / "research/dataset/inventory.csv", ROOT / "eval/labels.csv", ROOT / "eval/folds.csv"]
    all_sources = load_sources(*tables)
    sources = select_sources(all_sources, "development", video_ids) if video_ids else all_sources
    table_hash = dataset_fingerprint(tables)
    expected_training = build_training_manifest(all_sources, table_hash)
    training_path = ROOT / "eval/training-splits.json"
    if not training_path.is_file() or json.loads(training_path.read_text(encoding="utf-8")) != expected_training:
        raise ValueError("Generate the current development training manifest before archiving")
    paths = {path.relative_to(ROOT).as_posix(): path for path in [*tables, training_path]}
    video_paths = {}
    for source in sources:
        name = "data/dataset/" + source.rows[0].relative_file
        _safe_relative(name)
        path = source.video_path(dataset_dir)
        if not path.is_file() or file_sha256(path) != source.sha256:
            raise ValueError("Original source is missing or differs from organizer SHA256")
        paths[name] = path
        video_paths[name] = source.sha256
    payloads = [{"path": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path),
                 "kind": "original_video" if name in video_paths else "supervision_metadata"}
                for name, path in sorted(paths.items())]
    manifest = {"schema_version": 1, "dataset_fingerprint": table_hash,
                "transfer_scope": "development_subset" if video_ids else "all_sources",
                "metadata_scope_unique_sources": len(all_sources),
                "metadata_scope_annotation_rows": sum(len(source.rows) for source in all_sources),
                "unique_video_sources": len(video_paths), "source_annotation_rows": sum(len(source.rows) for source in sources),
                "development_video_sources": sum(source.fold != 0 for source in sources),
                "reserved_final_holdout_video_sources": sum(source.fold == 0 for source in sources),
                "holdout_policy": "Transfer is not evaluation permission: training readers only consume development; reserved fold 0 is evaluated once after the final model is frozen.",
                "derived_frames_included": False, "model_outputs_included": False,
                "payload_size_bytes": sum(item["size_bytes"] for item in payloads), "files": payloads}
    scope_hash = hashlib.sha256("\n".join(sorted(source.sha256 for source in sources)).encode("ascii")).hexdigest()[:12]
    default_name = f"dataset-{len(sources)}sha-{table_hash[:12]}" + (f"-{scope_hash}" if video_ids else "") + ".tar"
    output = (output or ROOT / "eval/runs/transfer" / default_name).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "eval/runs"):
        raise ValueError("Private transfer archives inside the repository must stay in ignored eval/runs/")
    if output.exists():
        raise ValueError("Transfer archive already exists; verify it instead of overwriting")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{secrets.token_hex(6)}.partial")
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for payload in payloads:
                info = tarfile.TarInfo(payload["path"])
                info.size, info.mtime, info.mode = payload["size_bytes"], 0, 0o600
                with paths[payload["path"]].open("rb") as stream:
                    archive.addfile(info, stream)
            _add_bytes(archive, MANIFEST_MEMBER, manifest_bytes)
        verified = verify(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(".manifest.json").write_bytes(manifest_bytes)
    archive_hash = file_sha256(output)
    output.with_suffix(".sha256").write_text(f"{archive_hash}  {output.name}\n", encoding="utf-8")
    return {**verified, "event": "transfer_prepared", "archive": str(output),
            "archive_size_bytes": output.stat().st_size, "archive_sha256": archive_hash}


def verify(path: Path) -> dict:
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members) or any(not member.isfile() for member in members):
            raise ValueError("Transfer archive must contain unique regular files only")
        for member in members:
            _safe_relative(member.name)
        manifest_file = archive.extractfile(MANIFEST_MEMBER)
        if manifest_file is None:
            raise ValueError("Transfer manifest is absent")
        manifest = json.load(manifest_file)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
            raise ValueError("Transfer manifest schema is invalid")
        expected = {item["path"]: item for item in manifest["files"]}
        if len(expected) != len(manifest["files"]) or MANIFEST_MEMBER in expected \
                or set(expected) | {MANIFEST_MEMBER} != {member.name for member in members}:
            raise ValueError("Transfer archive contains missing, duplicate or unexpected payloads")
        video_hashes = []
        for name, record in expected.items():
            member = archive.getmember(name)
            stream = archive.extractfile(member)
            if stream is None or member.size != record["size_bytes"] or _sha_stream(stream) != record["sha256"]:
                raise ValueError("Transfer payload checksum or size mismatch")
            if record["kind"] == "original_video":
                if not name.startswith("data/dataset/open/videos/") or PurePosixPath(name).suffix.lower() != ".mov":
                    raise ValueError("Original video is outside the bounded dataset namespace")
                video_hashes.append(record["sha256"])
            elif record["kind"] != "supervision_metadata" or name not in {
                "research/dataset/inventory.csv", "eval/labels.csv", "eval/folds.csv", "eval/training-splits.json"
            }:
                raise ValueError("Archive payload is not an approved video or supervision table")
        if len(video_hashes) != len(set(video_hashes)) or len(video_hashes) != manifest["unique_video_sources"]:
            raise ValueError("Video sources are not deduplicated by SHA256")
        return {"event": "transfer_verified", "dataset_fingerprint": manifest["dataset_fingerprint"],
                "transfer_scope": manifest.get("transfer_scope", "all_sources"),
                "metadata_scope_unique_sources": manifest.get("metadata_scope_unique_sources", manifest["unique_video_sources"]),
                "unique_video_sources": len(video_hashes), "payload_files": len(expected),
                "source_annotation_rows": manifest["source_annotation_rows"],
                "development_video_sources": manifest["development_video_sources"],
                "reserved_final_holdout_video_sources": manifest["reserved_final_holdout_video_sources"],
                "payload_size_bytes": manifest["payload_size_bytes"],
                "derived_frames_included": False, "model_outputs_included": False}


def extract(path: Path, destination: Path) -> dict:
    verified = verify(path)
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_relative = PurePosixPath("eval/runs/transfer/manifests") / f"{file_sha256(path)}.json"
    created, reused = 0, 0
    with tarfile.open(path, mode="r:") as archive:
        manifest = json.load(archive.extractfile(MANIFEST_MEMBER))
        expected = {item["path"]: item for item in manifest["files"]}
        # Check every destination before writing any payload. Existing different
        # files are never overwritten, including metadata from another split.
        targets = []
        for member in archive.getmembers():
            # A small priority subset and a later complete archive have distinct
            # manifests. Preserve both by archive identity, never overwrite one.
            relative = manifest_relative if member.name == MANIFEST_MEMBER else _safe_relative(member.name)
            target = (root / Path(*relative.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Transfer destination escapes the chosen workspace")
            source = archive.extractfile(member)
            expected_hash = expected[member.name]["sha256"] if member.name in expected else _sha_stream(source)
            if target.exists():
                if not target.is_file() or file_sha256(target) != expected_hash:
                    raise ValueError("An existing destination differs; no payloads have been overwritten")
                reused += 1
            else:
                targets.append((member, target))
        for member, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
            target.chmod(0o600)
            created += 1
    return {**verified, "event": "transfer_extracted", "created_files": created, "reused_identical_files": reused,
            "manifest_relative_file": manifest_relative.as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "extract"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--video-id", action="append", help="Prepare only selected development originals, retaining all four exact metadata files")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command != "prepare" and args.archive is None:
        parser.error("--archive is required for verify/extract")
    if args.command == "extract" and args.destination is None:
        parser.error("--destination is required for extract")
    if args.video_id and args.command != "prepare":
        parser.error("--video-id is only valid for prepare")
    if args.command == "prepare":
        result = prepare(args.archive, args.dataset_dir, args.video_id)
    elif args.command == "verify":
        result = verify(args.archive)
    else:
        result = extract(args.archive, args.destination)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
