from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from eval.prepare_transfer import MANIFEST_MEMBER, _add_bytes, _safe_relative, extract, verify


def fixture_archive(path: Path, *, extra_video: bool = False) -> None:
    # Inert byte strings verify transport only; no model/video quality is tested.
    payloads = {"data/dataset/open/videos/fixture.mov": b"unit-test-transport-bytes",
                "eval/labels.csv": b"unit-test-supervision-bytes"}
    if extra_video:
        payloads["data/dataset/open/videos/another.mov"] = b"another-unit-test-transport-payload"
    count = 2 if extra_video else 1
    manifest = {"schema_version": 1, "dataset_fingerprint": "test-only", "unique_video_sources": count,
                "source_annotation_rows": count, "development_video_sources": count,
                "reserved_final_holdout_video_sources": 0,
                "payload_size_bytes": sum(map(len, payloads.values())),
                "files": [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                           "kind": "original_video" if name.endswith(".mov") else "supervision_metadata"}
                          for name, data in payloads.items()]}
    with tarfile.open(path, "w") as archive:
        for name, data in payloads.items():
            _add_bytes(archive, name, data)
        _add_bytes(archive, MANIFEST_MEMBER, json.dumps(manifest).encode())


class TransferTests(unittest.TestCase):
    def test_priority_subset_and_full_archive_preserve_separate_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, full, destination = root / "small.tar", root / "full.tar", root / "destination"
            fixture_archive(small)
            fixture_archive(full, extra_video=True)
            first = extract(small, destination)
            second = extract(full, destination)
            self.assertNotEqual(first["manifest_relative_file"], second["manifest_relative_file"])
            self.assertTrue((destination / first["manifest_relative_file"]).is_file())
            self.assertTrue((destination / second["manifest_relative_file"]).is_file())
            self.assertEqual(second["reused_identical_files"], 2)
            self.assertEqual(second["created_files"], 2)
            self.assertEqual((destination / "data/dataset/open/videos/another.mov").read_bytes(),
                             b"another-unit-test-transport-payload")

    def test_unsafe_member_paths_are_rejected(self) -> None:
        for value in ("/absolute", "../escape", "safe/../escape", "safe\\escape", "C:/escape", "safe//empty"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _safe_relative(value)

    def test_extract_is_idempotent_and_never_overwrites_existing_different_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, destination = root / "fixture.tar", root / "destination"
            fixture_archive(archive)
            self.assertEqual(verify(archive)["unique_video_sources"], 1)
            conflicting = destination / "eval/labels.csv"
            conflicting.parent.mkdir(parents=True)
            conflicting.write_bytes(b"existing-different-labels")
            with self.assertRaisesRegex(ValueError, "existing destination differs"):
                extract(archive, destination)
            self.assertFalse((destination / "data").exists())
            self.assertEqual(conflicting.read_bytes(), b"existing-different-labels")
            clean = root / "clean"
            first, second = extract(archive, clean), extract(archive, clean)
            self.assertEqual(first["created_files"], 3)
            self.assertEqual(second["created_files"], 0)
            self.assertEqual(second["reused_identical_files"], 3)


if __name__ == "__main__":
    unittest.main()
