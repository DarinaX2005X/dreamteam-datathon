from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from eval.prepare_transfer import MANIFEST_MEMBER, _add_bytes, _safe_relative, extract, verify


def fixture_archive(path: Path) -> None:
    # Inert byte strings verify transport only; no model/video quality is tested.
    payloads = {"data/dataset/open/videos/fixture.mov": b"unit-test-transport-bytes",
                "eval/labels.csv": b"unit-test-supervision-bytes"}
    manifest = {"schema_version": 1, "dataset_fingerprint": "test-only", "unique_video_sources": 1,
                "source_annotation_rows": 1, "development_video_sources": 1,
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
