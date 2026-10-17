"""Filter an already verified transfer archive to development metadata only."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset import dataset_fingerprint, load_sources  # noqa: E402
from eval.prepare_transfer import MANIFEST_MEMBER, _add_bytes, verify  # noqa: E402
from eval.training import build_training_manifest  # noqa: E402


def prepare(source: Path, destination: Path) -> dict:
    original = verify(source)
    if original["unique_video_sources"] != 32 or original["reserved_final_holdout_video_sources"]:
        raise ValueError("Expected exactly the development video archive")
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(source, "r:") as archive, tempfile.TemporaryDirectory() as temporary:
        manifest = json.load(archive.extractfile(MANIFEST_MEMBER))
        names = ["research/dataset/inventory.csv", "eval/labels.csv", "eval/folds.csv"]
        tables = {name: list(csv.DictReader(io.StringIO(archive.extractfile(name).read().decode("utf-8"))))
                  for name in names}
        allowed = {row["video_id"] for row in tables[names[2]] if int(row["fold"]) in {1, 2, 3, 4}}
        replacement = {}
        paths = []
        for name in names:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(tables[name][0]))
            writer.writeheader()
            writer.writerows(row for row in tables[name] if row["video_id"] in allowed)
            replacement[name] = stream.getvalue().encode("utf-8")
            path = Path(temporary) / Path(name).name
            path.write_bytes(replacement[name])
            paths.append(path)
        fingerprint = dataset_fingerprint(paths)
        sources = load_sources(*paths)
        if len(sources) != 32 or any(source.fold == 0 for source in sources):
            raise ValueError("Development metadata filter failed")
        training = build_training_manifest(sources, fingerprint)
        replacement["eval/training-splits.json"] = (json.dumps(training, sort_keys=True, indent=2) + "\n").encode()
        manifest.update(dataset_fingerprint=fingerprint, metadata_scope_unique_sources=32,
                        metadata_scope_annotation_rows=len(allowed), metadata_holdout_rows=0)
        for record in manifest["files"]:
            if record["path"] in replacement:
                payload = replacement[record["path"]]
                record.update(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        manifest["payload_size_bytes"] = sum(record["size_bytes"] for record in manifest["files"])
        with tarfile.open(destination, "x:", format=tarfile.PAX_FORMAT) as output:
            for record in manifest["files"]:
                name = record["path"]
                if name in replacement:
                    _add_bytes(output, name, replacement[name])
                else:
                    output.addfile(archive.getmember(name), archive.extractfile(name))
            _add_bytes(output, MANIFEST_MEMBER, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return verify(destination)


if __name__ == "__main__":
    print(json.dumps(prepare(Path(sys.argv[1]), Path(sys.argv[2]))))
