"""Extract a local organizer archive without traversal or overwriting local data."""
from __future__ import annotations

import argparse
import shutil
import stat
import zipfile
import zlib
from pathlib import Path, PurePosixPath


def extract(archive_path: Path, destination: Path) -> int:
    root = destination.resolve()
    planned: list[tuple[zipfile.ZipInfo, Path]] = []
    seen: set[Path] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = PurePosixPath(entry.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
                raise ValueError(f"Unsafe archive member: {entry.filename!r}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Symlink archive member: {entry.filename!r}")
            target = (root / Path(*relative.parts)).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ValueError(f"Archive member outside output directory: {entry.filename!r}")
            if target in seen:
                raise ValueError(f"Multiple archive members target the same path: {entry.filename!r}")
            seen.add(target)
            if target.exists() and not entry.is_dir():
                checksum = 0
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum = zlib.crc32(block, checksum)
                if target.stat().st_size != entry.file_size or checksum != entry.CRC:
                    raise ValueError(f"Existing file differs; refusing overwrite: {target}")
            planned.append((entry, target))
        written = 0
        for entry, target in planned:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                written += 1
        return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    args = parser.parse_args()
    print(f"Extracted {extract(args.archive, args.output)} new files; existing identical files preserved.")


if __name__ == "__main__":
    main()
