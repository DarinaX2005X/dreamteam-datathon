"""Normalize the organizer's XLSX labels without guessing missing annotations.

Uses only Python's standard library. Video identifiers are Unicode-exact: the
Latin X in X-1/X-2 is not interchangeable with the Cyrillic Х in other names.
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
HEADERS = ("наименование файла", "таймстемп", "нарушение / без нарушения")
FIELDS = ("video_id", "has_violation", "label_sec", "type_code")


@dataclass(frozen=True)
class Label:
    video_id: str
    has_violation: int
    label_sec: float | None
    type_code: str = ""


def parse_timestamp(value: str | int | float | None) -> float | None:
    """Parse the exact Russian forms seen in open_labels.xlsx; reject guesses."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower().replace(",", ".")
    match = re.fullmatch(
        r"(?:(\d+)\s*минут(?:а|ы)?\s*)?(\d+(?:\.\d+)?)\s*секунд(?:а|ы)?", text
    )
    if not match:
        raise ValueError(f"Unsupported timestamp syntax: {value!r}")
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))
    if match.group(1) and seconds >= 60:
        raise ValueError(f"Seconds component must be below 60: {value!r}")
    return round(minutes * 60 + seconds, 3)


def _xlsx_rows(source: Path, sheet_name: str = "labels") -> list[list[str | None]]:
    with zipfile.ZipFile(source) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(cell.itertext()) for cell in root.findall("s:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        sheet = next((s for s in sheets if s.attrib.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Required sheet {sheet_name!r} is absent")
        relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation = next((r for r in relations if r.attrib.get("Id") == relation_id), None)
        if relation is None or relation.attrib.get("TargetMode") == "External":
            raise ValueError("Invalid worksheet relationship")
        target = relation.attrib["Target"]
        target = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        if ".." in PurePosixPath(target).parts:
            raise ValueError("Worksheet relationship traverses outside workbook")
        worksheet = ET.fromstring(archive.read(target))
        rows: list[list[str | None]] = []
        for row in worksheet.findall("s:sheetData/s:row", NS):
            cells: list[str | None] = [None, None, None]
            for cell in row.findall("s:c", NS):
                reference = cell.attrib.get("r", "")
                column = re.sub(r"\d", "", reference)
                if column not in ("A", "B", "C"):
                    if cell.find("s:v", NS) is not None or cell.find("s:is", NS) is not None:
                        raise ValueError(f"Unexpected nonempty column {column}")
                    continue
                if cell.find("s:f", NS) is not None:
                    raise ValueError(f"Formula in label cell {reference}; use explicit source values")
                value = cell.find("s:v", NS)
                kind = cell.attrib.get("t")
                if kind == "inlineStr":
                    inline = cell.find("s:is", NS)
                    text = "".join(inline.itertext()) if inline is not None else None
                elif kind == "s" and value is not None and value.text is not None:
                    text = shared[int(value.text)]
                else:
                    text = value.text if value is not None else None
                cells[ord(column) - ord("A")] = text
            if any(value is not None for value in cells):
                rows.append(cells)
        return rows


def read_labels(source: Path, videos_dir: Path | None = None) -> list[Label]:
    rows = _xlsx_rows(source)
    if not rows or tuple(rows[0]) != HEADERS:
        raise ValueError(f"Unexpected column schema: {rows[0] if rows else 'empty sheet'}")
    labels: list[Label] = []
    seen: set[str] = set()
    for row_number, (video_id, timestamp, verdict) in enumerate(rows[1:], start=2):
        if not video_id or Path(video_id).name != video_id or "/" in video_id or "\\" in video_id:
            raise ValueError(f"Unsafe or empty video identifier on row {row_number}")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id {video_id!r}")
        seen.add(video_id)
        if verdict not in ("нарушение", "без нарушения"):
            raise ValueError(f"Unknown source verdict on row {row_number}: {verdict!r}")
        has_violation = int(verdict == "нарушение")
        label_sec = parse_timestamp(timestamp)
        if bool(has_violation) != (label_sec is not None):
            raise ValueError(f"Verdict/timestamp mismatch on row {row_number}")
        labels.append(Label(video_id, has_violation, label_sec))
    if videos_dir is not None:
        videos = {path.stem: path for path in videos_dir.iterdir() if path.suffix.lower() == ".mov"}
        missing = seen - videos.keys()
        unlabelled = videos.keys() - seen
        if missing or unlabelled:
            raise ValueError(f"Video/label mismatch; missing={sorted(missing)!r}, unlabelled={sorted(unlabelled)!r}")
    return labels


def write_labels(labels: list[Label], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for label in labels:
            writer.writerow(asdict(label))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Organizer open_labels.xlsx")
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval/labels.csv"))
    args = parser.parse_args()
    labels = read_labels(args.source, args.videos_dir)
    write_labels(labels, args.output)
    positive = sum(label.has_violation for label in labels)
    print(f"Converted {len(labels)} labels: {positive} positive, {len(labels) - positive} negative. Types remain unassigned.")


if __name__ == "__main__":
    main()
