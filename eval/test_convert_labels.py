from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from eval.convert_labels import HEADERS, Label, parse_timestamp, read_labels, write_labels
from eval.make_splits import assign_folds, group_sources
from research.dataset.extract_archive import extract

ROOT = Path(__file__).resolve().parents[1]


def make_workbook(destination: Path, rows: list[tuple], *, formula: bool = False) -> None:
    encoded = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column, value in zip("ABC", row):
            if value is not None:
                injected_formula = "<f>1+1</f>" if formula and row_index == 2 and column == "B" else ""
                cells.append(f'<c r="{column}{row_index}" t="inlineStr">{injected_formula}<is><t>{escape(str(value))}</t></is></c>')
        encoded.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="labels" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(encoded) + '</sheetData></worksheet>')


class LabelConversionTests(unittest.TestCase):
    def test_actual_russian_forms_and_empty_timestamp(self) -> None:
        self.assertEqual(parse_timestamp("22 секунда"), 22.0)
        self.assertEqual(parse_timestamp("1минута 8 секунда"), 68.0)
        self.assertIsNone(parse_timestamp(None))
        self.assertIsNone(parse_timestamp(""))
        for value in ("about 22", "1минута 68 секунда", "-1 секунда", "NaN", "22"):
            with self.assertRaises(ValueError):
                parse_timestamp(value)

    def test_preserves_latin_and_cyrillic_identifiers_and_blank_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "labels.xlsx"
            make_workbook(source, [HEADERS, ("X-1", "22 секунда", "нарушение"), ("Х-1", None, "без нарушения"), ("Х-47", "1минута 8 секунда", "нарушение")])
            labels = read_labels(source)
            self.assertEqual(labels, [Label("X-1", 1, 22.0), Label("Х-1", 0, None), Label("Х-47", 1, 68.0)])
            output = Path(directory) / "labels.csv"
            write_labels(labels, output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["label_sec"], "")
            self.assertEqual(rows[1]["type_code"], "")

    def test_rejects_mismatched_labels_formula_and_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "labels.xlsx"
            for row in (("X-1", None, "нарушение"), ("X-1", "22 секунда", "без нарушения"), ("../X-1", "22 секунда", "нарушение")):
                make_workbook(source, [HEADERS, row])
                with self.assertRaises(ValueError):
                    read_labels(source)
            make_workbook(source, [HEADERS, ("X-1", "22 секунда", "нарушение")], formula=True)
            with self.assertRaises(ValueError):
                read_labels(source)
            make_workbook(source, [HEADERS, ("X-1", "22 секунда", "нарушение")])
            with self.assertRaises(ValueError):
                read_labels(source, Path(directory))

    @unittest.skipUnless((ROOT / "data/dataset/open/open_labels.xlsx").is_file(), "Private organizer dataset not installed")
    def test_real_workbook_exact_counts_and_duplicate_source_timestamps(self) -> None:
        labels = read_labels(ROOT / "data/dataset/open/open_labels.xlsx", ROOT / "data/dataset/open/videos")
        self.assertEqual(len(labels), 42)
        self.assertEqual(sum(label.has_violation for label in labels), 22)
        by_id = {label.video_id: label for label in labels}
        self.assertEqual(by_id["X-1"].label_sec, 22.0)
        self.assertEqual(by_id["Х-4"].label_sec, 57.0)
        self.assertEqual(by_id["Х-47"].label_sec, 68.0)
        self.assertNotIn("X-4", by_id)


class FoldTests(unittest.TestCase):
    def test_groups_preserve_multiple_events_and_never_split_copies(self) -> None:
        rows = [
            {"video_id": "Х-4", "sha256": "a" * 64, "has_violation": "1", "label_sec": "57"},
            {"video_id": "Х-47", "sha256": "a" * 64, "has_violation": "1", "label_sec": "68"},
            {"video_id": "Х-5", "sha256": "b" * 64, "has_violation": "0", "label_sec": ""},
        ]
        sources = group_sources(rows)
        self.assertEqual(sources[0]["label_secs"], [57.0, 68.0])
        self.assertEqual(sources[0]["video_ids"], ["Х-4", "Х-47"])
        self.assertEqual(assign_folds(sources, 2), assign_folds(list(reversed(sources)), 2))
        rows[1]["has_violation"] = "0"
        with self.assertRaises(ValueError):
            group_sources(rows)

    def test_real_inventory_folds_have_no_source_leakage(self) -> None:
        inventory = ROOT / "research/dataset/inventory.csv"
        if not inventory.exists():
            self.skipTest("Inventory not generated")
        with inventory.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        sources = group_sources(rows)
        folds = assign_folds(sources)
        self.assertEqual(len(sources), 40)
        self.assertEqual(len(folds), 40)
        counts = [sum(fold == value for fold in folds.values()) for value in range(5)]
        self.assertEqual(counts, [8, 8, 8, 8, 8])
        source_by_video = {row["video_id"]: row["sha256"] for row in rows}
        self.assertEqual(folds[source_by_video["Х-4"]], folds[source_by_video["Х-47"]])
        self.assertEqual(folds[source_by_video["Х-5"]], folds[source_by_video["Х-48"]])


class ExtractionTests(unittest.TestCase):
    def test_rejects_traversal_before_writing_any_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("open/safe.txt", "safe")
                stream.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                extract(archive, root / "dataset")
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((root / "dataset/open/safe.txt").exists())

    def test_preserves_identical_files_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("open/Х-1.txt", "source content")
            destination = root / "dataset"
            self.assertEqual(extract(archive, destination), 1)
            self.assertEqual(extract(archive, destination), 0)
            target = destination / "open/Х-1.txt"
            target.write_text("user content", encoding="utf-8")
            with self.assertRaises(ValueError):
                extract(archive, destination)
            self.assertEqual(target.read_text(encoding="utf-8"), "user content")


if __name__ == "__main__":
    unittest.main()
