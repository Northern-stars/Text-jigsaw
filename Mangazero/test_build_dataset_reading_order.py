"""Exercise raw pages -> window -> shuffled GT and provenance without OCR."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from Mangazero import build_dataset
from Manga109.reading_order_utils import ReadingOrderError, sort_bboxes_reading_order


class MangaZeroPageOrderTests(unittest.TestCase):
    def test_page_direction_is_independent_and_legacy_defaults_are_preserved(self):
        items = [{"id": i, "bbox": box} for i, box in enumerate(
            [[0, 0, 40, 50], [50, 0, 90, 50], [110, 0, 150, 50], [160, 0, 200, 50]])]
        for permutation in itertools.permutations(items):
            for rtl, expected, legacy in ((True, [1, 0, 3, 2], [3, 2, 1, 0]),
                                          (False, [0, 1, 2, 3], [0, 1, 2, 3])):
                ordered = sort_bboxes_reading_order(list(permutation), 50, rtl=rtl, page_split_x=100, page_rtl=False)
                self.assertEqual([p["id"] for p in ordered], expected)
                old_behavior = sort_bboxes_reading_order(list(permutation), 50, rtl=rtl, page_split_x=100)
                self.assertEqual([p["id"] for p in old_behavior], legacy)
        single = sort_bboxes_reading_order(items[:2], 50, rtl=True, page_rtl=False)
        self.assertEqual([p["id"] for p in single], [1, 0])
        with self.assertRaisesRegex(ReadingOrderError, "physical page boundary"):
            sort_bboxes_reading_order([{"bbox": [90, 0, 110, 50]}], 50, page_split_x=100, page_rtl=False)

    def test_build_reads_left_page_then_right_with_rtl_inside_each_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            Image.new("RGB", (100, 100), "white").save(raw / "left.png")
            Image.new("RGB", (100, 100), "white").save(raw / "right.png")
            # Source indices alternate between pages, with shuffled row/column
            # positions, to avoid accidentally testing the detection order.
            boxes = [[110, 60, 150, 100], [0, 0, 40, 40],
                     [160, 0, 200, 40], [50, 60, 90, 100],
                     [160, 60, 200, 100], [50, 0, 90, 40],
                     [110, 0, 150, 40], [0, 60, 40, 100]]
            annotation = {"image_path": "book/000.jpg", "meta": {"width1": 100, "width2": 100},
                          "_local_pages": {"url1_path": "left.png", "url2_path": "right.png"},
                          "frames": [{"bbox": box} for box in boxes]}
            (raw / "annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
            (raw / "manifest.jsonl").write_text(json.dumps({"annotation_path": "annotation.json"}) + "\n", encoding="utf-8")
            for direction, expected in (("--reading-order-rtl", [5, 1, 3, 7, 2, 6, 4, 0]),
                                        ("--reading-order-ltr", [1, 5, 7, 3, 6, 2, 0, 4])):
                with self.subTest(direction=direction):
                    output = root / direction.removeprefix("--")
                    argv = ["build_dataset.py", "--raw-dir", str(raw), "--output-dir", str(output),
                            "--panel-count", "8", "--puzzle-num", "1", "--train-puzzles", "1",
                            "--disable-ocr", "--page-layout", "spread", direction]
                    with patch.object(sys, "argv", argv), patch.object(build_dataset, "tqdm", None), contextlib.redirect_stdout(io.StringIO()):
                        build_dataset.main()
                    sample = json.loads((output / "000000/sample.json").read_text(encoding="utf-8"))
                    ordered = [sample["panels"][i] for i in sample["target_order"]]
                    self.assertEqual([p["source_panel_index"] for p in ordered], expected)
                    self.assertEqual([p["reading_rank_in_source"] for p in ordered], list(range(8)))
                    self.assertEqual([p["global_order"] for p in ordered], list(range(8)))
                    self.assertEqual([p["reading_rank_in_page"] for p in ordered], list(range(4)) * 2)
                    self.assertTrue(all(p["physical_page_id"].endswith(":left") for p in ordered[:4]))
                    self.assertTrue(all(p["physical_page_id"].endswith(":right") for p in ordered[4:]))
                    for p in ordered:
                        self.assertEqual(p["bbox"], boxes[p["source_panel_index"]])
                        self.assertEqual(p["page_reading_direction"], "ltr")
                        self.assertTrue((output / p["padded_path"]).is_file())
                    self.assertEqual(sample["page_reading_direction"], "ltr")
                    self.assertEqual(sample["reading_direction"], "rtl" if direction.endswith("rtl") else "ltr")
                    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
                    self.assertEqual(meta["page_reading_direction"], "ltr")
                    self.assertEqual(meta["split_counts"], {"train": 1, "val": 0, "test": 0})


if __name__ == "__main__":
    unittest.main()
