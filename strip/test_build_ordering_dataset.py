from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset
from strip.build_ordering_dataset import build_ordering_dataset


class StripOrderingTests(unittest.TestCase):
    def test_split_isolation_and_shared_manifest_format(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            raw_dir = root_path / "raw"
            output_dir = root_path / "ordering"
            for index in range(4):
                strip_dir = raw_dir / "strips" / f"{index:08d}"
                strip_dir.mkdir(parents=True)
                image = Image.new("RGB", (400, 100), "white")
                draw = ImageDraw.Draw(image)
                for column in range(4):
                    draw.rectangle((column * 100, 0, column * 100 + 99, 99), fill=(index * 40, column * 40, 20))
                image.save(strip_dir / "image.png")
                (strip_dir / "annotation.json").write_text("{}", encoding="utf-8")
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / "manifest.jsonl").write_text(
                "".join(json.dumps({
                    "strip_id": f"{index:08d}",
                    "image_path": f"strips/{index:08d}/image.png",
                    "annotation_path": f"strips/{index:08d}/annotation.json",
                }) + "\n" for index in range(4)), encoding="utf-8"
            )
            args = argparse.Namespace(
                raw_dir=raw_dir, output_dir=output_dir, panel_count=4, stride=1,
                shuffle_per_window=2, puzzle_num=6, train_puzzles=2, val_puzzles=2,
                test_puzzles=2, grid_rows=1, grid_cols=4, reading_order="ltr",
                target_width=32, target_height=32, image_format="jpg", seed=0,
            )
            result = build_ordering_dataset(args)
            self.assertEqual(result["split_counts"], {"train": 2, "val": 2, "test": 2})
            manifest = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest), 6)
            self.assertEqual({json.loads(line)["split"] for line in manifest}, {"train", "val", "test"})
            strips_by_split = {
                split: {json.loads(line)["strip_id"] for line in manifest if json.loads(line)["split"] == split}
                for split in ("train", "val", "test")
            }
            self.assertFalse(strips_by_split["train"] & strips_by_split["val"])
            self.assertFalse(strips_by_split["train"] & strips_by_split["test"])
            self.assertFalse(strips_by_split["val"] & strips_by_split["test"])
            for split, expected_count in (("train", 2), ("val", 2), ("test", 2)):
                dataset = MangaZeroPanelOrderingDataset(output_dir, split=split, use_layout=False)
                self.assertEqual(len(dataset), expected_count)


if __name__ == "__main__":
    unittest.main()
