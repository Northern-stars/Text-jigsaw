"""Checks that ordering datasets read the declared manifest split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Solver.data.panel_ordering_dataset import (
    MangaZeroPanelOrderingDataset,
    load_panel_ordering_splits,
)


class PanelOrderingDatasetSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_sample(self, name: str) -> str:
        sample_path = self.dataset_dir / name / "sample.json"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            json.dumps(
                {
                    "sequence_id": name,
                    "manga_id": "work",
                    "chapter_id": "chapter",
                    "panel_count": 2,
                    "target_order": [1, 0],
                    "panels": [],
                }
            ),
            encoding="utf-8",
        )
        return sample_path.relative_to(self.dataset_dir).as_posix()

    def write_jsonl(self, name: str, records: list[dict[str, object]]) -> None:
        path = self.dataset_dir / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_combined_manifest_filters_by_split(self) -> None:
        records = []
        for split, name in (("train", "train_sample"), ("val", "val_sample"), ("test", "test_sample")):
            records.append({"split": split, "sample_path": self.write_sample(name)})
        self.write_jsonl("manifest.jsonl", records)

        train, valid, test = load_panel_ordering_splits(self.dataset_dir, use_layout=False)

        self.assertEqual([sample.sequence_id for sample in train.samples], ["train_sample"])
        self.assertEqual([sample.sequence_id for sample in valid.samples], ["val_sample"])
        self.assertEqual([sample.sequence_id for sample in test.samples], ["test_sample"])

    def test_experiment_manifest_carries_rq2_metadata(self) -> None:
        sample_path = self.write_sample("rq2_sample")
        self.write_jsonl(
            "experiment.jsonl",
            [{
                "split": "train",
                "sample_path": sample_path,
                "source_dir": str(self.dataset_dir),
                "dataset": "manga109",
                "style": "manga",
                "work_id": "book_1",
                "page_key": "book_1/page_2",
                "window_id": "book_1/page_2/window_3",
            }],
        )

        dataset = MangaZeroPanelOrderingDataset(
            self.dataset_dir,
            split="train",
            manifest_path=self.dataset_dir / "experiment.jsonl",
        )
        item = dataset.samples[0]
        self.assertEqual(item.dataset, "manga109")
        self.assertEqual(item.style, "manga")
        self.assertEqual(item.work_id, "book_1")
        self.assertEqual(item.window_id, "book_1/page_2/window_3")

    def test_split_manifest_entries_resolve_sample_paths(self) -> None:
        sample_path = self.write_sample("train_sample")
        self.write_jsonl("train.jsonl", [{"sample_path": sample_path}])
        self.write_jsonl("manifest.jsonl", [{"sample_path": sample_path}])

        dataset = MangaZeroPanelOrderingDataset(self.dataset_dir, split="train")

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0].sequence_id, "train_sample")

    def test_splitless_manifest_requires_explicit_all_split(self) -> None:
        sample_path = self.write_sample("legacy_sample")
        self.write_jsonl("manifest.jsonl", [{"sample_path": sample_path}])

        with self.assertRaisesRegex(ValueError, "has no split metadata"):
            MangaZeroPanelOrderingDataset(self.dataset_dir, split="train")

        all_samples = MangaZeroPanelOrderingDataset(self.dataset_dir, split="all")
        self.assertEqual(len(all_samples), 1)


if __name__ == "__main__":
    unittest.main()
