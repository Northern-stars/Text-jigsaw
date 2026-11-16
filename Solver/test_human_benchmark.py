"""Tests for panel-order answer validation and puzzle rendering."""

from __future__ import annotations

import unittest
import itertools
import random
import argparse
import json
import io
from contextlib import redirect_stdout
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from Solver.human_benchmark import correct_display_order, make_puzzle_image, parse_answer, print_puzzle_source, run_benchmark
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset
from Solver.data.source_provenance import raw_source_location


class HumanBenchmarkTests(unittest.TestCase):
    def test_correct_order_is_mapped_to_randomized_display_numbers(self) -> None:
        # The display already lists the input slots in target reading order.
        self.assertEqual(correct_display_order([2, 0, 3, 1], [2, 0, 3, 1]), [1, 2, 3, 4])

    def test_generated_labels_survive_all_six_panel_input_permutations(self) -> None:
        rng = random.Random(109)
        for input_order in itertools.permutations(range(6)):
            target = [input_order.index(rank) for rank in range(6)]
            display = list(range(6))
            rng.shuffle(display)
            answer = correct_display_order(target, display)
            # Recover source reading ranks through the actual display slots.
            self.assertEqual([input_order[display[i - 1]] for i in answer], list(range(6)))

    def test_display_mapping_rejects_invalid_permutations(self) -> None:
        for target, display in (([0, 0], [0, 1]), ([0, 1], [1, 1]), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                correct_display_order(target, display)

    def test_answer_parser_accepts_delimiters_and_quit(self) -> None:
        self.assertEqual(parse_answer("3, 1 2", 3), [3, 1, 2])
        self.assertIsNone(parse_answer("q", 3))
        with self.assertRaises(ValueError):
            parse_answer("1 1 3", 3)

    def test_puzzle_sheet_labels_panels(self) -> None:
        panels = [np.zeros((3, 16, 16), dtype=np.float32) for _ in range(3)]
        image = make_puzzle_image(panels, [2, 0, 1], columns=2)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (68, 128))

    def test_remaining_answer_excludes_given_start(self) -> None:
        self.assertEqual(parse_answer("3, 2 4", 4, first_panel_id=2), [3, 2, 4])
        self.assertIsNone(parse_answer("q", 4, first_panel_id=2))
        for value in ("1 3 2 4", "2 2 4", "3 4", "2 3 5"):
            with self.assertRaises(ValueError):
                parse_answer(value, 4, first_panel_id=2)

    def test_given_start_has_its_own_row(self) -> None:
        panels = [np.full((3, 16, 16), value, dtype=np.uint8) for value in (50, 100, 150)]
        image = make_puzzle_image(panels, [2, 0, 1], columns=2, given_first=True)
        self.assertEqual(image.getpixel((12, 32)), (150, 150, 150))
        self.assertEqual(image.getpixel((40, 32)), (233, 233, 233))
        self.assertEqual(image.getpixel((12, 90)), (50, 50, 50))
        self.assertEqual(image.getpixel((40, 90)), (100, 100, 100))

    def test_benchmark_scores_only_remaining_panels_and_records_given_start(self) -> None:
        target = [2, 0, 3, 1]  # The opening panel is not input slot zero.
        sample = {
            "target_order": torch.tensor(target),
            "panel_images": np.zeros((4, 3, 16, 16), dtype=np.uint8),
            "panels": [{} for _ in target],
            "sequence_id": "test-sequence",
            "dataset": "mangazero",
        }
        for outcome in ("correct", "incorrect", "quit"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(
                    dataset_dir=Path(directory), split="test", num_questions=1,
                    image_width=16, image_height=16, columns=2, seed=7,
                    panel_text_mode="none", output_dir=None,
                    results_path=Path(directory) / "results.jsonl",
                )
                with patch("Solver.human_benchmark.MangaZeroPanelOrderingDataset", return_value=[sample]), \
                     patch("Solver.human_benchmark.make_puzzle_image") as render, \
                     patch("builtins.print") as output, patch("builtins.input") as user_input:
                    def respond(prompt):
                        display = render.call_args.args[1]
                        self.assertEqual(display[0], target[0])
                        self.assertEqual(sorted(display), list(range(4)))
                        self.assertTrue(render.call_args.kwargs["given_first"])
                        answer = [display.index(index) + 1 for index in target[1:-1]]
                        if outcome == "quit":
                            return "q"
                        if outcome == "incorrect":
                            answer.reverse()
                        return " ".join(map(str, answer))

                    user_input.side_effect = respond
                    score = run_benchmark(args)
                    self.assertEqual(score, (int(outcome == "correct"), int(outcome != "quit")))
                    printed = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
                    self.assertIn("dataset=mangazero | puzzle=test-sequence", printed)
                    self.assertIn(f"Dataset path: {Path(directory).resolve()}", printed)
                records = args.results_path.read_text(encoding="utf-8").splitlines()
                if outcome == "quit":
                    self.assertEqual(records, [])
                    continue
                record = json.loads(records[0])
                self.assertEqual(record["task"], "order_middle_given_first_last")
                self.assertEqual(record["given_display_order"], [1, 4])
                self.assertEqual(record["correct"], outcome == "correct")
                display = record["displayed_input_indices"]
                self.assertEqual([display[i - 1] for i in record["reference_display_order"]], target[1:-1])
                self.assertEqual(sorted(record["answer"]), [2, 3])

    def test_source_paths_follow_manifest_source_root_and_displayed_panel_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, source = root / "experiment", root / "manga109"
            selection.mkdir()
            source.mkdir()
            panels = [{"source_path": f"originals/{i}.png"} for i in range(3)]
            (source / "originals").mkdir()
            for i in range(3):
                (source / f"originals/{i}.png").touch()
            entry = {"sequence_id": "book_page_window", "dataset": "manga109",
                     "manga_id": "book", "chapter_id": "chapter", "panel_count": 3,
                     "panels": panels, "target_order": [2, 0, 1],
                     "source_dir": str(source), "split": "test"}
            (selection / "test.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
            dataset = MangaZeroPanelOrderingDataset(selection, split="test")
            capture = io.StringIO()
            with redirect_stdout(capture):
                print_puzzle_source(dataset, entry, 0, selection, [2, 0, 1])
            text = capture.getvalue()
            self.assertIn("dataset=manga109 | puzzle=book_page_window | split=test | index=0", text)
            self.assertIn(f"Source dataset path: {source.resolve()}", text)
            for position, index in enumerate([2, 0, 1], 1):
                self.assertIn(f"Raw source: id={index}.png | path={(source / panels[index]['source_path']).resolve()}", text)
            self.assertNotIn("target_order", text)

    def test_mangazero_raw_manifest_maps_source_ids_to_raw_directory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            puzzles, raw = root / "puzzles", root / "raw"
            puzzles.mkdir()
            raw.mkdir()
            entry = {"image_path": "book/012.jpg", "annotation_path": "000460/annotation.json",
                     "url1_path": "000460/page_left.png", "url2_path": "000460/page_right.png"}
            (raw / "manifest.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
            (raw / "000460").mkdir()
            (raw / entry["annotation_path"]).write_text(json.dumps({"meta": {"width1": 100}}), encoding="utf-8")
            # Dataset root metadata is sufficient; no dataset regeneration.
            (puzzles / "meta.json").write_text(json.dumps({"raw_dir": str(raw)}), encoding="utf-8")
            panel = {"source_image_path": "book/012.jpg", "bbox": [5, 0, 80, 40],
                     "page_split_x": 50, "padded_path": "000000/panels_padded/00.jpg"}
            for bbox in ([5, 0, 80, 40], [100, 0, 190, 40], [90, 0, 110, 40]):
                location, source_id = raw_source_location(dict(panel, bbox=bbox), puzzles)
                self.assertEqual(location, (raw / "000460").resolve())
                self.assertEqual(source_id, "book/012.jpg")
            capture = io.StringIO()
            with redirect_stdout(capture):
                print_puzzle_source([], {"sequence_id": "puzzle", "panels": [panel, panel]},
                                    0, puzzles, [0, 1], raw_dir=raw)
            self.assertIn(f"Raw source: id=000460 | path={(raw / '000460').resolve()}", capture.getvalue())
            self.assertEqual(capture.getvalue().count("Raw source:"), 1)
            self.assertNotIn("page_left", capture.getvalue())
            self.assertNotIn("page_right", capture.getvalue())
            self.assertNotIn("panels_padded", capture.getvalue())
            # An explicit root takes precedence; never invent a raw file path.
            self.assertEqual(raw_source_location(panel, puzzles, root / "absent"), (None, "book/012.jpg"))

    def test_manifest_without_annotation_resolves_raw_id_from_page_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(json.dumps({"image_path": "book/page.jpg",
                "url1_path": "000460/left.jpg", "url2_path": "000460/right.jpg"}) + "\n", encoding="utf-8")
            location, _ = raw_source_location({"source_image_path": "book/page.jpg", "bbox": [0, 0, 10, 10]},
                                        root, raw_dir=root)
            self.assertEqual(location, (root / "000460").resolve())


if __name__ == "__main__":
    unittest.main()
