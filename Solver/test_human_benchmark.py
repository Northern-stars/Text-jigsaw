"""Tests for panel-order answer validation and puzzle rendering."""

from __future__ import annotations

import unittest
import itertools
import random
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from Solver.human_benchmark import correct_display_order, make_puzzle_image, parse_answer, run_benchmark


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
                     patch("builtins.print"), patch("builtins.input") as user_input:
                    def respond(prompt):
                        display = render.call_args.args[1]
                        self.assertEqual(display[0], target[0])
                        self.assertEqual(sorted(display), list(range(4)))
                        self.assertTrue(render.call_args.kwargs["given_first"])
                        answer = [display.index(index) + 1 for index in target[1:]]
                        if outcome == "quit":
                            return "q"
                        if outcome == "incorrect":
                            answer.reverse()
                        return " ".join(map(str, answer))

                    user_input.side_effect = respond
                    score = run_benchmark(args)
                    self.assertEqual(score, (int(outcome == "correct"), int(outcome != "quit")))
                records = args.results_path.read_text(encoding="utf-8").splitlines()
                if outcome == "quit":
                    self.assertEqual(records, [])
                    continue
                record = json.loads(records[0])
                self.assertEqual(record["task"], "order_remaining_given_first")
                self.assertEqual(record["given_display_order"], [1])
                self.assertEqual(record["correct"], outcome == "correct")
                display = record["displayed_input_indices"]
                self.assertEqual([display[i - 1] for i in record["reference_display_order"]], target[1:])
                self.assertEqual(sorted(record["answer"]), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
