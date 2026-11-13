"""Tests for panel-order answer validation and puzzle rendering."""

from __future__ import annotations

import unittest
import itertools
import random

import numpy as np
from PIL import Image

from Solver.human_benchmark import correct_display_order, make_puzzle_image, parse_answer


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


if __name__ == "__main__":
    unittest.main()
