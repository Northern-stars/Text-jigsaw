"""Tests for panel-order answer validation and puzzle rendering."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from Solver.human_benchmark import correct_display_order, make_puzzle_image, parse_answer


class HumanBenchmarkTests(unittest.TestCase):
    def test_correct_order_is_mapped_to_randomized_display_numbers(self) -> None:
        # Input panels 0..3 have reading ranks [2, 0, 3, 1]. The display is
        # input panel order [2, 0, 3, 1], so the answer should be 4, 3, 2, 1.
        self.assertEqual(correct_display_order([2, 0, 3, 1], [2, 0, 3, 1]), [4, 3, 2, 1])

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
