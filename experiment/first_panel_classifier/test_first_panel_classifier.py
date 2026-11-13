from __future__ import annotations

import unittest

import torch

from experiment.first_panel_classifier.model import first_panel_targets


class FirstPanelTargetTests(unittest.TestCase):
    def test_maps_reading_rank_zero_to_input_slot(self):
        target_order = torch.tensor([[2, 0, 1], [1, 2, 0]])
        self.assertTrue(torch.equal(first_panel_targets(target_order), torch.tensor([1, 2])))

    def test_rejects_non_permutation(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            first_panel_targets(torch.tensor([[0, 0, 1]]))

    def test_rejects_single_panel(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            first_panel_targets(torch.tensor([[0]]))

    def test_rejects_wrong_rank(self):
        with self.assertRaisesRegex(ValueError, "\[batch, panel_count\]"):
            first_panel_targets(torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
