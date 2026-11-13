from __future__ import annotations

import unittest

import torch

from experiment.first_panel_classifier.targets import (
    LABEL_DEFINITION, first_panel_targets, validate_checkpoint_label_definition,
)


class FirstPanelTargetTests(unittest.TestCase):
    def test_selects_first_input_slot_in_reading_sequence(self):
        target_order = torch.tensor([[2, 0, 1], [1, 2, 0]])
        self.assertTrue(torch.equal(first_panel_targets(target_order), torch.tensor([2, 1])))

    def test_rejects_non_permutation(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            first_panel_targets(torch.tensor([[0, 0, 1]]))

    def test_rejects_single_panel(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            first_panel_targets(torch.tensor([[0]]))

    def test_rejects_wrong_rank(self):
        with self.assertRaisesRegex(ValueError, r"\[batch, panel_count\]"):
            first_panel_targets(torch.tensor([0, 1]))

    def test_rejects_checkpoints_with_the_old_or_missing_label(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_checkpoint_label_definition({"best_val": 0.5})
        validate_checkpoint_label_definition({"label_definition": LABEL_DEFINITION})


if __name__ == "__main__":
    unittest.main()
