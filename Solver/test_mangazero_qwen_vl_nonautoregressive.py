"""Tests for the Qwen-VL non-autoregressive decoder without model downloads."""

import unittest

import torch

from Solver.agent.mangazero_qwen_vl_nonautoregressive import (
    NonAutoregressivePointerDecoder,
    nonautoregressive_ordering_loss,
)


class NonAutoregressiveDecoderTests(unittest.TestCase):
    def test_parallel_decoder_outputs_assignment_scores_and_permutation(self):
        torch.manual_seed(3)
        decoder = NonAutoregressivePointerDecoder(
            panel_count=4, d_model=8, encoder_layers=1, decoder_layers=1,
            num_heads=2, dropout=0,
        ).eval()
        memory = torch.randn(2, 4, 8)
        logits = decoder(memory)
        self.assertEqual(tuple(logits.shape), (2, 4, 4))
        order = decoder.predict_order(memory)
        self.assertEqual(tuple(order.shape), (2, 4))
        self.assertTrue(all(sorted(row.tolist()) == [0, 1, 2, 3] for row in order))

    def test_two_direction_loss_is_differentiable(self):
        logits = torch.randn(2, 4, 4, requires_grad=True)
        target = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        loss = nonautoregressive_ordering_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_invalid_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "permutations"):
            nonautoregressive_ordering_loss(torch.randn(1, 3, 3), torch.tensor([[0, 0, 1]]))


if __name__ == "__main__":
    unittest.main()
