from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiment.RQ_2.rq2_style_balance import select_balanced_indices
from experiment.run_logger import ExperimentLogger


class RQ2SamplingTests(unittest.TestCase):
    def make_dataset(self):
        samples = []
        for style in ("manga", "comics"):
            for work in ("work_a", "work_b"):
                for puzzle in range(3):
                    samples.append(SimpleNamespace(style=style, work_id=f"{style}:{work}", sequence_id=f"{style}:{work}:{puzzle}"))
        return SimpleNamespace(samples=samples)

    def test_conditions_have_exact_equal_budget_and_mixed_style_balance(self):
        dataset = self.make_dataset()
        indices = {
            condition: select_balanced_indices(dataset, condition, budget=6, seed=11)
            for condition in ("manga_only", "comics_only", "mixed")
        }
        self.assertTrue(all(len(values) == 6 for values in indices.values()))
        selected = [dataset.samples[index] for index in indices["mixed"]]
        self.assertEqual(sum(sample.style == "manga" for sample in selected), 3)
        self.assertEqual(sum(sample.style == "comics" for sample in selected), 3)
        for condition, values in indices.items():
            self.assertEqual(len(values), len(set(values)))

    def test_sampler_is_deterministic_and_does_not_repeat_puzzles(self):
        dataset = self.make_dataset()
        first = select_balanced_indices(dataset, "mixed", budget=6, seed=7)
        self.assertEqual(first, select_balanced_indices(dataset, "mixed", budget=6, seed=7))


class ExperimentLoggerTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            logger = ExperimentLogger(Path(temp_dir) / "run", {"seed": 2})
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            path = logger.save_checkpoint("state.pt", model_state=expected, optimizer=optimizer, epoch=2, global_step=9)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            state = ExperimentLogger.load_checkpoint(path, optimizer=optimizer, restore_rng=False)
            model.load_state_dict(state["model"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["global_step"], 9)


if __name__ == "__main__":
    unittest.main()
