from __future__ import annotations

import unittest

from PIL import Image

from Solver.agent.vlm_ordering_solver import VLMOrderingSolver
from experiment.RQ_1.model_adapters import MODEL_ADAPTERS, create_adapter


class ModelAdapterTests(unittest.TestCase):
    def test_model_registry_matches_proposal_families(self):
        self.assertEqual(
            set(MODEL_ADAPTERS),
            {"qwen2.5-vl", "internvl2.5", "llama3.2-vision", "openai", "gemini", "claude"},
        )

    def test_shared_solver_protocol_and_parser(self):
        calls = []

        def infer(prompt):
            calls.append(prompt)
            return "[2, 1]"

        solver = VLMOrderingSolver(vlm_infer=infer, panel_count=3)
        result = solver.solve([Image.new("RGB", (2, 2), "white") for _ in range(3)], ["a", "b", "c"])
        self.assertEqual(result.order, [0, 2, 1])
        self.assertEqual(len(calls), 1)

    def test_unknown_model_family_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unknown model family"):
            create_adapter("unknown", "model")


if __name__ == "__main__":
    unittest.main()
