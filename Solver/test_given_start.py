"""Regression checks for given-opening comic puzzles; no model downloads needed."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from Solver.data.given_start import with_given_start, shuffle_remaining_batch, remaining_order_metrics
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.agent.mangazero_set2seq_solver import MangaZeroSetToSequenceSolver, pointer_cross_entropy
from Solver.agent.mangazero_qwen_vl_autoregressive import MangaZeroQwenVLAutoregressiveSolver, pointer_cross_entropy as qwen_loss
from Solver.agent.mangazero_permutation_classifier import MangaZeroPermutationClassifier
from Solver.agent.mangazero_qwen_vl_lora_classifier import MangaZeroQwenVLLoRAClassifier
from Solver.agent.mangazero_jigsaw_solver import MangaZeroJigsawSolver, build_permutation_bank
from Solver.agent.made2order_set2order_solver import Made2OrderPanelOrderingSolver
from Solver.agent.qwen_vl_vector_expert import QwenVLVectorExpertSolver
from Solver.agent.qwen_vl_ordering_solver import format_order_label
from Solver.agent.vlm_ordering_solver import VLMOrderingSolver, parse_ordering_answer
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingEnv
from Solver.train_mangazero_jigsaw import prepare_self_supervised_batch


class GivenStartTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "target_order": torch.tensor([2, 0, 3, 1]),
            "panel_images": torch.arange(4).float().reshape(4, 1, 1, 1),
            "layout_features": torch.arange(4).float().reshape(4, 1),
            "dialog_texts": ["0", "1", "2", "3"],
            "panels": [{"id": i} for i in range(4)],
        }

    def assert_modalities(self, batch):
        for images, layout, texts, panels in zip(batch["panel_images"], batch["layout_features"], batch["dialog_texts"], batch["panels"]):
            ids = images.flatten().tolist()
            self.assertEqual(ids, layout.flatten().tolist())
            self.assertEqual(ids, [int(text) for text in texts])
            self.assertEqual(ids, [panel["id"] for panel in panels])

    def test_start_uses_reading_order_and_all_modalities_follow(self):
        for target in itertools.permutations(range(4)):
            item = dict(self.item, target_order=torch.tensor(target))
            result = with_given_start(item)
            self.assertEqual(result["target_order"][0], 0)
            recovered = result["panel_images"][result["target_order"]].flatten().tolist()
            self.assertEqual(recovered, list(target))
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in result.items()}
            self.assert_modalities(batch)
        self.assertEqual(self.item["target_order"].tolist(), [2, 0, 3, 1])

    def test_online_shuffle_keeps_start_and_preserves_labels(self):
        batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in self.item.items()}
        observed = set()
        for _ in range(30):
            result = shuffle_remaining_batch(batch)
            self.assertEqual(result["panel_images"][0, 0].item(), 2)
            self.assertEqual(result["target_order"][0, 0].item(), 0)
            self.assertEqual(result["panel_images"][0, result["target_order"][0]].flatten().tolist(), [2, 0, 3, 1])
            self.assert_modalities(result)
            observed.add(tuple(result["target_order"][0].tolist()))
        self.assertGreater(len(observed), 1)

    def test_dataset_keeps_raw_reader_available_and_solver_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panels = []
            for i in range(4):
                Image.new("RGB", (4, 4), (i * 50,) * 3).save(root / f"{i}.png")
                panels.append({"padded_path": f"{i}.png", "dialog_text": str(i)})
            record = {"sequence_id": "sample", "manga_id": "book", "chapter_id": "chapter", "panel_count": 4,
                      "target_order": [2, 0, 3, 1], "panels": panels}
            (root / "sample.json").write_text(json.dumps(record), encoding="utf-8")
            (root / "train.jsonl").write_text(json.dumps({"sample_path": "sample.json"}) + "\n", encoding="utf-8")
            raw = MangaZeroPanelOrderingDataset(root)[0]
            given = MangaZeroPanelOrderingDataset(root, given_first=True)[0]
            self.assertEqual(raw["target_order"].tolist(), [2, 0, 3, 1])
            self.assertEqual(given["dialog_texts"], ["2", "0", "1", "3"])
            batch = collate_panel_ordering_batch([given])
            self.assertEqual(batch["given_first_index"].tolist(), [0])
            self.assertEqual(batch["target_order"].tolist(), [[0, 1, 3, 2]])
            shuffled = shuffle_remaining_batch(batch)
            self.assertEqual(shuffled["input_panel_indices"][0, shuffled["target_order"][0]].tolist(), [2, 0, 3, 1])

    def test_losses_ignore_start_and_metrics_do_not_award_free_credit(self):
        target = torch.tensor([[0, 2, 3, 1]])
        for loss_fn in (pointer_cross_entropy, qwen_loss):
            logits = torch.randn(1, 4, 4, requires_grad=True)
            loss = loss_fn(logits, target)
            loss.backward()
            self.assertEqual(logits.grad[:, 0].abs().sum(), 0)
            self.assertGreater(logits.grad[:, 1:].abs().sum(), 0)
        scores = torch.randn(1, 4, 4, requires_grad=True)
        Made2OrderPanelOrderingSolver.ranking_loss(scores, target).backward()
        self.assertEqual(scores.grad[:, 0].abs().sum(), 0)
        self.assertEqual(scores.grad[:, :, 0].abs().sum(), 0)
        metrics = remaining_order_metrics(torch.tensor([[0, 1, 3, 2]]), torch.tensor([[0, 1, 2, 3]]))
        self.assertAlmostEqual(metrics["position_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["pairwise_accuracy"], 2 / 3)
        self.assertEqual(metrics["exact_match"], 0)
        self.assertEqual(remaining_order_metrics(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))["pairwise_accuracy"], 1)

    def test_pointer_models_use_start_as_prefix_and_mask_it(self):
        model = MangaZeroSetToSequenceSolver(
            panel_count=4, solver_type="text", text_feature_dim=8, text_vocab_size=32,
            d_model=8, num_heads=2, encoder_layers=1, decoder_layers=1, use_layout=False, dropout=0,
        ).eval()
        images = torch.zeros(1, 4, 3, 4, 4)
        texts = [["start", "a", "b", "c"]]
        logits = model(images, torch.tensor([[0, 2, 1, 3]]), dialog_texts=texts)
        self.assertTrue(torch.isneginf(logits[:, 1:, 0]).all())
        self.assertTrue(torch.isfinite(pointer_cross_entropy(logits, torch.tensor([[0, 2, 1, 3]]))))
        order = model.greedy_decode(images, dialog_texts=texts)
        self.assertEqual(order[0, 0].item(), 0)
        self.assertEqual(sorted(order[0].tolist()), list(range(4)))
        # Exercise the real Qwen decoding loop using a lightweight pointer head.
        memory = torch.randn(1, 4, 8)
        qwen_order = MangaZeroQwenVLAutoregressiveSolver.greedy_decode(model, memory)
        self.assertEqual(qwen_order[0, 0].item(), 0)
        self.assertEqual(sorted(qwen_order[0].tolist()), list(range(4)))
        model.num_experts = 2
        model.activate_expert = lambda _: None
        model.clear_expert = lambda: None
        model.encode_panels = lambda *args, **kwargs: memory
        ensemble_order = QwenVLVectorExpertSolver.ensemble_predict(model, images)
        self.assertEqual(ensemble_order[0, 0].item(), 0)
        self.assertEqual(sorted(ensemble_order[0].tolist()), list(range(4)))

    def test_classifiers_only_enumerate_continuations(self):
        model = MangaZeroPermutationClassifier(
            panel_count=4, solver_type="text", text_feature_dim=8, text_vocab_size=32,
            d_model=8, num_heads=2, encoder_layers=1, use_layout=False,
        )
        self.assertEqual(model.num_classes, 6)
        self.assertTrue(torch.all(model.class_orders[:, 0] == 0))
        self.assertEqual(model.target_to_class_indices(model.class_orders).tolist(), list(range(6)))
        self.assertEqual(MangaZeroQwenVLLoRAClassifier._num_permutations(6, 120), 120)
        with self.assertRaises(ValueError):
            model.target_to_class_indices(torch.tensor([[2, 0, 3, 1]]))
        for limit in (3, None):
            bank = build_permutation_bank(4, limit)
            self.assertEqual(len(bank), limit or 6)
            self.assertTrue(all(order[0] == 0 for order in bank))

    def test_jigsaw_preparation_uses_labels_and_keeps_start(self):
        model = MangaZeroJigsawSolver(4, feature_dim=8, hidden_dim=16)
        batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in self.item.items()}
        prepared = prepare_self_supervised_batch(batch, model)
        self.assertEqual(prepared["panel_images"][0, 0].item(), 2)
        self.assert_modalities(prepared)
        self.assertEqual(prepared["panel_images"][0, prepared["target_order"][0]].flatten().tolist(), [2, 0, 3, 1])
        self.assertEqual(model.num_classes, 6)
        self.assertEqual(model.target_to_class_indices(prepared["target_order"]).numel(), 1)

    def test_made2order_prediction_keeps_start_even_if_rank_scores_disagree(self):
        scores = torch.zeros(1, 4, 4)
        scores[0, 0, 3] = 100
        scores[0, 1, 3] = 100
        scores[0, 2, 1] = 100
        scores[0, 3, 2] = 100
        stub = SimpleNamespace(forward=lambda _: (scores, None))
        self.assertEqual(Made2OrderPanelOrderingSolver.predict_order(stub, None).tolist(), [[0, 2, 3, 1]])

    def test_jigsaw_rejects_same_shape_checkpoint_with_different_class_meanings(self):
        model = MangaZeroJigsawSolver(4, max_permutations=3, feature_dim=8, hidden_dim=16)
        model.load_state_dict(model.state_dict())
        state = model.state_dict()
        state["permutation_bank"] = state["permutation_bank"].clone()
        state["permutation_bank"][1] = torch.tensor([1, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "given-start task"):
            model.load_state_dict(state)

    def test_vlm_normal_mirage_guess_and_training_labels_match(self):
        for one_indexed, response in ((False, "[3, 1, 2]"), (True, "[4, 2, 3]")):
            solver = VLMOrderingSolver(lambda _: response, one_indexed=one_indexed)
            for method in (solver.solve, solver.mirage_inference, solver.guess_inference):
                result = method([Image.new("RGB", (2, 2)) for _ in range(4)])
                self.assertEqual(result.order, [0, 3, 1, 2])
                self.assertIn(f"Panel {int(one_indexed)} is the given opening panel", result.prompt[-1]["content"][-1]["text"])
            self.assertEqual(format_order_label([0, 3, 1, 2], start_index=int(one_indexed)), response.strip("[]"))
        for response in ("[0, 3, 1, 2]", "[1, 1, 2]", "[1, 2]", "[1, 2, 4]"):
            with self.assertRaises(ValueError):
                parse_ordering_answer(response, 4)

    def test_environment_starts_with_given_panel_and_rewards_only_tail(self):
        sample = with_given_start(dict(self.item, sequence_id="test"))
        with patch("Solver.env.mangazero_panel_env.MangaZeroPanelOrderingDataset", return_value=[sample]):
            env = MangaZeroPanelOrderingEnv("unused")
        obs = env.reset(0)
        self.assertEqual(obs["selected_order"].tolist(), [0])
        self.assertEqual(obs["selected_mask"].tolist(), [True, False, False, False])
        _, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertEqual(info["position_accuracy"], 0)
        env.reset(0)
        for action in sample["target_order"][1:]:
            _, reward, done, info = env.step(action)
        self.assertTrue(done)
        self.assertEqual(reward, 2)
        self.assertEqual(info["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
