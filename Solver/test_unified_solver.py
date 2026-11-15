"""Endpoint semantics and real unified training/checkpoint paths, without downloads."""

import contextlib
import io
import itertools
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from Solver.puzzle import (OrderingPuzzle, ordering_metrics, reorder_batch,
                           shuffle_middle_batch, with_given_endpoints)
from Solver.runtime import read_checkpoint, train_epoch
from Solver.solvers import create_solver
from Solver.train import main as train_main
from Solver.test import main as test_main


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.config = types.SimpleNamespace(hidden_size=8, use_cache=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        return types.SimpleNamespace(hidden_states=(hidden,), loss=hidden.square().mean())


class FakeProcessor:
    tokenizer = types.SimpleNamespace(convert_tokens_to_ids=lambda _: 9, unk_token_id=0)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, conversation, **kwargs):
        return "panels"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[9, 1, 9, 2, 9, 3, 9]] * len(text))}


class EndpointTests(unittest.TestCase):
    def test_all_modalities_and_provenance_follow_both_endpoints(self):
        for permutation in itertools.permutations(range(4)):
            item = {"target_order": torch.tensor(permutation),
                    "panel_images": torch.arange(4).reshape(4, 1),
                    "layout_features": torch.arange(4).reshape(4, 1),
                    "dialog_texts": list(map(str, range(4))), "panels": list(range(4))}
            given = with_given_endpoints(item)
            self.assertEqual(given["panel_images"][0].item(), permutation[0])
            self.assertEqual(given["panel_images"][-1].item(), permutation[-1])
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in given.items()}
            for _ in range(3):
                batch = shuffle_middle_batch(batch)
                images = batch["panel_images"].flatten().tolist()
                self.assertEqual(images, batch["layout_features"].flatten().tolist())
                self.assertEqual(images, list(map(int, batch["dialog_texts"][0])))
                self.assertEqual(images, batch["panels"][0])
                self.assertEqual(images, batch["input_panel_indices"][0].tolist())
                self.assertEqual(batch["panel_images"][0, batch["target_order"][0]].flatten().tolist(), list(permutation))

    def test_metrics_exclude_both_known_panels(self):
        metrics = ordering_metrics(torch.tensor([[0, 2, 1, 3]]), torch.tensor([[0, 1, 2, 3]]))
        self.assertEqual(metrics, {"exact_match": 0.0, "position_accuracy": 0.0, "pairwise_accuracy": 0.0})

    def test_inference_never_receives_gt_or_original_metadata(self):
        class Probe:
            def predict(self, batch):
                self.keys = set(batch)
                return torch.tensor([[0, 1, 2, 3]])
        solver = Probe()
        OrderingPuzzle(solver).infer({"panel_images": torch.zeros(1, 4, 3, 4, 4),
            "target_order": torch.tensor([[0, 2, 1, 3]]), "panels": [[{"global_order": 1}]],
            "input_panel_indices": torch.arange(4).unsqueeze(0)})
        self.assertEqual(solver.keys, {"panel_images"})

    def test_pointer_losses_ignore_both_endpoints(self):
        from Solver.agent.mangazero_set2seq_solver import pointer_cross_entropy
        from Solver.agent.mangazero_qwen_vl_nonautoregressive import nonautoregressive_ordering_loss
        for function in (pointer_cross_entropy, nonautoregressive_ordering_loss):
            logits = torch.randn(1, 4, 4, requires_grad=True)
            function(logits, torch.tensor([[0, 2, 1, 3]])).backward()
            self.assertEqual(logits.grad[:, (0, -1)].abs().sum().item(), 0)
            self.assertEqual(logits.grad[:, :, (0, -1)].abs().sum().item(), 0)
            self.assertGreater(logits.grad[:, 1:-1, 1:-1].abs().sum().item(), 0)


class UnifiedTrainingTests(unittest.TestCase):
    def create_dataset(self, root):
        for index in range(4):
            Image.new("RGB", (16, 16), (index * 50,) * 3).save(root / f"{index}.png")
        record = {"sequence_id": "example", "panel_count": 4, "target_order": [2, 0, 3, 1],
                  "panels": [{"padded_path": f"{i}.png", "dialog_text": str(i)} for i in range(4)]}
        for split in ("train", "val", "test"):
            (root / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def run_cli_roundtrip(self, name, options):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            self.create_dataset(root)
            run = root / "run"
            args = ["--solver", name, "--dataset-dir", str(root), "--panel-count", "4",
                    "--device", "cpu", "--image-width", "16", "--image-height", "16",
                    "--epochs", "1", "--save-dir", str(run), "--model-options", json.dumps(options)]
            train_main(args)
            checkpoint = read_checkpoint(run / "last.pt")
            self.assertEqual(checkpoint["solver"], name)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue((run / "best.pt").is_file())
            self.assertTrue((run / "metrics.jsonl").is_file())
            test_main(["--solver", name, "--dataset-dir", str(root), "--checkpoint", str(run / "best.pt"), "--device", "cpu"])
            train_main(args + ["--epochs", "2", "--resume", str(run / "last.pt")])
            self.assertEqual(read_checkpoint(run / "last.pt")["epoch"], 2)

    def test_text_pointer_and_classifier_roundtrip(self):
        common = {"solver_type": "text", "d_model": 8, "text_feature_dim": 8,
                  "text_vocab_size": 32, "num_heads": 2, "encoder_layers": 1, "dropout": 0.0}
        self.run_cli_roundtrip("set2seq", {**common, "decoder_layers": 1})
        self.run_cli_roundtrip("permutation", common)

    def test_nar_trains_and_resumes_with_stubbed_backbone(self):
        transformers = types.SimpleNamespace(AutoProcessor=FakeProcessor, Qwen2_5_VLForConditionalGeneration=FakeQwen)
        vision = types.SimpleNamespace(process_vision_info=lambda _: ([], None))
        with patch.dict("sys.modules", {"transformers": transformers, "qwen_vl_utils": vision}):
            for name in ("qwen_vl_nonautoregressive", "qwen_vl_autoregressive", "qwen_vl_classifier"):
                options = {"use_lora": False, "torch_dtype": "float32"}
                if name != "qwen_vl_classifier":
                    options.update(decoder_dim=8, num_heads=2, decoder_layers=1, decoder_dropout=0.0)
                if name == "qwen_vl_nonautoregressive":
                    options["encoder_layers"] = 1
                with self.subTest(name=name):
                    self.run_cli_roundtrip(name, options)

    def test_jigsaw_and_made2order_roundtrip(self):
        self.run_cli_roundtrip("jigsaw", {"feature_dim": 8, "hidden_dim": 16, "dropout": 0.0})
        self.run_cli_roundtrip("made2order", {"patch_size": 8, "dim": 16, "heads": 2, "decoder_dim": 8})

    def test_generative_solver_uses_own_loss_predict_and_checkpoint(self):
        from Solver.solvers.generative import GenerativeSolver
        class FakeGenerative:
            def __init__(self):
                self.backbone = FakeQwen()
            def compute_loss(self, images, texts, target):
                self.target = target
                return self.backbone.embedding.weight.square().mean()
            def solve(self, images, texts):
                return types.SimpleNamespace(order=[0, 2, 1, 3])
            def trainable_state_dict(self):
                return self.backbone.state_dict()
            def load_trainable_state_dict(self, weights):
                self.backbone.load_state_dict(weights)
        model = FakeGenerative()
        solver = GenerativeSolver(model)
        batch = {"panel_images": torch.zeros(1, 4, 3, 8, 8), "target_order": torch.tensor([[0, 2, 1, 3]])}
        puzzle = OrderingPuzzle(solver)
        puzzle.training_loss(batch).backward()
        self.assertIsNotNone(model.backbone.embedding.weight.grad)
        self.assertEqual(puzzle.infer(batch).tolist(), [[0, 2, 1, 3]])
        solver.load_checkpoint_state(solver.checkpoint_state())

    def test_accumulation_matches_large_batch_with_partial_last_window(self):
        class RegressionSolver(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.2))
            def prepare_training_batch(self, batch):
                return batch
            def training_loss(self, batch):
                return ((batch["panel_images"] * self.weight - 1) ** 2).mean()
        def batch(values):
            return {"panel_images": torch.tensor(values), "target_order": torch.tensor([[0, 1, 2]] * len(values))}
        args = types.SimpleNamespace(device="cpu", grad_accum_steps=2, grad_clip_norm=0)
        micro, large = RegressionSolver(), RegressionSolver()
        optimizer = torch.optim.SGD(micro.parameters(), lr=0.1)
        train_epoch(micro, [batch([1., 2.]), batch([3.]), batch([4.])], optimizer, args)
        args.grad_accum_steps = 1
        train_epoch(large, [batch([1., 2., 3.]), batch([4.])], torch.optim.SGD(large.parameters(), lr=0.1), args)
        torch.testing.assert_close(micro.weight, large.weight)


if __name__ == "__main__":
    unittest.main()
