"""Tests for the Qwen-VL non-autoregressive decoder without model downloads."""

import json
import re
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from Solver.agent.mangazero_qwen_vl_nonautoregressive import (
    MangaZeroQwenVLNonAutoregressiveSolver,
    NonAutoregressivePointerDecoder,
    nonautoregressive_ordering_loss,
)


class RecordingProcessor:
    """Serialize the actual message into tiny text/image token sequences."""

    def __init__(self):
        self.conversations = []
        self.vocabulary = {}
        self.tokenizer = types.SimpleNamespace(convert_tokens_to_ids=lambda _: 1, unk_token_id=0)

    def apply_chat_template(self, conversation, **kwargs):
        self.conversations.append(conversation)
        tokens = []
        for item in conversation[0]["content"]:
            if item["type"] == "image":
                tokens.extend([2, 1, 1, 3])
            else:
                for word in re.findall(r"\w+|[^\w\s]", item["text"]):
                    tokens.append(self.vocabulary.setdefault(word, len(self.vocabulary) + 4))
        return json.dumps(tokens)

    def __call__(self, text, images, **kwargs):
        self.images = images
        rows = [json.loads(value) for value in text]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])}


def input_only_model():
    model = MangaZeroQwenVLNonAutoregressiveSolver.__new__(MangaZeroQwenVLNonAutoregressiveSolver)
    nn.Module.__init__(model)
    model.panel_count = 4
    model.processor = RecordingProcessor()
    model.backbone = nn.Module()
    model.backbone.get_input_embeddings = lambda: nn.Embedding(1, 1)
    model._image_pad_token_id = None
    return model


class NonAutoregressiveInputTests(unittest.TestCase):
    def test_all_text_precedes_images_and_slot_mapping_survives_both_processor_paths(self):
        images = torch.stack([torch.full((3, 4, 4), index / 3) for index in range(4)])
        batch = images.unsqueeze(0).repeat(2, 1, 1, 1, 1)
        dialogue = [[f"sample{b}_slot{i}" for i in range(4)] for b in range(2)]
        seen = []

        def process_vision_info(conversations):
            seen.extend(conversations)
            return [item["image"] for c in conversations for item in c[0]["content"] if item["type"] == "image"], None

        for module in (None, types.SimpleNamespace(process_vision_info=process_vision_info)):
            with self.subTest(vision_utils=module is not None), patch.dict("sys.modules", {"qwen_vl_utils": module}):
                model = input_only_model()
                output = model.prepare_inputs(batch, dialogue)
                self.assertEqual(output["input_ids"].shape[0], 2)
                for b, conversation in enumerate(model.processor.conversations):
                    content = conversation[0]["content"]
                    prefix = content[0]["text"]
                    self.assertIn("known opening", prefix)
                    self.assertIn("known ending", prefix)
                    for i in range(4):
                        self.assertIn(f"panel {i}: sample{b}_slot{i}", prefix)
                        self.assertEqual(content[1 + 2 * i]["text"].strip(), f"Panel {i}:")
                        image = content[2 + 2 * i]["image"]
                        self.assertEqual(image.getpixel((0, 0)), (i * 85,) * 3)
                    self.assertEqual(content[-1]["type"], "image")
        self.assertEqual(len(seen), 2)

    def test_missing_text_and_misaligned_text_batches(self):
        model = input_only_model()
        images = torch.zeros(1, 4, 3, 4, 4)
        for dialogue in (None, [], [["", "", "", ""]]):
            with patch.dict("sys.modules", {"qwen_vl_utils": None}):
                model.prepare_inputs(images, dialogue)
            self.assertNotIn("OCR text", model.processor.conversations[-1][0]["content"][0]["text"])
        for dialogue in ([["one"]], [[""] * 4, [""] * 4]):
            with self.assertRaisesRegex(ValueError, "one text per input panel"):
                model.prepare_inputs(images, dialogue)

    def test_last_panel_ocr_affects_all_pooled_panels_with_real_causal_qwen_layers(self):
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

        # No pretrained weights or vision tower. Real causal Qwen text layers
        # exercise prepare_inputs -> backbone -> image pooling -> projection.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = input_only_model()
            config = Qwen2_5_VLTextConfig(
                vocab_size=512, hidden_size=24, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=512, use_cache=False, attention_dropout=0,
                rope_scaling={"type": "default", "mrope_section": [1, 1, 1]})
            config._attn_implementation = "eager"
            model.backbone = Qwen2_5_VLTextModel(config)
            model.panel_projection = nn.Linear(24, 24)
            model.panel_norm = nn.LayerNorm(24)
        model.eval()
        images = torch.zeros(1, 4, 3, 4, 4)
        with patch.dict("sys.modules", {"qwen_vl_utils": None}):
            initial = model.encode_panels(images, [["", "", "", "alpha"]])
            changed = model.encode_panels(images, [["", "", "", "omega"]])
        self.assertEqual(tuple(initial.shape), (1, 4, 24))
        self.assertTrue(torch.all((initial - changed).abs().amax(-1) > 1e-7))
        initial[0, 0, 0].backward()
        ocr_token = model.processor.vocabulary["alpha"]
        self.assertGreater(model.backbone.get_input_embeddings().weight.grad[ocr_token].norm().item(), 0)


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
        target = torch.tensor([[0, 2, 1, 3], [0, 1, 2, 3]])
        loss = nonautoregressive_ordering_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_invalid_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "permutations"):
            nonautoregressive_ordering_loss(torch.randn(1, 3, 3), torch.tensor([[0, 0, 1]]))


if __name__ == "__main__":
    unittest.main()
