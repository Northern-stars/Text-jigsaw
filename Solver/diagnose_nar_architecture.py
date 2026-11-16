"""Offline structural probes; no pretrained Qwen weights or dataset required.

The random tiny Qwen text model demonstrates causal visibility only. Its inputs
stand in for already-embedded image/text tokens, not real visual features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from Solver.agent.mangazero_qwen_vl_nonautoregressive import (
    NonAutoregressivePointerDecoder,
    nonautoregressive_ordering_loss,
)


def causal_visibility_probe():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    config = Qwen2_5_VLTextConfig(
        vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rope_scaling={"type": "default", "mrope_section": [1, 1, 1]},
        use_cache=False, attention_dropout=0,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLTextModel(config).eval()
    # Prefix [0:2], image A [2:6], image B [6:10], trailing text [10:16].
    inputs = torch.randn(1, 16, 24)

    def encode(values):
        return model(inputs_embeds=values, return_dict=True).last_hidden_state

    with torch.no_grad():
        baseline = encode(inputs)
        suffix_changed = inputs.clone()
        suffix_changed[:, 10:] = torch.randn_like(suffix_changed[:, 10:])
        suffix_hidden = encode(suffix_changed)
        prefix_changed = inputs.clone()
        prefix_changed[:, :2] = torch.randn_like(prefix_changed[:, :2])
        prefix_hidden = encode(prefix_changed)
        later_image_changed = inputs.clone()
        later_image_changed[:, 6:10] = torch.randn_like(later_image_changed[:, 6:10])
        later_hidden = encode(later_image_changed)
        earlier_image_changed = inputs.clone()
        earlier_image_changed[:, 2:6] = torch.randn_like(earlier_image_changed[:, 2:6])
        earlier_hidden = encode(earlier_image_changed)

    differentiable_inputs = inputs.clone().requires_grad_()
    pooled_images = encode(differentiable_inputs)[:, 2:10].mean(1)
    # A random linear probe avoids a constant objective induced by final norm.
    objective = (pooled_images * torch.randn_like(pooled_images)).sum()
    gradient, = torch.autograd.grad(objective, differentiable_inputs)
    return {
        "limits": "Random tiny Qwen2.5-VL text model, synthetic embeddings; no vision tower or real features.",
        "image_hidden_change_when_suffix_text_changes": (baseline[:, 2:10] - suffix_hidden[:, 2:10]).abs().max().item(),
        "image_hidden_change_when_prefix_text_changes": (baseline[:, 2:10] - prefix_hidden[:, 2:10]).abs().max().item(),
        "earlier_image_change_when_later_image_changes": (baseline[:, 2:6] - later_hidden[:, 2:6]).abs().max().item(),
        "later_image_change_when_earlier_image_changes": (baseline[:, 6:10] - earlier_hidden[:, 6:10]).abs().max().item(),
        "suffix_embedding_gradient_from_image_readout": gradient[:, 10:].norm().item(),
        "prefix_embedding_gradient_from_image_readout": gradient[:, :2].norm().item(),
    }


@torch.no_grad()
def decoder_probe(checkpoint=None):
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("solver") != "qwen_vl_nonautoregressive":
            raise ValueError("Expected qwen_vl_nonautoregressive checkpoint")
        head = state["model"]["weights"]["decoder"]
        options = dict(state["config"].get("model_options") or {})
        for key in ("encoder_layers", "decoder_layers", "num_heads"):
            if state["config"].get(key) is not None:
                options[key] = state["config"][key]
        count = state["config"]["panel_count"]
        dimension = head["panel_projection.weight"].shape[0]
        decoder = NonAutoregressivePointerDecoder(
            count, dimension, encoder_layers=options.get("encoder_layers", 2),
            decoder_layers=options.get("decoder_layers", 1),
            num_heads=options.get("num_heads", 8), dropout=options.get("decoder_dropout", .1))
        decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in head.items() if k.startswith("decoder.")})
    else:
        count, dimension = 6, 256
        decoder = NonAutoregressivePointerDecoder(count, dimension)
    decoder.eval()
    memory = F.layer_norm(torch.randn(4, count, dimension), (dimension,))
    baseline = decoder(memory)
    swapped = memory.clone()
    swapped[:, [0, count - 1]] = swapped[:, [count - 1, 0]]
    endpoint_difference = (baseline[:, 1:-1, 1:-1] - decoder(swapped)[:, 1:-1, 1:-1]).abs().max().item()
    encoded = decoder.set_encoder(memory)
    decoded = decoder.position_decoder(decoder.position_queries.expand(memory.size(0), -1, -1), encoded)
    projected = decoder.query_projection(decoder.output_norm(decoded))
    cosine = F.normalize(projected[:, 1:-1], dim=-1)
    mask = ~torch.eye(count - 2, dtype=torch.bool)
    return {
        "source": str(checkpoint) if checkpoint else "random initialization",
        "endpoint_swap_middle_logits_max_difference": endpoint_difference,
        "position_query_parameter_rms": decoder.position_queries.square().mean().sqrt().item(),
        "projected_middle_query_cosine_mean": (cosine @ cosine.transpose(-2, -1))[:, mask].mean().item(),
        "limit": "Decoder-only endpoint swap; real Qwen features can implicitly encode input positions.",
    }


def tied_position_loss_probe():
    # In the collapsed limit all positions share a panel preference vector.
    # A valid permutation contains each panel exactly once, so a uniform shared
    # vector minimizes this restricted objective even on one labelled example.
    target = torch.tensor([[0, 3, 1, 4, 2, 5]])
    preference = torch.zeros(1, 6, requires_grad=True)
    logits = preference[:, None, :].expand(1, 6, 6)
    loss = nonautoregressive_ordering_loss(logits, target)
    gradient, = torch.autograd.grad(loss, preference)
    return {"loss": loss.item(), "uniform_baseline": 2 * math.log(4),
            "tied_panel_preference_gradient_norm": gradient.norm().item(),
            "limit": "Exactly tied positions are an analytical limiting case, not a claim that saved parameters are identical."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/nar_architecture_diagnosis.json"))
    args = parser.parse_args()
    if args.checkpoint and args.output.resolve() == args.checkpoint.resolve():
        parser.error("Output must not overwrite checkpoint")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report = {"causal_visibility": causal_visibility_probe(),
              "fresh_decoder": decoder_probe(), "tied_positions": tied_position_loss_probe()}
    if args.checkpoint:
        report["checkpoint_decoder"] = decoder_probe(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
