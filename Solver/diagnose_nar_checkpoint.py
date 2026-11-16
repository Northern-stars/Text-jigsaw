"""Inspect a NAR checkpoint on CPU without downloading Qwen or reading data.

Synthetic probes exercise saved projection/decoder weights, not actual comic
features. Neither checkpoint nor optimizer state is modified on disk.
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
from torch import nn
from torch.nn import functional as F

from Solver.agent.mangazero_qwen_vl_nonautoregressive import (
    NonAutoregressivePointerDecoder,
    nonautoregressive_ordering_loss,
)


def stats(tensor):
    dtype = str(tensor.dtype)
    tensor = tensor.detach().float()
    return {"shape": list(tensor.shape), "dtype": dtype,
            "finite": bool(tensor.isfinite().all()), "norm": tensor.norm().item(),
            "mean": tensor.mean().item(), "std": tensor.std(unbiased=False).item(),
            "abs_max": tensor.abs().max().item(),
            "nonzero_fraction": tensor.ne(0).float().mean().item()}


def off_diagonal_cosine(values):
    values = F.normalize(values.detach().float(), dim=-1)
    similarities = values @ values.transpose(-2, -1)
    mask = ~torch.eye(values.size(-2), dtype=torch.bool, device=values.device)
    selected = similarities[..., mask]
    return {"mean": selected.mean().item(), "min": selected.min().item(), "max": selected.max().item()}


@torch.no_grad()
def probe(decoder, memory):
    encoded = decoder.set_encoder(memory)
    decoded = decoder.position_decoder(decoder.position_queries.expand(memory.size(0), -1, -1), encoded)
    queries = decoder.query_projection(decoder.output_norm(decoded))
    keys = decoder.key_projection(memory)
    preactivation = queries.unsqueeze(2) + keys.unsqueeze(1)
    activation = preactivation.tanh()
    scores = decoder.score_projection(activation).squeeze(-1)[:, 1:-1, 1:-1]
    probabilities = scores.softmax(dim=-1)
    # Double centering removes row-only and column-only preferences. The
    # residual measures the position-panel interaction needed for ordering.
    interaction = scores - scores.mean(1, keepdim=True) - scores.mean(2, keepdim=True) + scores.mean((1, 2), keepdim=True)
    return {
        "memory_cosine": off_diagonal_cosine(memory[:, 1:-1]),
        "decoded_position_cosine": off_diagonal_cosine(decoded[:, 1:-1]),
        "query_cosine": off_diagonal_cosine(queries[:, 1:-1]),
        "query_vector_norm": queries.norm(dim=-1).mean().item(),
        "key_vector_norm": keys.norm(dim=-1).mean().item(),
        "preactivation_abs_mean": preactivation.abs().mean().item(),
        "tanh_abs_gt_3_fraction": preactivation.abs().gt(3).float().mean().item(),
        "tanh_abs_gt_5_fraction": preactivation.abs().gt(5).float().mean().item(),
        "tanh_derivative_mean": (1 - activation.square()).mean().item(),
        "row_logits_std": scores.std(-1, unbiased=False).mean().item(),
        "column_logits_std": scores.std(1, unbiased=False).mean().item(),
        "interaction_rms": interaction.square().mean().sqrt().item(),
        "row_entropy": -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean().item(),
        "first_middle_scores": scores[0].tolist(),
        "example_predictions": decoder.predict_order(memory)[:4].tolist(),
    }


def diagnose(checkpoint, *, steps=100, seed=42):
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("solver") != "qwen_vl_nonautoregressive":
        raise ValueError("Expected a unified qwen_vl_nonautoregressive checkpoint")
    config = state["config"]
    weights = state["model"]["weights"]
    head = weights["decoder"]
    count = config["panel_count"]
    options = config["model_options"]
    dimension = head["panel_projection.weight"].size(0)
    hidden_size = head["panel_projection.weight"].size(1)

    def make_decoder(load=True):
        torch.manual_seed(seed)
        decoder = NonAutoregressivePointerDecoder(
            count, dimension, encoder_layers=options.get("encoder_layers", 2),
            decoder_layers=options.get("decoder_layers", 1),
            num_heads=options.get("num_heads", 8), dropout=options.get("decoder_dropout", 0.1))
        if load:
            decoder.load_state_dict({key.removeprefix("decoder."): value for key, value in head.items() if key.startswith("decoder.")}, strict=True)
        return decoder

    decoder = make_decoder().eval()
    fresh = make_decoder(False).eval()
    norm = nn.LayerNorm(dimension)
    norm.load_state_dict({key.removeprefix("panel_norm."): value for key, value in head.items() if key.startswith("panel_norm.")})
    projection = nn.Linear(hidden_size, dimension)
    projection.load_state_dict({key.removeprefix("panel_projection."): value for key, value in head.items() if key.startswith("panel_projection.")})

    all_weights = {f"{kind}.{key}": value for kind, values in weights.items() for key, value in values.items()}
    invalid = [key for key, value in all_weights.items() if not value.isfinite().all()]
    groups = []
    for group in state["optimizer"]["param_groups"]:
        saved_states = [state["optimizer"]["state"][index] for index in group["params"] if index in state["optimizer"]["state"]]
        step_values = [int(value["step"]) for value in saved_states]
        groups.append({"lr": group["lr"], "parameters": len(group["params"]),
                       "states": len(saved_states), "step_min": min(step_values), "step_max": max(step_values),
                       "nonfinite_moments": sum(not bool(v["exp_avg"].isfinite().all() and v["exp_avg_sq"].isfinite().all()) for v in saved_states),
                       "first_moment_l2": math.sqrt(sum(float(v["exp_avg"].float().square().sum()) for v in saved_states))})
    # Optimizer decoder group follows named_parameters order. Verify every
    # moment shape before attaching names to the saved optimizer state.
    head_names = list(head)
    head_group = state["optimizer"]["param_groups"][-1]
    moments = {}
    if len(head_group["params"]) == len(head_names):
        for name, index in zip(head_names, head_group["params"]):
            saved = state["optimizer"]["state"][index]
            if saved["exp_avg"].shape != head[name].shape:
                raise ValueError(f"Optimizer/head parameter order mismatch at {name}")
            moments[name] = {"exp_avg_norm": saved["exp_avg"].float().norm().item(),
                             "exp_avg_sq_norm": saved["exp_avg_sq"].float().norm().item()}
    lora_b = {key: value for key, value in weights["backbone"].items() if "lora_B" in key}
    report = {"checkpoint": str(checkpoint), "epoch": state["epoch"], "best_val": state["best_val"],
              "config": config, "limits": "No real dataset or Qwen base weights; all activations below use synthetic memory.",
              "random_baseline": {"loss": 2 * math.log(count - 2), "exact_match": 1 / math.factorial(count - 2),
                                  "position_accuracy": 1 / (count - 2), "pairwise_accuracy": 0.5},
              "integrity": {"tensors": len(all_weights), "nonfinite_weights": invalid,
                            "lora_B_tensors": len(lora_b), "lora_B_nonzero_tensors": sum(bool(t.ne(0).any()) for t in lora_b.values())},
              "optimizer": groups, "head_parameter_stats": {key: stats(value) for key, value in head.items()},
              "head_optimizer_moments": moments,
              "position_parameter_cosine": off_diagonal_cosine(decoder.position_queries[:, 1:-1]),
              "probes": {}, "synthetic_overfit": {}}
    torch.manual_seed(seed + 1)
    memory = norm(F.layer_norm(torch.randn(16, count, dimension), (dimension,))).detach()
    raw_hidden = torch.randn(16, count, hidden_size)
    raw_projected = norm(projection(raw_hidden)).detach()
    for name, model, inputs in (("fresh_normalized_memory", fresh, memory),
                                ("checkpoint_normalized_memory", decoder, memory),
                                ("checkpoint_projected_isotropic_hidden", decoder, raw_projected)):
        report["probes"][name] = probe(model, inputs)

    # Same non-uniform targets/features for checkpoint vs fresh. Full decoder
    # only: Qwen and projection are absent, and all updates remain in memory.
    torch.manual_seed(seed + 2)
    target = torch.stack([torch.cat([torch.tensor([0]), torch.randperm(count - 2) + 1, torch.tensor([count - 1])]) for _ in range(memory.size(0))])
    for name, model in (("fresh", fresh), ("checkpoint", decoder)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        trajectory = []
        model.train()
        for step in range(steps + 1):
            optimizer.zero_grad()
            logits = model(memory)
            loss = nonautoregressive_ordering_loss(logits, target)
            loss.backward()
            if step in (0, 10, 50, steps):
                trajectory.append({"step": step, "loss": loss.item(),
                                   "query_parameter_grad_norm": model.position_queries.grad.norm().item(),
                                   "score_weight_grad_norm": model.score_projection.weight.grad.norm().item()})
            if step < steps:
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            report["synthetic_overfit"][name] = {"trajectory": trajectory,
                "exact_match": model.predict_order(memory).eq(target).all(1).float().mean().item()}

    metrics_path = checkpoint.parent / "metrics.jsonl"
    if metrics_path.is_file():
        report["training_history"] = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output or args.checkpoint.parent / "diagnosis.json"
    if output.resolve() == args.checkpoint.resolve():
        parser.error("Output must not overwrite the checkpoint")
    report = diagnose(args.checkpoint, steps=args.steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("epoch", "integrity", "optimizer", "position_parameter_cosine", "probes", "synthetic_overfit")}, indent=2))
    print(f"Full report: {output}")


if __name__ == "__main__":
    main()
