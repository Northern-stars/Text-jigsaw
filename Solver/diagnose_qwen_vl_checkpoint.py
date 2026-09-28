"""Diagnose whether a MangaZero Qwen-VL LoRA autoregressive checkpoint loads correctly.

Run this on the machine that has both the checkpoint and the Qwen model:

    python Solver/diagnose_qwen_vl_checkpoint.py \
        --checkpoint /path/to/epoch_0010.pt \
        --dataset-dir /path/to/ordering_output \
        --panel-count 6

Checks performed (no training):
1.  checkpoint structure (full_state vs trainable_state_dict)
2.  decoder weights actually changed after load_checkpoint
3.  LoRA weights actually landed inside the PEFT backbone
4.  predict_order output changes after loading (proves weights affect inference)
5.  active adapter state and trainable parameter counts
6.  exact tensor parity between checkpoint and live model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(".")

from torch.utils.data import DataLoader

from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import parse_split_ratio, split_dataset
from Solver.train_mangazero_qwen_vl_autoregressive import build_model, load_checkpoint
from Solver.test_mangazero_qwen_vl_autoregressive import with_model_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Qwen-VL LoRA checkpoint loading.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional. When given, run one real inference batch before and after load.",
    )
    p.add_argument("--panel-count", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_checkpoint_file(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def snapshot_decoder(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("backbone.")
    }


def snapshot_lora(model: Any) -> dict[str, torch.Tensor]:
    try:
        from peft import get_peft_model_state_dict
        source = get_peft_model_state_dict(model.backbone)
    except ImportError:
        source = {
            name: tensor
            for name, tensor in model.backbone.state_dict().items()
            if "lora_" in name
        }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in source.items()
        if isinstance(tensor, torch.Tensor)
    }


def count_changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[int, int, int]:
    shared = sorted(set(before) & set(after))
    changed = 0
    unchanged = 0
    for name in shared:
        if before[name].shape != after[name].shape:
            changed += 1
            continue
        if bool((before[name].float() == after[name].float()).all()):
            unchanged += 1
        else:
            changed += 1
    return len(shared), changed, unchanged


def report_parity(label: str, before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
    shared, changed, unchanged = count_changed(before, after)
    print(f"[{label}] shared={shared} changed={changed} unchanged={unchanged}")
    if shared == 0:
        print(f"[{label}] FAIL: no shared keys between the two snapshots")
    for name in sorted(set(before) & set(after))[:10]:
        a = before[name].float()
        b = after[name].float()
        diff = (a - b).abs().max().item()
        if diff > 0.0:
            print(f"  changed {name}: absmax_diff={diff:.6g}")


def compare_with_checkpoint(model: Any, checkpoint_model: dict[str, Any]) -> None:
    print("\n=== Checkpoint vs live model ===")
    decoder_source = checkpoint_model.get("decoder", {})
    backbone_source = checkpoint_model.get("backbone", {})

    if not decoder_source and not backbone_source:
        print("checkpoint model is a raw state_dict; skipping structured parity check")
        return

    live_decoder = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith("backbone.")
    }
    shared, exact, differ = count_live_against_source(live_decoder, decoder_source)
    print(f"decoder: live={len(live_decoder)} ckpt={len(decoder_source)} "
          f"shared={shared} exact={exact} differ={differ}")
    missing = sorted(set(live_decoder) - set(decoder_source))
    print(f"  keys_missing_from_checkpoint={len(missing)} {missing[:10]}")

    live_lora = snapshot_lora(model)
    shared, exact, differ = count_live_against_source(live_lora, backbone_source)
    print(f"backbone_lora: live={len(live_lora)} ckpt={len(backbone_source)} "
          f"shared={shared} exact={exact} differ={differ}")
    ckpt_only = sorted(set(backbone_source) - set(live_lora))
    print(f"  keys_only_in_checkpoint={len(ckpt_only)} {ckpt_only[:10]}")


def count_live_against_source(
    live: dict[str, torch.Tensor],
    source: dict[str, Any],
) -> tuple[int, int, int]:
    shared = 0
    exact = 0
    differ = 0
    for name in sorted(set(live) & set(source)):
        shared += 1
        live_tensor = live[name]
        source_tensor = source[name]
        if not isinstance(source_tensor, torch.Tensor):
            differ += 1
            continue
        source_tensor = source_tensor.to(live_tensor.device)
        if source_tensor.shape == live_tensor.shape and bool((source_tensor.float() == live_tensor.float()).all()):
            exact += 1
        else:
            differ += 1
    return shared, exact, differ


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"torch={torch.__version__} device={device}")

    checkpoint = load_checkpoint_file(args.checkpoint, device)
    print("\n=== Checkpoint structure ===")
    print("top_level_keys:", sorted(checkpoint.keys()))
    print("solver_family:", checkpoint.get("solver_family"))
    print("full_state:", checkpoint.get("full_state"))
    print("epoch:", checkpoint.get("epoch"))

    raw_config = checkpoint.get("config", {})
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    print("config_keys:", sorted(config.keys()))
    for key in (
        "panel_count",
        "model_name",
        "torch_dtype",
        "attn_implementation",
        "min_pixels",
        "max_pixels",
        "load_in_4bit",
        "device_map",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "decoder_layers",
        "num_heads",
        "decoder_dim",
        "decoder_dropout",
        "image_width",
        "image_height",
        "split_ratio",
        "save_full_state",
    ):
        print(f"  {key}: {config.get(key)}")

    model_state = checkpoint.get("model", {})
    if isinstance(model_state, dict) and ("decoder" in model_state or "backbone" in model_state):
        decoder_state = model_state.get("decoder", {})
        backbone_state = model_state.get("backbone", {})
        print(f"model=trainable_state_dict decoder_keys={len(decoder_state)} backbone_keys={len(backbone_state)}")
        for name in list(decoder_state)[:8]:
            tensor = decoder_state[name]
            print(f"  decoder[{name}] shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        for name in list(backbone_state)[:8]:
            tensor = backbone_state[name]
            print(f"  backbone[{name}] shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    elif isinstance(model_state, dict):
        print(f"model=raw_state_dict keys={len(model_state)}")
        print("  first_keys:", list(model_state)[:12])
    else:
        print(f"model type is {type(model_state).__name__}, which is unexpected")

    merged_config = with_model_defaults(config)
    model_args = argparse.Namespace(**merged_config)
    model_args.panel_count = args.panel_count
    model_args.load = None

    print("\n=== Building model from checkpoint config ===")
    model = build_model(model_args)
    load_in_4bit = bool(merged_config.get("load_in_4bit", False))
    if not load_in_4bit:
        model = model.to(device)
    else:
        model.panel_projection.to(device)
        model.pointer_decoder.to(device)

    print("active_adapter_before_load:", getattr(model.backbone, "active_adapter", "n/a"))
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_lora = [name for name in trainable_names if "lora_" in name]
    print(f"trainable_before_load: total={len(trainable_names)} lora={len(trainable_lora)}")

    batch = None
    if args.dataset_dir is not None:
        split_ratio = parse_split_ratio(str(merged_config.get("split_ratio", "0.8,0.1,0.1")))
        image_width = int(merged_config.get("image_width", 224))
        image_height = int(merged_config.get("image_height", 224))
        dataset = MangaZeroPanelOrderingDataset(
            args.dataset_dir,
            split="all",
            image_size=(image_width, image_height),
            use_layout=False,
        )
        train_dataset, _, _ = split_dataset(dataset, split_ratio, args.seed)
        loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_panel_ordering_batch,
        )
        batch = next(iter(loader))
        print("\n=== Real batch loaded ===")
        print("panel_images:", tuple(batch["panel_images"].shape))
        print("target_order:", batch["target_order"].tolist())
        print("sequence_id:", batch["sequence_id"])

    decoder_before = snapshot_decoder(model)
    lora_before = snapshot_lora(model)

    predict_before = None
    memory_before = None
    if batch is not None:
        model.eval()
        images = batch["panel_images"].to(device)
        with torch.no_grad():
            memory_before = model.encode_panels(images, dialog_texts=batch["dialog_texts"])
            predict_before = model.predict_order(images, dialog_texts=batch["dialog_texts"])
        print("predict_before_load:", predict_before.tolist())
        print("memory_norm_before_load:", memory_before.float().norm(dim=-1).mean().item())

    print("\n=== load_checkpoint ===")
    epoch = load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    print("load_checkpoint_epoch:", epoch)

    decoder_after = snapshot_decoder(model)
    lora_after = snapshot_lora(model)

    report_parity("decoder before->after", decoder_before, decoder_after)
    report_parity("lora before->after", lora_before, lora_after)
    compare_with_checkpoint(model, model_state)

    print("\n=== Post-load state ===")
    print("active_adapter_after_load:", getattr(model.backbone, "active_adapter", "n/a"))
    try:
        from peft import get_peft_model_state_dict
        current_lora = get_peft_model_state_dict(model.backbone)
        names = [name for name, tensor in current_lora.items() if isinstance(tensor, torch.Tensor)]
        nonzero = [
            name for name in names
            if current_lora[name].float().abs().max().item() > 0.0
        ]
        print(f"live_lora_tensors={len(names)} nonzero={len(nonzero)}")
        if not nonzero:
            print("FAIL: all LoRA tensors are exactly zero; LoRA has no effect on the backbone")
    except ImportError as exc:
        print("peft unavailable:", exc)

    if batch is not None:
        model.eval()
        images = batch["panel_images"].to(device)
        with torch.no_grad():
            memory_after = model.encode_panels(images, dialog_texts=batch["dialog_texts"])
            predict_after = model.predict_order(images, dialog_texts=batch["dialog_texts"])
        print("predict_after_load:", predict_after.tolist())
        print("memory_norm_after_load:", memory_after.float().norm(dim=-1).mean().item())
        if memory_before is not None and memory_after is not None:
            delta = (memory_before.float() - memory_after.float()).abs().max().item()
            print("memory_delta_absmax:", delta)
            print("prediction_changed_by_load:", bool((predict_before != predict_after).any().item()))
        target = batch["target_order"].to(predict_after.device)
        hits = int((predict_after == target).sum().item())
        print(f"single_batch_position_hits={hits}/{target.numel()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
