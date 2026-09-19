"""Evaluate a Qwen2.5-VL + LoRA autoregressive MangaZero ordering checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import format_split_metrics, parse_split_ratio, split_dataset
from Solver.train_mangazero_qwen_vl_autoregressive import build_model, evaluate, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a Qwen2.5-VL + LoRA autoregressive MangaZero orderer.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "val", "test"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-ratio", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint_file(args.checkpoint, device)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config = with_model_defaults(config)
    model_args = argparse.Namespace(**config)
    model_args.panel_count = args.panel_count
    model_args.load = None

    split_ratio = parse_split_ratio(str(args.split_ratio or config.get("split_ratio", "0.8,0.1,0.1")))
    image_width = int(config.get("image_width", 224))
    image_height = int(config.get("image_height", 224))
    batch_size = int(args.batch_size or config.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 0))

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(image_width, image_height),
        use_layout=False,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, split_ratio, args.seed)
    split_name = "val" if args.split == "valid" else args.split
    target_dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }[split_name]
    if len(target_dataset) == 0:
        raise ValueError(f"Selected split {args.split!r} is empty")

    loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_panel_ordering_batch,
    )
    model = build_model(model_args)
    if not bool(getattr(model_args, "load_in_4bit", False)):
        model = model.to(device)
    else:
        model.panel_projection.to(device)
        model.pointer_decoder.to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    metrics = evaluate(model, loader, device)
    print(format_split_metrics(split_name, metrics))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": str(args.checkpoint),
                    "dataset_dir": str(args.dataset_dir),
                    "split": split_name,
                    "seed": args.seed,
                    "metrics": metrics,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def load_checkpoint_file(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def with_model_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "min_pixels": None,
        "max_pixels": 512 * 28 * 28,
        "load_in_4bit": False,
        "device_map": None,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "decoder_layers": 2,
        "num_heads": 8,
        "decoder_dim": 256,
        "decoder_dropout": 0.1,
    }
    merged = dict(defaults)
    merged.update(config)
    return merged


if __name__ == "__main__":
    main()
