"""Evaluate a trained Made2Order panel-ordering checkpoint on a chosen split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from Solver.agent.made2order_set2order_solver import Made2OrderPanelOrderingSolver
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_made2order_panel_ordering import evaluate, format_split_metrics, parse_split_ratio, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a Made2Order panel-ordering checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "val", "test"))
    parser.add_argument("--seed", type=int, default=0, help="Must match the training seed for the same split.")
    parser.add_argument("--split-ratio", default=None, help="Defaults to the checkpoint config or 0.8,0.1,0.1.")
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

    split_ratio = parse_split_ratio(str(args.split_ratio or config.get("split_ratio", "0.8,0.1,0.1")))
    image_width = int(config.get("image_width", 224))
    image_height = int(config.get("image_height", 224))
    patch_size = int(config.get("patch_size", 16))
    batch_size = int(args.batch_size or config.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 0))

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(image_width, image_height),
        use_layout=False,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, split_ratio, args.seed)
    split_name = "val" if args.split == "valid" else args.split
    split_map = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    target_dataset = split_map[split_name]
    if len(target_dataset) == 0:
        raise ValueError(f"Selected split {args.split!r} is empty")

    print(
        "dataset_split "
        f"train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"test={len(test_dataset)} "
        f"selected={split_name} "
        f"seed={args.seed}"
    )

    loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_panel_ordering_batch,
    )
    model = build_model(args.panel_count, config).to(device)
    model.load_state_dict(checkpoint["model"])

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
                    "split_ratio": list(split_ratio),
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


def build_model(panel_count: int, config: dict[str, Any]) -> Made2OrderPanelOrderingSolver:
    return Made2OrderPanelOrderingSolver(
        panel_count=panel_count,
        image_size=(
            int(config.get("image_height", 224)),
            int(config.get("image_width", 224)),
        ),
        patch_size=int(config.get("patch_size", 16)),
        dim=int(config.get("dim", 256)),
        heads=int(config.get("heads", 4)),
        decoder_dim=int(config.get("decoder_dim", 64)),
    )


if __name__ == "__main__":
    main()
