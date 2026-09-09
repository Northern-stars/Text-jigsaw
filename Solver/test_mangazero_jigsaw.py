"""Evaluate a MangaZero self-supervised Jigsaw-style checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from Solver.agent.mangazero_jigsaw_solver import MangaZeroJigsawSolver
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_jigsaw import evaluate
from Solver.train_mangazero_panel_ordering import format_split_metrics, parse_split_ratio, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a MangaZero Jigsaw-style checkpoint.")
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

    split_ratio = parse_split_ratio(str(args.split_ratio or config.get("split_ratio", "0.8,0.1,0.1")))
    use_layout = bool(config.get("use_layout", False))
    image_width = int(config.get("image_width", 224))
    image_height = int(config.get("image_height", 224))
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 0))

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(image_width, image_height),
        use_layout=use_layout,
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
    model = build_model(args.panel_count, config, use_layout).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device, namespace_from_config(config))
    print(format_split_metrics(split_name, metrics))
    print(f"{split_name}_class_acc={metrics['class_accuracy']:.4f}")
    print(f"{split_name}_candidate_coverage={metrics['candidate_coverage']:.4f}")

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


def namespace_from_config(config: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(use_layout=bool(config.get("use_layout", False)))


def load_checkpoint_file(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def build_model(
    panel_count: int,
    config: dict[str, Any],
    use_layout: bool,
) -> MangaZeroJigsawSolver:
    return MangaZeroJigsawSolver(
        panel_count=panel_count,
        max_permutations=int(config.get("max_permutations", 1000)),
        permutation_seed=int(config.get("permutation_seed", 0)),
        feature_dim=int(config.get("feature_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        use_layout=use_layout,
        dropout=float(config.get("dropout", 0.5)),
    )


if __name__ == "__main__":
    main()
