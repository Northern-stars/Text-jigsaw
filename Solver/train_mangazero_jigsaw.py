"""Train the MangaZero self-supervised Jigsaw-style ordering solver."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(".")

from Solver.agent.mangazero_jigsaw_solver import (
    MangaZeroJigsawSolver,
    gather_by_order,
    inverse_permutation,
    jigsaw_cross_entropy,
    recover_canonical_batch,
)
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import (
    format_split_metrics,
    ordering_metrics,
    parse_split_ratio,
    save_metrics_json,
    split_dataset,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MangaZero with a Jigsaw-style permutation classification task.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--test-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--max-permutations", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--use-layout", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path, default=Path("Solver/checkpoints_mangazero_jigsaw"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is not None:
        args.epoch = args.epochs
    if args.test_per_epoch <= 0:
        raise ValueError("--test-per-epoch must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split_ratio = parse_split_ratio(args.split_ratio)

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(args.image_width, args.image_height),
        use_layout=args.use_layout,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, split_ratio, args.seed)
    print(
        "dataset_split "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)
    model = build_model(args).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if args.load is not None:
        start_epoch = load_checkpoint(args.load, model, optimizer, device) + 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)

    best_val = -1.0
    for epoch in range(start_epoch, args.epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device, args) if val_loader is not None else {}
        test_metrics = None
        if test_loader is not None and epoch % args.test_per_epoch == 0:
            test_metrics = evaluate(model, test_loader, device, args)
        print(format_jigsaw_metrics(epoch, args.epoch, train_metrics, val_metrics, test_metrics))

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args)
        if val_metrics and val_metrics["exact_match"] >= best_val:
            best_val = val_metrics["exact_match"]
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, args)
        if test_metrics is not None:
            save_metrics_json(args.save_dir / f"test_epoch_{epoch:04d}.json", test_metrics)

    save_checkpoint(args.save_dir / "last.pt", model, optimizer, args.epoch, args)
    test_metrics = evaluate(model, test_loader, device, args) if test_loader is not None else {}
    if test_metrics:
        print(format_split_metrics("test", test_metrics))
        print(f"test_class_acc={test_metrics['class_accuracy']:.4f}")
        save_metrics_json(args.save_dir / "test.json", test_metrics)


def make_loader(dataset: Subset, args: argparse.Namespace, shuffle: bool) -> DataLoader | None:
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_panel_ordering_batch,
    )


def build_model(args: argparse.Namespace) -> MangaZeroJigsawSolver:
    return MangaZeroJigsawSolver(
        panel_count=args.panel_count,
        max_permutations=args.max_permutations,
        permutation_seed=args.permutation_seed,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        use_layout=args.use_layout,
        dropout=args.dropout,
    )


def train_one_epoch(
    model: MangaZeroJigsawSolver,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    if loader is None:
        return {"loss": 0.0, "class_accuracy": 0.0}
    model.train()
    losses = []
    class_accuracy = []
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True) if tqdm is not None else loader
    for batch in progress:
        prepared = prepare_self_supervised_batch(batch, model)
        prepared = move_batch(prepared, device, args.use_layout)
        logits = model(
            prepared["panel_images"],
            layout_features=prepared.get("layout_features"),
        )
        loss = jigsaw_cross_entropy(logits, prepared["target_order"], model)
        class_targets = model.target_to_class_indices(prepared["target_order"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
        class_accuracy.append(float(logits.argmax(dim=-1).eq(class_targets).float().mean().item()))
        if tqdm is not None:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    return {
        "loss": float(mean(losses)) if losses else 0.0,
        "class_accuracy": float(mean(class_accuracy)) if class_accuracy else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: MangaZeroJigsawSolver,
    loader: DataLoader | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    losses = []
    class_accuracy = []
    exact = []
    position = []
    pairwise = []
    covered_samples = 0
    total_samples = 0
    for batch in loader:
        batch = move_batch(batch, device, args.use_layout)
        total_samples += batch["target_order"].size(0)
        try:
            class_targets = model.target_to_class_indices(batch["target_order"])
        except ValueError:
            continue
        covered_samples += batch["target_order"].size(0)
        logits = model(
            batch["panel_images"],
            layout_features=batch.get("layout_features"),
        )
        losses.append(float(torch.nn.functional.cross_entropy(logits, class_targets).item()))
        class_accuracy.append(float(logits.argmax(dim=-1).eq(class_targets).float().mean().item()))
        pred = model.predict_order(
            batch["panel_images"],
            layout_features=batch.get("layout_features"),
        )
        metrics = ordering_metrics(pred, batch["target_order"])
        exact.append(metrics["exact_match"])
        position.append(metrics["position_accuracy"])
        pairwise.append(metrics["pairwise_accuracy"])
    return {
        "loss": float(mean(losses)) if losses else 0.0,
        "class_accuracy": float(mean(class_accuracy)) if class_accuracy else 0.0,
        "exact_match": float(mean(exact)) if exact else 0.0,
        "position_accuracy": float(mean(position)) if position else 0.0,
        "pairwise_accuracy": float(mean(pairwise)) if pairwise else 0.0,
        "candidate_coverage": covered_samples / total_samples if total_samples else 0.0,
    }


def prepare_self_supervised_batch(
    batch: dict[str, Any],
    model: MangaZeroJigsawSolver,
) -> dict[str, Any]:
    panel_count = model.panel_count
    target_order = batch["target_order"].long()
    if target_order.ndim != 2 or target_order.size(1) != panel_count:
        raise ValueError(
            f"expected target_order shape [B,{panel_count}], got {tuple(target_order.shape)}"
        )
    canonical_images, canonical_layout, canonical_panels = recover_canonical_batch(
        batch["panel_images"],
        batch["panels"],
        batch.get("layout_features"),
    )
    batch_size = target_order.size(0)
    class_ids = torch.randint(
        low=0,
        high=model.num_classes,
        size=(batch_size,),
    )
    permutation = model.permutation_bank[class_ids].to(device=canonical_images.device)
    prepared = dict(batch)
    prepared["panel_images"] = gather_by_order(canonical_images, permutation)
    # ``permutation`` is input_position -> canonical_panel. The current
    # MangaZero API expects its inverse: canonical_panel -> input_position.
    prepared["target_order"] = inverse_permutation(permutation)
    if canonical_layout is not None:
        prepared["layout_features"] = gather_by_order(canonical_layout, permutation)
    if canonical_panels is not None:
        prepared["panels"] = [
            [sample_items[index] for index in sample_order]
            for sample_items, sample_order in zip(canonical_panels, permutation.tolist())
        ]
    return prepared


def move_batch(batch: dict[str, Any], device: torch.device, use_layout: bool) -> dict[str, Any]:
    moved = dict(batch)
    moved["panel_images"] = batch["panel_images"].to(device)
    moved["target_order"] = batch["target_order"].to(device)
    if use_layout and "layout_features" in batch:
        moved["layout_features"] = batch["layout_features"].to(device)
    return moved


def save_checkpoint(
    path: Path,
    model: MangaZeroJigsawSolver,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": vars(args),
            "solver_family": "mangazero_jigsaw",
            "num_classes": model.num_classes,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MangaZeroJigsawSolver,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def format_jigsaw_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> str:
    parts = [
        f"epoch={epoch}/{total_epochs}",
        f"train_loss={train_metrics['loss']:.6f}",
        f"train_class={train_metrics['class_accuracy']:.4f}",
    ]
    for prefix, metrics in (("val", val_metrics), ("test", test_metrics or {})):
        if metrics:
            parts.extend(
                [
                    f"{prefix}_loss={metrics['loss']:.6f}",
                    f"{prefix}_class={metrics['class_accuracy']:.4f}",
                    f"{prefix}_coverage={metrics.get('candidate_coverage', 0.0):.4f}",
                    f"{prefix}_exact={metrics['exact_match']:.4f}",
                    f"{prefix}_pos={metrics['position_accuracy']:.4f}",
                    f"{prefix}_pairwise={metrics['pairwise_accuracy']:.4f}",
                ]
            )
    return " ".join(parts)


if __name__ == "__main__":
    main()
