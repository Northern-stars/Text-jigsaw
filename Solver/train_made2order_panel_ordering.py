"""Train the independent Made2Order panel-ordering solver.

The model comes from third_party/made2order (ECCV 2024), which is imported
at runtime through Solver/agent/made2order_set2order_solver.py. Training
reuses the same MangaZero panel-ordering datasets and metrics as the other
solver entry points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(".")

from Solver.agent.made2order_set2order_solver import Made2OrderPanelOrderingSolver
from Solver.data.panel_ordering_dataset import (
    MangaZeroPanelOrderingDataset,
    collate_panel_ordering_batch,
    load_panel_ordering_splits,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Made2Order panel-ordering solver.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Mangazero/ordering_dataset"))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the dataset slice for quick smoke runs.")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="Alias of --epoch.")
    parser.add_argument("--test-per-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16, help="Square patch size used by upstream Made2Order.")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=64)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-dir", type=Path, default=Path("Solver/checkpoints_made2order"))
    parser.add_argument("--load", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["panel_images"] = batch["panel_images"].to(device)
    moved["target_order"] = batch["target_order"].to(device)
    return moved


def randomize_training_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Shuffle panels again online; labels follow the dataset convention."""
    panel_images = batch["panel_images"]
    target_order = batch["target_order"].long()
    batch_size, panel_count = target_order.shape

    # target_order[i] maps input slot i to its canonical reading rank, so the
    # canonical (unshuffled) input is panel_images.gather(1, target_order).
    expanded_order = target_order.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(panel_images)
    canonical_images = panel_images.gather(1, expanded_order)
    permutation = torch.stack([torch.randperm(panel_count) for _ in range(batch_size)], dim=0)
    expanded_perm = permutation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(panel_images)
    shuffled = canonical_images.gather(1, expanded_perm)

    moved = dict(batch)
    moved["panel_images"] = shuffled
    moved["target_order"] = torch.argsort(permutation, dim=1)
    return moved


def ordering_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    exact = pred.eq(target).all(dim=1).float().mean().item()
    position = pred.eq(target).float().mean().item()
    pairwise_scores = []
    for pred_row, target_row in zip(pred.tolist(), target.tolist()):
        pairwise_scores.append(pairwise_accuracy(pred_row, target_row))
    return {
        "exact_match": float(exact),
        "position_accuracy": float(position),
        "pairwise_accuracy": float(mean(pairwise_scores)) if pairwise_scores else 0.0,
    }


def pairwise_accuracy(pred_order: list[int], target_order: list[int]) -> float:
    pred_rank = {panel: rank for rank, panel in enumerate(pred_order)}
    target_rank = {panel: rank for rank, panel in enumerate(target_order)}
    correct = 0
    total = 0
    for i, panel_i in enumerate(target_order):
        for panel_j in target_order[i + 1:]:
            total += 1
            if pred_rank[panel_i] < pred_rank[panel_j]:
                correct += 1
    return correct / total if total else 0.0


def format_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> str:
    parts = [f"epoch={epoch}/{total_epochs}", f"train_loss={train_metrics['loss']:.6f}"]
    if val_metrics:
        parts.extend(
            [
                f"val_loss={val_metrics['loss']:.6f}",
                f"val_exact={val_metrics['exact_match']:.4f}",
                f"val_pos={val_metrics['position_accuracy']:.4f}",
                f"val_pairwise={val_metrics['pairwise_accuracy']:.4f}",
            ]
        )
    if test_metrics is not None:
        parts.extend(
            [
                f"test_loss={test_metrics['loss']:.6f}",
                f"test_exact={test_metrics['exact_match']:.4f}",
                f"test_pos={test_metrics['position_accuracy']:.4f}",
                f"test_pairwise={test_metrics['pairwise_accuracy']:.4f}",
            ]
        )
    return " ".join(parts)


def format_split_metrics(split_name: str, metrics: dict[str, float]) -> str:
    return (
        f"{split_name}_loss={metrics['loss']:.6f} "
        f"{split_name}_exact={metrics['exact_match']:.4f} "
        f"{split_name}_pos={metrics['position_accuracy']:.4f} "
        f"{split_name}_pairwise={metrics['pairwise_accuracy']:.4f}"
    )


def train_one_epoch(
    model: Made2OrderPanelOrderingSolver,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    losses = []
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True) if tqdm is not None else loader
    for batch in progress:
        batch = randomize_training_batch(batch)
        batch = move_batch(batch, device)
        scores, _ = model(batch["panel_images"])
        loss = model.ranking_loss(scores, batch["target_order"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
        if tqdm is not None:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    return {"loss": float(mean(losses)) if losses else 0.0}


@torch.no_grad()
def evaluate(
    model: Made2OrderPanelOrderingSolver,
    loader: DataLoader | None,
    device: torch.device,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    losses = []
    exact = []
    position = []
    pairwise = []
    for batch in loader:
        batch = move_batch(batch, device)
        scores, _ = model(batch["panel_images"])
        losses.append(float(model.ranking_loss(scores, batch["target_order"]).item()))
        pred = model.predict_order(batch["panel_images"])
        metrics = ordering_metrics(pred, batch["target_order"])
        exact.append(metrics["exact_match"])
        position.append(metrics["position_accuracy"])
        pairwise.append(metrics["pairwise_accuracy"])
    return {
        "loss": float(mean(losses)) if losses else 0.0,
        "exact_match": float(mean(exact)) if exact else 0.0,
        "position_accuracy": float(mean(position)) if position else 0.0,
        "pairwise_accuracy": float(mean(pairwise)) if pairwise else 0.0,
    }


def save_checkpoint(
    path: Path,
    model: Made2OrderPanelOrderingSolver,
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
        },
        path,
    )


def save_metrics_json(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def load_checkpoint(
    path: Path,
    model: Made2OrderPanelOrderingSolver,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    args = parse_args()
    if args.epochs is not None:
        args.epoch = args.epochs
    if args.test_per_epoch <= 0:
        raise ValueError("--test-per-epoch must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_dataset, val_dataset, test_dataset = load_panel_ordering_splits(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_width, args.image_height),
        use_layout=False,
    )
    if args.max_samples is not None and args.max_samples > 0:
        train_dataset = Subset(
            train_dataset,
            list(range(min(args.max_samples, len(train_dataset)))),
        )
    print(
        "dataset_split "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_panel_ordering_batch,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_panel_ordering_batch,
        )
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_panel_ordering_batch,
        )

    model = Made2OrderPanelOrderingSolver(
        panel_count=args.panel_count,
        image_size=(args.image_height, args.image_width),
        patch_size=args.patch_size,
        dim=args.dim,
        heads=args.heads,
        decoder_dim=args.decoder_dim,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.load is not None:
        start_epoch = load_checkpoint(args.load, model, optimizer, device) + 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)

    best_val = -1.0
    for epoch in range(start_epoch, args.epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip_norm, epoch)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        test_metrics = None
        if test_loader is not None and epoch % args.test_per_epoch == 0:
            test_metrics = evaluate(model, test_loader, device)
        print(format_metrics(epoch, args.epoch, train_metrics, val_metrics, test_metrics))

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args)
        if val_metrics and val_metrics["exact_match"] >= best_val:
            best_val = val_metrics["exact_match"]
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, args)
        if test_metrics is not None:
            save_metrics_json(args.save_dir / f"test_epoch_{epoch:04d}.json", test_metrics)

    save_checkpoint(args.save_dir / "last.pt", model, optimizer, args.epoch, args)

    test_metrics = evaluate(model, test_loader, device) if test_loader is not None else {}
    if test_metrics:
        print(format_split_metrics("test", test_metrics))
        save_metrics_json(args.save_dir / "test.json", test_metrics)
    else:
        print("test skipped: no test split available")


if __name__ == "__main__":
    main()
