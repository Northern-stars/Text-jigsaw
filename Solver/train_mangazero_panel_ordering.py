"""Train a set-to-sequence solver for MangaZero panel ordering."""

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

from Solver.agent.mangazero_set2seq_solver import MangaZeroSetToSequenceSolver, pointer_cross_entropy
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MangaZero panel-ordering set-to-sequence solver.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Mangazero/ordering_dataset"))
    parser.add_argument("--solver-type", default="visual", choices=("visual", "text", "multimodal"))
    parser.add_argument("--epoch", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--epochs", type=int, default=None, help="Alias of --epoch.")
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--test-per-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-feature-dim", type=int, default=512)
    parser.add_argument("--vit-backbone", default="pretrained", choices=("pretrained", "lightweight"))
    parser.add_argument("--vit-pretrained", action="store_true", default=True)
    parser.add_argument("--no-vit-pretrained", action="store_false", dest="vit_pretrained")
    parser.add_argument("--vit-freeze", action="store_true")
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--vit-layers", type=int, default=4)
    parser.add_argument("--vit-num-heads", type=int, default=8)
    parser.add_argument("--text-feature-dim", type=int, default=256)
    parser.add_argument("--text-vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-layout", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-dir", type=Path, default=Path("Solver/checkpoints_mangazero"))
    parser.add_argument("--load", type=Path, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is not None:
        args.epoch = args.epochs
    split_ratio = parse_split_ratio(args.split_ratio)
    if args.test_per_epoch <= 0:
        raise ValueError("--test-per-epoch must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_layout = args.use_layout

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(args.image_width, args.image_height),
        use_layout=use_layout,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, split_ratio, args.seed)
    print(
        "dataset_split "
        f"train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"test={len(test_dataset)}"
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

    model = MangaZeroSetToSequenceSolver(
        panel_count=args.panel_count,
        solver_type=args.solver_type,
        image_feature_dim=args.image_feature_dim,
        image_size=(args.image_height, args.image_width),
        vit_backbone=args.vit_backbone,
        vit_pretrained=args.vit_pretrained,
        vit_freeze=args.vit_freeze,
        vit_patch_size=args.vit_patch_size,
        vit_layers=args.vit_layers,
        vit_num_heads=args.vit_num_heads,
        text_feature_dim=args.text_feature_dim,
        text_vocab_size=args.text_vocab_size,
        d_model=args.d_model,
        use_layout=use_layout,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.load is not None:
        start_epoch = load_checkpoint(args.load, model, optimizer, device) + 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)

    best_val = -1.0
    for epoch in range(start_epoch, args.epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, use_layout, args.grad_clip_norm, epoch)
        val_metrics = evaluate(model, val_loader, device, use_layout) if val_loader is not None else {}
        test_metrics = None
        if test_loader is not None and epoch % args.test_per_epoch == 0:
            test_metrics = evaluate(model, test_loader, device, use_layout)
        print(format_metrics(epoch, args.epoch, train_metrics, val_metrics, test_metrics))

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args)
        if val_metrics and val_metrics["exact_match"] >= best_val:
            best_val = val_metrics["exact_match"]
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, args)
        if test_metrics is not None:
            save_metrics_json(args.save_dir / f"test_epoch_{epoch:04d}.json", test_metrics)

    save_checkpoint(args.save_dir / "last.pt", model, optimizer, args.epoch, args)

    test_metrics = evaluate(model, test_loader, device, use_layout) if test_loader is not None else {}
    if test_metrics:
        print(format_split_metrics("test", test_metrics))
        save_metrics_json(args.save_dir / "test.json", test_metrics)
    else:
        print("test skipped: no test split available")


def parse_split_ratio(raw_ratio: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in raw_ratio.split(","))
    if len(values) != 3:
        raise ValueError("--split-ratio must contain three comma-separated values")
    total = sum(values)
    if total <= 0:
        raise ValueError("--split-ratio sum must be positive")
    return tuple(value / total for value in values)


def split_dataset(
    dataset: MangaZeroPanelOrderingDataset,
    ratio: tuple[float, float, float],
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = round(len(indices) * ratio[0])
    val_end = train_end + round(len(indices) * ratio[1])
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def train_one_epoch(
    model: MangaZeroSetToSequenceSolver,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_layout: bool,
    grad_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    losses = []
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True) if tqdm is not None else loader
    for batch in progress:
        batch = randomize_training_batch(batch, use_layout)
        batch = move_batch(batch, device, use_layout)
        logits = model(
            batch["panel_images"],
            target_order=batch["target_order"],
            layout_features=batch.get("layout_features"),
            dialog_texts=batch.get("dialog_texts"),
        )
        loss = pointer_cross_entropy(logits, batch["target_order"])
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
    model: MangaZeroSetToSequenceSolver,
    loader: DataLoader | None,
    device: torch.device,
    use_layout: bool,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    losses = []
    exact = []
    position = []
    pairwise = []
    for batch in loader:
        batch = move_batch(batch, device, use_layout)
        logits = model(
            batch["panel_images"],
            target_order=batch["target_order"],
            layout_features=batch.get("layout_features"),
            dialog_texts=batch.get("dialog_texts"),
        )
        losses.append(float(pointer_cross_entropy(logits, batch["target_order"]).item()))
        pred = model.greedy_decode(
            batch["panel_images"],
            layout_features=batch.get("layout_features"),
            dialog_texts=batch.get("dialog_texts"),
        )
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


def move_batch(batch: dict[str, Any], device: torch.device, use_layout: bool) -> dict[str, Any]:
    moved = dict(batch)
    moved["panel_images"] = batch["panel_images"].to(device)
    moved["target_order"] = batch["target_order"].to(device)
    if use_layout and "layout_features" in batch:
        moved["layout_features"] = batch["layout_features"].to(device)
    return moved


def randomize_training_batch(batch: dict[str, Any], use_layout: bool) -> dict[str, Any]:
    """Restore canonical panel order from labels, then shuffle again online."""
    if "panel_images" not in batch or "target_order" not in batch:
        return batch

    moved = dict(batch)
    panel_images = batch["panel_images"]
    target_order = batch["target_order"].long()
    batch_size, panel_count = target_order.shape

    canonical_images = gather_batch_by_order(panel_images, target_order)
    permutation = torch.stack([torch.randperm(panel_count) for _ in range(batch_size)], dim=0)
    moved["panel_images"] = gather_batch_by_order(canonical_images, permutation)
    moved["target_order"] = torch.argsort(permutation, dim=1)

    if use_layout and "layout_features" in batch:
        moved["layout_features"] = gather_batch_by_order(batch["layout_features"], target_order)
        moved["layout_features"] = gather_batch_by_order(moved["layout_features"], permutation)

    if "dialog_texts" in batch:
        canonical_dialog_texts = reorder_nested_list(batch["dialog_texts"], target_order)
        moved["dialog_texts"] = reorder_nested_list(canonical_dialog_texts, permutation)

    if "panels" in batch:
        canonical_panels = reorder_nested_list(batch["panels"], target_order)
        moved["panels"] = reorder_nested_list(canonical_panels, permutation)

    return moved


def gather_batch_by_order(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected batched tensor with at least 2 dims, got shape {tuple(tensor.shape)}")
    index = order
    for _ in range(tensor.ndim - 2):
        index = index.unsqueeze(-1)
    index = index.expand(*order.shape, *tensor.shape[2:])
    return tensor.gather(1, index)


def reorder_nested_list(items: list[Any], order: torch.Tensor) -> list[list[Any]]:
    reordered: list[list[Any]] = []
    for sample_items, sample_order in zip(items, order.tolist()):
        reordered.append([sample_items[index] for index in sample_order])
    return reordered


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
    parts = [
        f"epoch={epoch}/{total_epochs}",
        f"train_loss={train_metrics['loss']:.6f}",
    ]
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


def save_checkpoint(
    path: Path,
    model: MangaZeroSetToSequenceSolver,
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
    model: MangaZeroSetToSequenceSolver,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


if __name__ == "__main__":
    main()
