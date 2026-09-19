"""Train Qwen2.5-VL-3B + LoRA for MangaZero panel ordering classification."""

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

from Solver.agent.mangazero_qwen_vl_lora_classifier import (
    MangaZeroQwenVLLoRAClassifier,
    qwen_vl_cross_entropy,
)
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import (
    format_metrics,
    format_split_metrics,
    ordering_metrics,
    parse_split_ratio,
    randomize_training_batch,
    save_metrics_json,
    split_dataset,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL + LoRA classifier for MangaZero ordering.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--test-per-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--max-permutation-classes", type=int, default=50000)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto for 4-bit loading.")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--classifier-hidden-dim", type=int, default=0)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-dir", type=Path, default=Path("Solver/checkpoints_mangazero_qwen_vl_lora"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-full-state", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is not None:
        args.epoch = args.epochs
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split_ratio = parse_split_ratio(args.split_ratio)

    full_dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(args.image_width, args.image_height),
        use_layout=False,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, split_ratio, args.seed)
    print(f"dataset_split train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}", flush=True)

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    model = build_model(args)
    if not args.load_in_4bit:
        model = model.to(device)
    else:
        model.classifier.to(device)
    optimizer = build_optimizer(model, args)

    start_epoch = 1
    if args.load is not None:
        start_epoch = load_checkpoint(args.load, model, optimizer, device) + 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    print_trainable_parameters(model)

    best_val = -1.0
    for epoch in range(start_epoch, args.epoch + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        test_metrics = None
        if test_loader is not None and epoch % args.test_per_epoch == 0:
            test_metrics = evaluate(model, test_loader, device)
        print(format_metrics(epoch, args.epoch, train_metrics, val_metrics, test_metrics), flush=True)
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
        print(format_split_metrics("test", test_metrics), flush=True)
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


def build_model(args: argparse.Namespace) -> MangaZeroQwenVLLoRAClassifier:
    target_modules = tuple(
        item.strip()
        for item in args.lora_target_modules.split(",")
        if item.strip()
    )
    return MangaZeroQwenVLLoRAClassifier(
        panel_count=args.panel_count,
        model_name=args.model_name,
        max_permutation_classes=args.max_permutation_classes,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation or None,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        classifier_hidden_dim=args.classifier_hidden_dim,
        classifier_dropout=args.classifier_dropout,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )


def build_optimizer(model: MangaZeroQwenVLLoRAClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_one_epoch(
    model: MangaZeroQwenVLLoRAClassifier,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    if loader is None:
        return {"loss": 0.0}
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True) if tqdm is not None else loader
    for step, batch in enumerate(progress, start=1):
        batch = randomize_training_batch(batch, use_layout=False)
        batch = move_batch(batch, device)
        logits = model(batch["panel_images"], dialog_texts=batch.get("dialog_texts"))
        loss = qwen_vl_cross_entropy(logits, batch["target_order"], model) / args.grad_accum_steps
        loss.backward()
        if step % args.grad_accum_steps == 0:
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item() * args.grad_accum_steps))
        if tqdm is not None:
            progress.set_postfix({"loss": f"{losses[-1]:.4f}"})
    if len(losses) % args.grad_accum_steps != 0:
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": float(mean(losses)) if losses else 0.0}


@torch.no_grad()
def evaluate(
    model: MangaZeroQwenVLLoRAClassifier,
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
    class_accuracy = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["panel_images"], dialog_texts=batch.get("dialog_texts"))
        class_targets = model.target_to_class_indices(batch["target_order"])
        losses.append(float(torch.nn.functional.cross_entropy(logits, class_targets).item()))
        class_accuracy.append(float(logits.argmax(dim=-1).eq(class_targets).float().mean().item()))
        pred = model.predict_order(batch["panel_images"], dialog_texts=batch.get("dialog_texts"))
        metrics = ordering_metrics(pred, batch["target_order"])
        exact.append(metrics["exact_match"])
        position.append(metrics["position_accuracy"])
        pairwise.append(metrics["pairwise_accuracy"])
    return {
        "loss": float(mean(losses)) if losses else 0.0,
        "exact_match": float(mean(exact)) if exact else 0.0,
        "position_accuracy": float(mean(position)) if position else 0.0,
        "pairwise_accuracy": float(mean(pairwise)) if pairwise else 0.0,
        "class_accuracy": float(mean(class_accuracy)) if class_accuracy else 0.0,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["panel_images"] = batch["panel_images"].to(device)
    moved["target_order"] = batch["target_order"].to(device)
    return moved


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters trainable={trainable} total={total} ratio={trainable / max(1, total):.6f}", flush=True)


def save_checkpoint(
    path: Path,
    model: MangaZeroQwenVLLoRAClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    model_state = model.state_dict() if args.save_full_state else model.trainable_state_dict()
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": vars(args),
            "solver_family": "qwen_vl_lora_classifier",
            "full_state": args.save_full_state,
            "num_classes": model.num_classes,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MangaZeroQwenVLLoRAClassifier,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_trainable_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


if __name__ == "__main__":
    main()
