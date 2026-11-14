"""Train a Qwen-VL hidden-state classifier for the first panel of a puzzle.

The target is the first input-slot index in ``target_order``. This is a
small feasibility task: the VLM provides one pooled puzzle representation and
the classifier head predicts one of the input panel slots.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cli import parse_args_with_config, save_startup_config
from experiment.run_logger import ExperimentLogger
from Solver.data.panel_ordering_dataset import (
    MangaZeroPanelOrderingDataset,
    collate_panel_ordering_batch,
)
from experiment.first_panel_classifier.model import QwenVLFirstPanelClassifier
from experiment.first_panel_classifier.targets import first_panel_targets
from experiment.first_panel_classifier.targets import LABEL_DEFINITION, validate_checkpoint_label_definition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--train-backbone", action="store_true", help="Add LoRA adapters to the Qwen backbone.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/first_panel_classifier"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parse_args_with_config(parser, argv, script_file=__file__)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    train_set = MangaZeroPanelOrderingDataset(
        args.dataset_dir, split="train", image_size=(args.image_width, args.image_height), use_layout=False
    )
    val_set = MangaZeroPanelOrderingDataset(
        args.dataset_dir, split="val", image_size=(args.image_width, args.image_height), use_layout=False, allow_empty=True
    )
    test_set = MangaZeroPanelOrderingDataset(
        args.dataset_dir, split="test", image_size=(args.image_width, args.image_height), use_layout=False, allow_empty=True
    )
    if not len(train_set):
        raise ValueError("training split is empty")

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "resume"}
    config["experiment"] = "first_panel_classifier"
    config["label_definition"] = LABEL_DEFINITION
    run_id = args.run_id or f"{args.model_name.rsplit('/', 1)[-1]}_seed{args.seed}"
    run_dir = args.run_root / run_id
    logger = ExperimentLogger(run_dir, config, resume=args.resume is not None)
    save_startup_config(args, run_dir / "checkpoints")

    device = torch.device(args.device)
    model = QwenVLFirstPanelClassifier(
        panel_count=args.panel_count,
        model_name=args.model_name,
        revision=args.model_revision,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation or None,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        train_backbone=args.train_backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(item.strip() for item in args.lora_target_modules.split(",") if item.strip()),
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        classifier_dropout=args.classifier_dropout,
    )
    if args.device_map is None:
        model.to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch) if len(val_set) else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch) if len(test_set) else None
    optimizer = build_optimizer(model, args)
    start_epoch = 1
    global_step = 0
    best_val = -1.0
    if args.resume is not None:
        state = ExperimentLogger.load_checkpoint(args.resume, optimizer=optimizer, map_location=device)
        validate_checkpoint_label_definition(state.get("extra", {}))
        model.load_trainable_state_dict(state["model"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("extra", {}).get("best_val", -1.0))

    logger.log(f"start train={len(train_set)} val={len(val_set)} test={len(test_set)} train_backbone={args.train_backbone}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, steps = train_one_epoch(model, train_loader, optimizer, device, args)
        global_step += steps
        val_metrics = evaluate(model, val_loader, device, args) if val_loader is not None else {}
        record = {"type": "epoch", "epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics, "optimizer_steps": steps}
        logger.log_metrics(record)
        logger.log(f"epoch={epoch}/{args.epochs} train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics.get('accuracy', 0.0):.4f}")
        extra = {
            "best_val": best_val,
            "panel_count": args.panel_count,
            "label_definition": LABEL_DEFINITION,
        }
        if args.save_every > 0 and epoch % args.save_every == 0:
            logger.save_checkpoint(f"epoch_{epoch:04d}.pt", model_state=model.trainable_state_dict(), optimizer=optimizer, epoch=epoch, global_step=global_step, extra=extra)
        if val_metrics and val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            logger.save_checkpoint(
                "best.pt", model_state=model.trainable_state_dict(), optimizer=optimizer,
                epoch=epoch, global_step=global_step,
                extra={"best_val": best_val, "label_definition": LABEL_DEFINITION},
            )

    logger.save_checkpoint(
        "last.pt", model_state=model.trainable_state_dict(), optimizer=optimizer,
        epoch=args.epochs, global_step=global_step,
        extra={"best_val": best_val, "label_definition": LABEL_DEFINITION},
    )
    test_metrics = evaluate(model, test_loader, device, args) if test_loader is not None else {}
    logger.log_metrics({"type": "final_test", "test": test_metrics})
    logger.log(f"final test_acc={test_metrics.get('accuracy', 0.0):.4f}")


def build_optimizer(model: QwenVLFirstPanelClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    head = list(model.classifier.parameters())
    backbone = [parameter for name, parameter in model.backbone.named_parameters() if parameter.requires_grad]
    groups = [{"params": head, "lr": args.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_one_epoch(model: QwenVLFirstPanelClassifier, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    correct = total = steps = 0
    for batch_index, batch in enumerate(tqdm(loader), start=1):
        images = batch["panel_images"].to(device)
        targets = first_panel_targets(batch["target_order"].to(device))
        logits = model(images, dialog_texts=batch.get("dialog_texts"))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        (loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        losses.append(float(loss.item()))
        correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
        total += int(targets.numel())
    return {"loss": float(np.mean(losses)) if losses else 0.0, "accuracy": correct / total if total else 0.0}, steps


@torch.no_grad()
def evaluate(model: QwenVLFirstPanelClassifier, loader: DataLoader | None, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    losses: list[float] = []
    correct = total = top_k_correct = 0
    for batch in loader:
        images = batch["panel_images"].to(device)
        targets = first_panel_targets(batch["target_order"].to(device))
        logits = model(images, dialog_texts=batch.get("dialog_texts"))
        losses.append(float(torch.nn.functional.cross_entropy(logits, targets).item()))
        correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
        top_k = min(2, logits.shape[-1])
        top_k_correct += int(logits.topk(top_k, dim=-1).indices.eq(targets.unsqueeze(-1)).any(dim=-1).sum().item())
        total += int(targets.numel())
    return {"loss": float(np.mean(losses)) if losses else 0.0, "accuracy": correct / total if total else 0.0, "top2_accuracy": top_k_correct / total if total else 0.0}


def validate_args(args: argparse.Namespace) -> None:
    if args.panel_count <= 1 or args.batch_size <= 0 or args.epochs <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("panel-count must be >1; epochs, batch-size, and grad-accum-steps must be positive")
    if args.train_backbone and args.lora_r <= 0:
        raise ValueError("--lora-r must be positive")
    if args.load_in_4bit and not args.train_backbone:
        raise ValueError("--load-in-4bit requires --train-backbone")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
