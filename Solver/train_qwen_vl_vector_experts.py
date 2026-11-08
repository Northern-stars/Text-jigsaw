"""Train vector-conditioned LoRA experts on a shared Qwen-VL pointer solver."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(".")

from Solver.agent.qwen_vl_vector_expert import (
    QwenVLVectorExpertSolver,
    expert_loss,
    iter_expert_losses,
)
from Solver.data.panel_ordering_dataset import (
    collate_panel_ordering_batch,
    load_panel_ordering_splits,
)
from Solver.train_mangazero_panel_ordering import (
    ordering_metrics,
    randomize_training_batch,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train vector-conditioned LoRA experts for Qwen-VL panel ordering."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("Solver/checkpoints_qwen_vl_vector_experts"))
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--generated-lora-rank", type=int, default=4)
    parser.add_argument("--generated-lora-last-n-layers", type=int, default=2)
    parser.add_argument("--generated-lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--feature-mode", choices=("independent", "generated"), default="independent")
    parser.add_argument("--feature-hidden-dim", type=int, default=128)
    parser.add_argument("--feature-init-checkpoint", type=Path, default=None,
                        help="Initialize generated features from a stage-one independent-feature checkpoint.")
    parser.add_argument("--feature-fit-steps", type=int, default=500)
    parser.add_argument("--feature-fit-lr", type=float, default=1e-3)
    parser.add_argument("--mapping-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr-expert", type=float, default=1e-2)
    parser.add_argument("--lr-feature-mapper", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load", type=Path, default=None, help="Resume a vector-expert checkpoint.")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_dataset, val_dataset, test_dataset = load_panel_ordering_splits(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_width, args.image_height),
        use_layout=False,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    model = QwenVLVectorExpertSolver(
        base_checkpoint=args.base_checkpoint,
        panel_count=args.panel_count,
        latent_dim=args.latent_dim,
        num_experts=args.num_experts,
        lora_rank=args.generated_lora_rank,
        lora_last_n_layers=args.generated_lora_last_n_layers,
        lora_target_modules=tuple(
            item.strip() for item in args.generated_lora_target_modules.split(",") if item.strip()
        ),
        feature_mode=args.feature_mode,
        feature_hidden_dim=args.feature_hidden_dim,
        mapping_seed=args.mapping_seed,
    )
    if not bool(getattr(model.backbone, "is_loaded_in_4bit", False)):
        model.to(device)
    else:
        model.panel_projection.to(device)
        model.panel_norm.to(device)
        model.pointer_decoder.to(device)
        model.output_norm.to(device)
        model.query_projection.to(device)
        model.key_projection.to(device)
        model.type_embedding.data = model.type_embedding.data.to(device)
        model.bos_token.data = model.bos_token.data.to(device)
        model.expert_vectors.data = model.expert_vectors.data.to(device)
        if model.expert_features is not None:
            model.expert_features.data = model.expert_features.data.to(device)
        if model.feature_hypernetwork is not None:
            model.feature_hypernetwork.to(device)
        for module in model.generated_modules:
            module.a0 = module.a0.to(device)
            module.map_a = module.map_a.to(device)
            module.map_b = module.map_b.to(device)

    optimizer = build_optimizer(model, args)
    start_epoch = 1
    if args.feature_init_checkpoint is not None:
        if args.feature_mode != "generated":
            raise ValueError("--feature-init-checkpoint requires --feature-mode generated")
        if args.load is not None:
            raise ValueError("use either --load or --feature-init-checkpoint, not both")
        init_checkpoint = torch_load(args.feature_init_checkpoint, "cpu")
        if Path(init_checkpoint.get("base_checkpoint", "")).resolve() != Path(model.base_checkpoint).resolve():
            raise ValueError("stage-one and stage-two checkpoints use different shared base checkpoints")
        model.initialize_generated_features(
            init_checkpoint["expert_state"],
            fit_steps=args.feature_fit_steps,
            fit_lr=args.feature_fit_lr,
        )
    if args.load is not None:
        checkpoint = torch_load(args.load, device)
        if Path(checkpoint.get("base_checkpoint", "")).resolve() != Path(model.base_checkpoint).resolve():
            raise ValueError("resume checkpoint was trained from a different shared base checkpoint")
        model.load_expert_checkpoint_state(checkpoint["expert_state"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    print(
        f"experts={args.num_experts} targets={len(model.generated_modules)} "
        f"trainable={sum(p.numel() for p in model.expert_trainable_parameters())}",
        flush=True,
    )

    best_val = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        print(format_metrics(epoch, train_metrics, val_metrics, {}), flush=True)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args)
        if val_metrics and val_metrics["exact_match"] >= best_val:
            best_val = val_metrics["exact_match"]
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, args)
    save_checkpoint(args.save_dir / "last.pt", model, optimizer, args.epochs, args)
    if test_loader is not None:
        best_path = args.save_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch_load(best_path, device)
            model.load_expert_checkpoint_state(best_checkpoint["expert_state"])
        test_metrics = evaluate(model, test_loader, device)
        with (args.save_dir / "test.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, ensure_ascii=False, indent=2)


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader | None:
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_panel_ordering_batch,
    )


def build_optimizer(
    model: QwenVLVectorExpertSolver,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    groups = [{"params": [model.expert_vectors], "lr": args.lr_expert}]
    if model.expert_features is not None:
        groups.append({"params": [model.expert_features], "lr": args.lr_expert})
    if model.feature_hypernetwork is not None:
        groups.append({"params": model.feature_hypernetwork.parameters(), "lr": args.lr_feature_mapper})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_one_epoch(
    model: QwenVLVectorExpertSolver,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    if loader is None:
        return {"loss": 0.0}
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True) if tqdm is not None else loader
    for step, batch in enumerate(progress, start=1):
        batch = randomize_training_batch(batch, use_layout=False)
        batch = move_batch(batch, device)
        expert_batch_losses = []
        for loss in iter_expert_losses(model, batch):
            expert_batch_losses.append(float(loss.detach().item()))
            (loss / model.num_experts / args.grad_accum_steps).backward()
        if step % args.grad_accum_steps == 0:
            clip_and_step(model, optimizer, args)
        losses.append(mean(expert_batch_losses) if expert_batch_losses else 0.0)
        if tqdm is not None:
            progress.set_postfix({"loss": f"{losses[-1]:.4f}"})
    if len(losses) % args.grad_accum_steps:
        clip_and_step(model, optimizer, args)
    return {"loss": float(mean(losses)) if losses else 0.0}


def clip_and_step(
    model: QwenVLVectorExpertSolver,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    parameters = model.expert_trainable_parameters()
    if args.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(
    model: QwenVLVectorExpertSolver,
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
        losses.append(float(expert_loss(model, batch).item()))
        prediction = model.ensemble_predict(
            batch["panel_images"],
            dialog_texts=batch.get("dialog_texts"),
        )
        metrics = ordering_metrics(prediction, batch["target_order"])
        exact.append(metrics["exact_match"])
        position.append(metrics["position_accuracy"])
        pairwise.append(metrics["pairwise_accuracy"])
    return {
        "loss": float(mean(losses)) if losses else 0.0,
        "exact_match": float(mean(exact)) if exact else 0.0,
        "position_accuracy": float(mean(position)) if position else 0.0,
        "pairwise_accuracy": float(mean(pairwise)) if pairwise else 0.0,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["panel_images"] = batch["panel_images"].to(device)
    moved["target_order"] = batch["target_order"].to(device)
    return moved


def save_checkpoint(
    path: Path,
    model: QwenVLVectorExpertSolver,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "solver_family": "qwen_vl_vector_experts",
            "base_checkpoint": model.base_checkpoint,
            "expert_state": model.expert_checkpoint_state(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": vars(args),
        },
        path,
    )


def format_metrics(
    epoch: int,
    train: dict[str, float],
    val: dict[str, float],
    test: dict[str, float],
) -> str:
    parts = [f"epoch={epoch}", f"train_loss={train['loss']:.6f}"]
    for split, metrics in (("val", val), ("test", test)):
        if metrics:
            parts.extend(
                f"{split}_{key}={metrics[key]:.6f}"
                for key in ("loss", "exact_match", "position_accuracy", "pairwise_accuracy")
            )
    return " ".join(parts)


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    main()
