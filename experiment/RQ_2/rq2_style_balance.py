"""Run the budget-matched RQ2 style-balance LoRA experiment."""

from __future__ import annotations
# Resolve the project config package when this file is run directly.
import sys as _cli_sys
from pathlib import Path as _CliPath
_cli_root = next((p for p in _CliPath(__file__).resolve().parents if (p / "config" / "cli.py").is_file()), None)
if _cli_root is not None and str(_cli_root) not in _cli_sys.path:
    _cli_sys.path.insert(0, str(_cli_root))
from config.cli import parse_args_with_config, save_startup_config

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Solver.agent.qwen_vl_ordering_solver import QwenVLOrderingSolver
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import ordering_metrics, randomize_training_batch
from experiment.run_logger import ExperimentLogger


CONDITIONS = ("manga_only", "comics_only", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/rq2"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-revision", default=None, help="Pin a model/processor commit hash for reproducibility.")
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--train-puzzles", type=int, default=None, help="Common budget per condition; defaults to the smallest style pool.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parse_args_with_config(parser, script_file=__file__)


def _validate_work_splits(rows: list[dict[str, Any]]) -> None:
    split_by_work: dict[str, str] = {}
    for row in rows:
        style = row.get("style")
        work = row.get("canonical_work_id") or row.get("work_id")
        if style not in {"manga", "comics"} or not work:
            raise ValueError("Every experiment manifest row must include style and work_id/canonical_work_id")
        previous = split_by_work.setdefault(str(work), str(row.get("split", "")))
        if previous != row.get("split"):
            raise ValueError(f"Work {work!r} appears in more than one split")


def _dataset(manifest: Path, split: str, args: argparse.Namespace) -> MangaZeroPanelOrderingDataset:
    return MangaZeroPanelOrderingDataset(
        dataset_dir=manifest.parent,
        manifest_path=manifest,
        split=split,
        image_size=(args.image_width, args.image_height),
        use_layout=False,
        given_first=True,
        allow_empty=True,
    )


def select_balanced_indices(dataset: MangaZeroPanelOrderingDataset, condition: str, budget: int, seed: int) -> list[int]:
    pools: dict[str, dict[str, list[int]]] = {"manga": defaultdict(list), "comics": defaultdict(list)}
    for index, sample in enumerate(dataset.samples):
        style = sample.style
        work = sample.work_id or sample.manga_id
        if style not in pools or not work:
            raise ValueError(f"Sample {sample.sequence_id} is missing valid RQ2 style/work metadata")
        pools[style][work].append(index)
    chosen_styles = {
        "manga_only": ("manga",),
        "comics_only": ("comics",),
        "mixed": ("manga", "comics"),
    }[condition]
    if any(not pools[style] for style in chosen_styles):
        raise ValueError(f"Condition {condition} has no samples for required styles {chosen_styles}")
    rng = random.Random(seed)
    shuffled: dict[str, dict[str, list[int]]] = {}
    work_order: dict[str, list[str]] = {}
    cursors: dict[str, dict[str, int]] = {}
    for style in chosen_styles:
        shuffled[style] = {}
        work_order[style] = sorted(pools[style])
        rng.shuffle(work_order[style])
        cursors[style] = {}
        for work, indices in pools[style].items():
            values = list(indices)
            rng.shuffle(values)
            shuffled[style][work] = values
            cursors[style][work] = 0
    style_slots = list(chosen_styles)
    if condition == "mixed":
        style_slots = ["manga", "comics"]
    selected: list[int] = []
    work_cursor = {style: 0 for style in chosen_styles}
    for slot in range(budget):
        style = style_slots[slot % len(style_slots)]
        works = work_order[style]
        found = None
        for offset in range(len(works)):
            work_index = (work_cursor[style] + offset) % len(works)
            work = works[work_index]
            cursor = cursors[style][work]
            candidates = shuffled[style][work]
            if cursor < len(candidates):
                found = candidates[cursor]
                cursors[style][work] += 1
                work_cursor[style] = (work_index + 1) % len(works)
                break
        if found is None:
            raise ValueError(f"Insufficient unique {style} puzzles for budget {budget}; do not oversample")
        selected.append(found)
    return selected


def _build_solver(args: argparse.Namespace) -> QwenVLOrderingSolver:
    return QwenVLOrderingSolver(
        model_name=args.model_name,
        revision=args.model_revision,
        panel_count=args.panel_count,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation or None,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(x.strip() for x in args.lora_target_modules.split(",") if x.strip()),
        max_new_tokens=args.max_new_tokens,
    )


@torch.no_grad()
def evaluate(solver: QwenVLOrderingSolver, loader: DataLoader, device: torch.device, split: str) -> dict[str, Any]:
    solver.backbone.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    by_style: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        images = batch["panel_images"].to(device)
        for index, image_set in enumerate(images):
            panels = [image_set[j] for j in range(image_set.shape[0])]
            texts = batch["dialog_texts"][index]
            style = batch["style"][index]
            try:
                result = solver.solve(panels, texts)
                prediction = torch.tensor(result.order, dtype=torch.long).unsqueeze(0)
                target = batch["target_order"][index:index + 1].cpu()
                metrics = ordering_metrics(prediction, target)
                valid = True
                raw_output = result.raw_output
            except Exception as exc:
                metrics = {"exact_match": 0.0, "position_accuracy": 0.0, "pairwise_accuracy": 0.0}
                valid = False
                raw_output = f"{type(exc).__name__}: {exc}"
            for key, value in metrics.items():
                totals[key].append(float(value))
                by_style[style][key].append(float(value))
            totals["valid_output"].append(float(valid))
            by_style[style]["valid_output"].append(float(valid))
            if hasattr(solver, "experiment_logger"):
                solver.experiment_logger.log_metrics({
                    "type": "prediction", "split": split, "sequence_id": batch["sequence_id"][index],
                    "style": style, "work_id": batch["work_id"][index], "target": batch["target_order"][index].tolist(),
                    "prediction": result.order if valid else None, "raw_output": raw_output,
                    "metrics": metrics, "valid_output": valid,
                })
    result = {key: float(np.mean(values)) for key, values in totals.items()}
    result["by_style"] = {style: {key: float(np.mean(values)) for key, values in metrics.items()} for style, metrics in by_style.items()}
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("epochs, batch-size, and grad-accum-steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    all_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    _validate_work_splits(all_rows)
    train_base = _dataset(args.manifest, "train", args)
    style_counts = {style: sum(s.style == style for s in train_base.samples) for style in ("manga", "comics")}
    if not all(style_counts.values()):
        raise ValueError(f"RQ2 needs train samples from both styles, got {style_counts}")
    budget = args.train_puzzles or min(style_counts.values())
    if budget > (style_counts["manga"] if args.condition == "manga_only" else style_counts["comics"] if args.condition == "comics_only" else 2 * min(style_counts.values())):
        raise ValueError(f"Budget {budget} exceeds available unique samples for {args.condition}: {style_counts}")
    selected = select_balanced_indices(train_base, args.condition, budget, args.seed)
    train_dataset = Subset(train_base, selected)
    val_dataset = _dataset(args.manifest, "val", args)
    test_dataset = _dataset(args.manifest, "test", args)
    if not len(val_dataset) or not len(test_dataset):
        raise ValueError("RQ2 requires non-empty work-held-out val and test splits")
    for split_name, split_dataset in (("val", val_dataset), ("test", test_dataset)):
        observed_styles = {sample.style for sample in split_dataset.samples}
        if observed_styles != {"manga", "comics"}:
            raise ValueError(f"RQ2 {split_name} split must contain both styles; found {sorted(observed_styles)}")

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "resume"}
    config["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config["task"] = "order_remaining_given_first"
    config.update({"effective_train_puzzles": budget, "train_style_counts": {style: sum(train_base.samples[i].style == style for i in selected) for style in ("manga", "comics")}, "selected_sequence_ids": [train_base.samples[i].sequence_id for i in selected]})
    run_id = args.run_id or f"{args.condition}_seed{args.seed}"
    run_dir = args.run_root / run_id
    logger = ExperimentLogger(run_dir, config, resume=args.resume is not None)
    save_startup_config(args, run_dir / "checkpoints")
    device = torch.device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)

    solver = _build_solver(args)
    if not args.load_in_4bit and args.device_map is None:
        solver.backbone.to(device)
    solver.experiment_logger = logger
    optimizer = torch.optim.AdamW([p for p in solver.backbone.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    global_step = 0
    best_val = -1.0
    if args.resume is not None:
        state = ExperimentLogger.load_checkpoint(args.resume, optimizer=optimizer, map_location=device)
        solver.load_trainable_state_dict(state["model"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("extra", {}).get("best_val", -1.0))
        logger.log(f"resumed checkpoint={args.resume} epoch={start_epoch} global_step={global_step}")
    logger.log(f"start condition={args.condition} styles={config['train_style_counts']} budget={budget} train_works={len({train_base.samples[i].work_id for i in selected})}")

    for epoch in range(start_epoch, args.epochs + 1):
        solver.backbone.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = randomize_training_batch(batch, use_layout=False)
            images = batch["panel_images"].to(device)
            target = batch["target_order"].to(device)
            loss = solver.compute_loss(images, batch.get("dialog_texts"), target) / args.grad_accum_steps
            loss.backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in solver.backbone.parameters() if p.requires_grad], args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.item() * args.grad_accum_steps))
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_metrics = evaluate(solver, val_loader, device, "val")
        record = {"type": "epoch", "epoch": epoch, "global_step": global_step, "condition": args.condition, "train_loss": train_loss, "val": val_metrics, "optimizer_steps_this_epoch": math.ceil(len(train_loader) / args.grad_accum_steps)}
        logger.log_metrics(record)
        logger.log(f"epoch={epoch}/{args.epochs} step={global_step} train_loss={train_loss:.6f} val_exact={val_metrics['exact_match']:.4f} val_pairwise={val_metrics['pairwise_accuracy']:.4f}")
        logger.save_checkpoint("latest.pt", model_state=solver.trainable_state_dict(), optimizer=optimizer, epoch=epoch, global_step=global_step, extra={"best_val": best_val})
        logger.save_checkpoint(f"epoch_{epoch:04d}.pt", model_state=solver.trainable_state_dict(), optimizer=optimizer, epoch=epoch, global_step=global_step, extra={"best_val": best_val})
        if val_metrics["exact_match"] > best_val:
            best_val = val_metrics["exact_match"]
            logger.save_checkpoint("best.pt", model_state=solver.trainable_state_dict(), optimizer=optimizer, epoch=epoch, global_step=global_step, extra={"best_val": best_val})
    state = ExperimentLogger.load_checkpoint(run_dir / "checkpoints" / "best.pt", restore_rng=False, map_location=device)
    solver.load_trainable_state_dict(state["model"])
    test_metrics = evaluate(solver, test_loader, device, "test")
    logger.log_metrics({"type": "final_test", "condition": args.condition, "selected_epoch": state["epoch"], "test": test_metrics})
    logger.log(f"final_test selected_epoch={state['epoch']} exact={test_metrics['exact_match']:.4f} pairwise={test_metrics['pairwise_accuracy']:.4f}")
    logger.save_checkpoint("final.pt", model_state=solver.trainable_state_dict(), optimizer=optimizer, epoch=state["epoch"], global_step=global_step, extra={"test": test_metrics, "best_val": best_val})


if __name__ == "__main__":
    main()
