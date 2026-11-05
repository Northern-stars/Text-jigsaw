"""Shared checkpoint-backed evaluation helpers for RQ3 and RQ4."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Solver.agent.qwen_vl_ordering_solver import QwenVLOrderingSolver
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import ordering_metrics
from experiment.run_logger import ExperimentLogger


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="RQ2 best.pt or compatible logger checkpoint")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--panel-count", type=int, default=6)
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
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)


def build_solver(args: argparse.Namespace) -> QwenVLOrderingSolver:
    solver = QwenVLOrderingSolver(
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
        lora_target_modules=tuple(v.strip() for v in args.lora_target_modules.split(",") if v.strip()),
        max_new_tokens=args.max_new_tokens,
    )
    state = ExperimentLogger.load_checkpoint(args.checkpoint, restore_rng=False, map_location=args.device)
    solver.load_trainable_state_dict(state["model"])
    if not args.load_in_4bit and args.device_map is None:
        solver.backbone.to(torch.device(args.device))
    solver.backbone.eval()
    return solver


def load_dataset(manifest: Path, split: str, args: argparse.Namespace) -> MangaZeroPanelOrderingDataset:
    return MangaZeroPanelOrderingDataset(
        manifest.parent,
        split=split,
        image_size=(args.image_width, args.image_height),
        use_layout=False,
        allow_empty=True,
        manifest_path=manifest,
    )


def snapshot_checkpoint(source: Path, run_dir: Path) -> dict[str, str]:
    """Copy the evaluated weights into the run and return stable provenance."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = run_dir / "checkpoints" / f"source_{source.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    copied_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if copied_digest != digest:
        raise IOError(f"Checkpoint copy hash mismatch: {source} -> {target}")
    return {"source_checkpoint": str(source.resolve()), "checkpoint_sha256": digest, "checkpoint_snapshot": str(target)}


def run_condition(
    solver: QwenVLOrderingSolver,
    dataset: MangaZeroPanelOrderingDataset,
    args: argparse.Namespace,
    logger: ExperimentLogger,
    *,
    condition: str,
    prompt_factory: Callable[[list[Any], list[str]], Any],
    include_images: bool,
    include_text: bool,
    seed: int = 0,
    extra: Callable[[list[int], list[int], str, bool], dict[str, Any]] | None = None,
    answer_parser: Callable[[Any, int], list[int]] | None = None,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)
    scores: dict[str, list[float]] = {"exact_match": [], "position_accuracy": [], "pairwise_accuracy": [], "valid_output": []}
    failures: dict[str, int] = {}
    solver.backbone.eval()
    for batch in loader:
        images = batch["panel_images"].to(torch.device(args.device))
        for i in range(images.shape[0]):
            sequence_id = batch["sequence_id"][i]
            panel_count = images.shape[1]
            seed_value = int(hashlib.sha256(f"{seed}:{sequence_id}".encode("utf-8")).hexdigest()[:16], 16)
            display_order = list(range(panel_count))
            random.Random(seed_value).shuffle(display_order)
            original_to_display = {original: shown for shown, original in enumerate(display_order)}
            panels = [images[i, j] for j in display_order]
            texts = [batch["dialog_texts"][i][j] for j in display_order] if include_text else None
            target_original = batch["target_order"][i].tolist()
            target = [original_to_display[int(panel)] for panel in target_original]
            if not include_images:
                panels = [None] * len(panels)
            raw = ""
            try:
                prompt = prompt_factory(panels, texts)
                raw = solver.vlm_infer(prompt)
                prediction = (answer_parser or solver.answer_generator)(raw, len(panels))
                solver._validate(prediction, len(panels))
                valid = True
                metrics = ordering_metrics(torch.tensor(prediction).unsqueeze(0), torch.tensor(target).unsqueeze(0))
                err = ""
            except Exception as exc:
                raw = raw or f"{type(exc).__name__}: {exc}"
                prediction = []
                valid = False
                err = type(exc).__name__
                failures[err] = failures.get(err, 0) + 1
                metrics = {"exact_match": 0.0, "position_accuracy": 0.0, "pairwise_accuracy": 0.0}
            for key, value in metrics.items():
                scores[key].append(float(value))
            scores["valid_output"].append(float(valid))
            record = {
                "type": "prediction", "condition": condition,
                "sequence_id": sequence_id, "style": batch["style"][i],
                "work_id": batch["work_id"][i], "display_order_original_panel_ids": display_order,
                "target": target, "prediction": prediction if valid else None, "raw_output": str(raw),
                "metrics": metrics, "valid_output": valid, "error_type": err,
            }
            if extra:
                record.update(extra(prediction, record["target"], str(raw), valid))
            logger.log_metrics(record)
    result: dict[str, Any] = {key: float(np.mean(values)) if values else 0.0 for key, values in scores.items()}
    result["parse_failures"] = failures
    return result
