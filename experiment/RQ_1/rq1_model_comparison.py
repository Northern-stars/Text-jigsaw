"""Run a layout-blind RQ1 model comparison through the shared VLM interface."""

from __future__ import annotations
# Resolve the project config package when this file is run directly.
import sys as _cli_sys
from pathlib import Path as _CliPath
_cli_root = next((p for p in _CliPath(__file__).resolve().parents if (p / "config" / "cli.py").is_file()), None)
if _cli_root is not None and str(_cli_root) not in _cli_sys.path:
    _cli_sys.path.insert(0, str(_cli_root))
from config.cli import parse_args_with_config

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch
from Solver.train_mangazero_panel_ordering import ordering_metrics
from experiment.RQ_1.model_adapters import MODEL_ADAPTERS, create_adapter
from experiment.run_logger import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--model-family", choices=sorted(MODEL_ADAPTERS), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--max-puzzles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parse_args_with_config(parser, script_file=__file__)


def _display_order(sequence_id: str, count: int, seed: int) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode("utf-8")).hexdigest()
    order = list(range(1, count))
    random.Random(int(digest[:16], 16)).shuffle(order)
    return [0] + order


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 to preserve per-puzzle prompt and latency records")
    dataset = MangaZeroPanelOrderingDataset(
        args.manifest.parent,
        split=args.split,
        image_size=(224, 224),
        use_layout=False,
        given_first=True,
        allow_empty=True,
        manifest_path=args.manifest,
    )
    if not len(dataset):
        raise ValueError(f"Manifest has no {args.split} samples")
    if args.max_puzzles is not None:
        if args.max_puzzles <= 0:
            raise ValueError("--max-puzzles must be positive")
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        dataset = Subset(dataset, indices[:args.max_puzzles])

    run_dir = args.run_root / args.run_id
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"task": "order_remaining_given_first", "experiment": "RQ1_model_comparison", "layout_blind": True, "adapter_protocol": "VLMOrderingSolver.solve"})
    logger = ExperimentLogger(run_dir, config)
    adapter_kwargs = {
        "panel_count": args.panel_count,
        "revision": args.model_revision,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
    }
    if args.model_family == "llama3.2-vision":
        adapter_kwargs["model_name"] = args.model_name
    solver = create_adapter(args.model_family, args.model_name, **adapter_kwargs)
    if args.model_family == "qwen2.5-vl" and not args.device_map:
        solver.backbone.to(torch.device(args.device))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)
    metrics = {"exact_match": [], "position_accuracy": [], "pairwise_accuracy": [], "valid_output": []}
    latencies = []
    failures: dict[str, int] = {}
    for batch in loader:
        images = batch["panel_images"][0]
        sequence_id = batch["sequence_id"][0]
        display_order = _display_order(sequence_id, images.shape[0], args.seed)
        panels = [images[index] for index in display_order]
        texts = [batch["dialog_texts"][0][index] for index in display_order]
        inverse = {original: shown for shown, original in enumerate(display_order)}
        target = [inverse[int(index)] for index in batch["target_order"][0].tolist()]
        started_at = time.perf_counter()
        try:
            result = solver.solve(panels, texts)
            score = ordering_metrics(torch.tensor(result.order).unsqueeze(0), torch.tensor(target).unsqueeze(0))
            valid = True
            error = ""
        except Exception as exc:
            result = type("FailedResult", (), {"order": [], "raw_output": f"{type(exc).__name__}: {exc}"})()
            score = {"exact_match": 0.0, "position_accuracy": 0.0, "pairwise_accuracy": 0.0}
            valid = False
            error = type(exc).__name__
            failures[error] = failures.get(error, 0) + 1
        latency_seconds = time.perf_counter() - started_at
        latencies.append(latency_seconds)
        for key, value in score.items():
            metrics[key].append(float(value))
        metrics["valid_output"].append(float(valid))
        logger.log_metrics({
            "type": "prediction", "model_family": args.model_family, "model_name": args.model_name,
            "sequence_id": sequence_id, "style": batch["style"][0], "work_id": batch["work_id"][0],
            "display_order_original_panel_ids": batch["input_panel_indices"][0, display_order].tolist(), "target": target,
            "task": "order_remaining_given_first", "given_first_index": 0,
            "prediction": result.order if valid else None, "raw_output": result.raw_output,
            "metrics": score, "valid_output": valid, "error_type": error,
            "latency_seconds": latency_seconds,
            "model_revision": args.model_revision,
            "provider_metadata": getattr(solver, "last_metadata", {}),
        })
    aggregate = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
    aggregate["parse_failures"] = failures
    aggregate["mean_latency_seconds"] = float(np.mean(latencies)) if latencies else 0.0
    logger.log_metrics({"type": "final", "results": aggregate})
    logger.log(f"model={args.model_family}:{args.model_name} split={args.split} exact={aggregate['exact_match']:.4f} pairwise={aggregate['pairwise_accuracy']:.4f}")
    (run_dir / "rq1_results.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
