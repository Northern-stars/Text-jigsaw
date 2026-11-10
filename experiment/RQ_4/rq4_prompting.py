"""Compare direct, chain-of-thought, and structured-evidence prompting."""

from __future__ import annotations
# Resolve the project config package when this file is run directly.
import sys as _cli_sys
from pathlib import Path as _CliPath
_cli_root = next((p for p in _CliPath(__file__).resolve().parents if (p / "config" / "cli.py").is_file()), None)
if _cli_root is not None and str(_cli_root) not in _cli_sys.path:
    _cli_sys.path.insert(0, str(_cli_root))
from config.cli import parse_args_with_config

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Solver.agent.vlm_ordering_solver import build_ordering_messages, parse_ordering_answer
from experiment.eval_utils import add_model_args, build_solver, load_dataset, run_condition, snapshot_checkpoint
from experiment.run_logger import ExperimentLogger


MODES = ("direct", "cot", "structured")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--seed", type=int, default=0)
    add_model_args(parser)
    return parse_args_with_config(parser, script_file=__file__)


def prompt_for(mode: str, start_index: int):
    def make(panels, texts):
        if mode == "direct":
            return build_ordering_messages(panels, texts, start_index=start_index)
        if mode == "cot":
            system = "Think step by step about character continuity, actions, scene changes, and dialogue. Then output only the final line: FINAL ORDER: [panel numbers]."
            return build_ordering_messages(panels, texts, start_index=start_index, system_prompt=system)
        system = (
            "Provide concise observable evidence for each panel (characters, action, scene transition, and text clues), "
            "then finish with a line exactly in the form ORDER: [panel numbers]."
        )
        return build_ordering_messages(panels, texts, start_index=start_index, system_prompt=system)
    return make


def structured_parser(raw: Any, count: int) -> list[int]:
    text = str(raw)
    match = re.search(r"ORDER\s*:\s*(\[[^\]]+\]|[0-9][0-9,\s]*)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("structured response has no ORDER line")
    return parse_ordering_answer(match.group(1), count)


def cot_parser(raw: Any, count: int) -> list[int]:
    match = re.search(r"FINAL\s+ORDER\s*:\s*(\[[^\]]+\]|[0-9][0-9,\s]*)", str(raw), flags=re.IGNORECASE)
    if not match:
        raise ValueError("CoT response has no FINAL ORDER line")
    return parse_ordering_answer(match.group(1), count)


def error_profile(prediction: list[int], target: list[int], raw: str, valid: bool) -> dict[str, Any]:
    if not valid:
        return {"error_type": "invalid_output"}
    if prediction == target:
        return {"error_type": "none"}
    wrong_positions = sum(a != b for a, b in zip(prediction, target))
    if wrong_positions == 2 and all(abs(a - b) == 1 for a, b in zip(prediction, target) if a != b):
        category = "local_adjacent_swap"
    elif prediction[0] != target[0] or prediction[-1] != target[-1]:
        category = "boundary_or_transition"
    else:
        category = "other_permutation_error"
    return {"error_type": category}


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("RQ4 currently requires --batch-size 1 for reproducible per-puzzle error profiles")
    torch.manual_seed(args.seed)
    solver = build_solver(args)
    dataset = load_dataset(args.manifest, args.split, args)
    if not len(dataset):
        raise ValueError(f"manifest has no {args.split} samples")
    run_dir = args.run_root / args.run_id
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"experiment": "RQ4_prompting", "modes": list(MODES), "same_checkpoint": True, "same_decoding_budget": True})
    config.update(snapshot_checkpoint(args.checkpoint, run_dir))
    logger = ExperimentLogger(run_dir, config)
    results = {}
    for mode in MODES:
        result = run_condition(
            solver, dataset, args, logger, condition=mode,
            prompt_factory=prompt_for(mode, solver.start_index), include_images=True, include_text=True,
            seed=args.seed, extra=error_profile,
            answer_parser=structured_parser if mode == "structured" else cot_parser if mode == "cot" else None,
        )
        results[mode] = result
        logger.log(f"mode={mode} exact={result['exact_match']:.4f} pairwise={result['pairwise_accuracy']:.4f} valid={result['valid_output']:.4f}")
    logger.log_metrics({"type": "final", "experiment": "RQ4", "split": args.split, "results": results})
    (run_dir / "rq4_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
