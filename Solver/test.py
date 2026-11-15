"""Unified checkpoint evaluation; constructor settings come from checkpoint."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Solver.cli import parse_args
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset
from Solver.puzzle import TASK
from Solver.runtime import evaluate, make_loader, read_checkpoint, restore_checkpoint
from Solver.solvers import create_solver


def main(argv=None):
    args = parse_args(argv, testing=True)
    state = read_checkpoint(args.checkpoint)
    name = state["solver"]
    if args.solver is not None and args.solver != name:
        raise ValueError("--solver differs from the checkpoint solver")
    config = state["config"]
    if args.panel_count is not None and args.panel_count != config["panel_count"]:
        raise ValueError("--panel-count differs from checkpoint")
    for key in ("model_name", "decoder_dim", "decoder_layers", "encoder_layers", "num_heads", "use_lora", "use_layout"):
        if getattr(args, key) is not None:
            raise ValueError(f"--{key.replace('_', '-')} must be restored from the checkpoint, not overridden during testing")
    options = dict(config["model_options"])
    # Only placement/precision may change during evaluation, never architecture.
    if args.model_options:
        raise ValueError("Evaluation model options are restored from the checkpoint")
    for key in ("device_map", "torch_dtype", "load_in_4bit"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if name == "made2order":
        options["device"] = args.device
    solver = create_solver(name, options).place(args.device, dispatched=bool(options.get("device_map") or options.get("load_in_4bit")))
    restore_checkpoint(state, solver)
    dataset = MangaZeroPanelOrderingDataset(args.dataset_dir, split=args.split,
        image_size=(config["image_width"], config["image_height"]),
        use_layout=options.get("use_layout", False), given_endpoints=True)
    if any(sample.panel_count != config["panel_count"] for sample in dataset.samples):
        raise ValueError("Dataset panel count differs from checkpoint")
    metrics = evaluate(solver, make_loader(dataset, args), args.device)
    result = {"solver": name, "task": TASK, "split": args.split, "metrics": metrics}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
