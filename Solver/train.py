"""Unified ordering training: python -m Solver.train --solver NAME ..."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from config.cli import save_startup_config
from experiment.run_logger import ExperimentLogger
from Solver.cli import model_options, parse_args
from Solver.data.panel_ordering_dataset import load_panel_ordering_splits
from Solver.puzzle import TASK
from Solver.runtime import (evaluate, make_loader, read_checkpoint, restore_checkpoint,
                            save_checkpoint, seed_everything, train_epoch)
from Solver.solvers import create_solver


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    options = model_options(args)
    args.save_dir = args.save_dir or (args.resume.parent if args.resume else Path("Solver/checkpoints") / args.solver)
    config = json.loads(json.dumps({**vars(args), "model_options": options, "task": TASK}, default=str))
    state = read_checkpoint(args.resume) if args.resume else None
    if state is not None:
        prior = state["config"]
        for key in ("solver", "model_options", "dataset_dir", "batch_size", "grad_accum_steps", "seed", "image_width", "image_height"):
            if prior[key] != config[key]:
                raise ValueError(f"Resume configuration differs for {key}")
    logger_config = {key: value for key, value in config.items() if key not in ("resume", "epochs", "config_file")}
    logger = ExperimentLogger(args.save_dir, logger_config, resume=state is not None)
    save_startup_config(args, args.save_dir)
    datasets = load_panel_ordering_splits(args.dataset_dir, (args.image_width, args.image_height),
                                         options.get("use_layout", False), given_endpoints=True)
    for dataset in datasets:
        if any(sample.panel_count != args.panel_count for sample in dataset.samples):
            raise ValueError("Dataset panel count differs from --panel-count")
    loaders = [make_loader(dataset, args, shuffle=index == 0) for index, dataset in enumerate(datasets)]
    if loaders[0] is None:
        raise ValueError("Empty training split")
    solver = create_solver(args.solver, options).place(
        args.device, dispatched=bool(options.get("device_map") or options.get("load_in_4bit")))
    groups = solver.optimizer_groups(args)
    if not groups:
        raise ValueError("Solver has no trainable parameters; enable LoRA for a generative Qwen solver")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    start, best = 1, -1.0
    if state is not None:
        restore_checkpoint(state, solver, optimizer)
        start, best = state["epoch"] + 1, state["best_val"]
    if start > args.epochs:
        raise ValueError("--epochs must exceed the resumed checkpoint epoch")
    logger.log(f"solver={args.solver} task={TASK} splits={[len(ds) for ds in datasets]}")
    for epoch in range(start, args.epochs + 1):
        train = train_epoch(solver, loaders[0], optimizer, args)
        val = evaluate(solver, loaders[1], args.device)
        improved = bool(val) and val["exact_match"] > best
        if improved:
            best = val["exact_match"]
        logger.log_metrics({"epoch": epoch, "train": train, "val": val})
        logger.log(f"epoch={epoch}/{args.epochs} train_loss={train['loss']:.6f} val={val}")
        save = lambda name: save_checkpoint(args.save_dir / name, solver, optimizer, epoch=epoch, best_val=best, config=config)
        save("last.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save(f"epoch_{epoch:04d}.pt")
        if improved:
            save("best.pt")
    # Select on validation, then evaluate test once using the selected model.
    selected = args.save_dir / "best.pt"
    if not selected.exists():
        selected = args.save_dir / "last.pt"
    restore_checkpoint(read_checkpoint(selected), solver)
    test = evaluate(solver, loaders[2], args.device)
    logger.log_metrics({"type": "final_test", "checkpoint": str(selected), "test": test})
    logger.log(f"selected_checkpoint={selected} test={test}")
    (args.save_dir / "test.json").write_text(json.dumps(test, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
