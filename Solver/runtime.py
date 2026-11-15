"""Shared epoch runner, validation and reproducible checkpoint IO."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Solver.data.panel_ordering_dataset import collate_panel_ordering_batch
from Solver.puzzle import TASK, OrderingPuzzle, ordering_metrics
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, *, shuffle=False):
    if not len(dataset):
        return None
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, collate_fn=collate_panel_ordering_batch)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_epoch(solver, loader, optimizer, args):
    solver.train()
    puzzle = OrderingPuzzle(solver)
    optimizer.zero_grad(set_to_none=True)
    total_loss = total = updates = 0
    # Normalize each accumulation window by its actual sample count, including
    # the last incomplete batch/window, so gradients match one larger batch.
    window_samples = 0
    for index, batch in enumerate(tqdm(loader), 1):
        batch = move_batch(solver.prepare_training_batch(batch), args.device)
        samples = batch["panel_images"].size(0)
        loss = puzzle.training_loss(batch)
        (loss * samples).backward()
        total_loss += float(loss.detach()) * samples
        total += samples
        window_samples += samples
        if index % args.grad_accum_steps == 0 or index == len(loader):
            parameters = [p for p in solver.parameters() if p.requires_grad]
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
            updates += 1
    return {"loss": total_loss / total, "samples": total, "optimizer_steps": updates}


@torch.no_grad()
def evaluate(solver, loader, device):
    if loader is None:
        return {}
    solver.eval()
    puzzle = OrderingPuzzle(solver)
    totals = {key: 0.0 for key in ("exact_match", "position_accuracy", "pairwise_accuracy")}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = puzzle.infer(batch)
        metrics = ordering_metrics(pred, batch["target_order"].to(pred.device))
        samples = pred.size(0)
        for key, value in metrics.items():
            totals[key] += value * samples
        count += samples
    return {**{key: value / count for key, value in totals.items()}, "samples": count}


def save_checkpoint(path, solver, optimizer, *, epoch, best_val, config):
    state = {"schema_version": 1, "task": TASK, "solver": config["solver"], "config": config,
             "model": solver.checkpoint_state(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "best_val": best_val, "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1 or state.get("task") != TASK:
        raise ValueError("Checkpoint task/schema differs: retrain using the endpoint puzzle and unified entry point")
    return state


def restore_checkpoint(state, solver, optimizer=None):
    solver.load_checkpoint_state(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
