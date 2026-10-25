"""Smoke-test the Made2Order panel-ordering solver on real local data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(".")

from Solver.agent.made2order_set2order_solver import Made2OrderPanelOrderingSolver
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Made2Order solver on a small dataset slice.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Mangazero/ordering_dataset"))
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)

    full = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        image_size=(args.image_size, args.image_size),
        use_layout=False,
    )
    if len(full) == 0:
        raise SystemExit(f"dataset {args.dataset_dir} is empty")
    dataset = torch.utils.data.Subset(full, list(range(min(args.max_samples, len(full)))))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_panel_ordering_batch)

    model = Made2OrderPanelOrderingSolver(
        panel_count=args.panel_count,
        image_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        images = batch["panel_images"].to(device)
        target = batch["target_order"].to(device)
        scores, attn = model(images)
        loss = model.ranking_loss(scores, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pred = model.predict_order(images)
        metrics = model.order_metrics(pred, target)
        print(
            f"step={step} loss={float(loss.detach()):.4f} "
            f"exact={metrics['exact_match']:.3f} pos={metrics['position_accuracy']:.3f} "
            f"scores={tuple(scores.shape)} attn={tuple(attn.shape)}"
        )
    print("smoke ok")


if __name__ == "__main__":
    main()