"""Shared contract: input slot 0 is given; only slots 1..K-1 are predicted.

Full orders retain the leading zero for compatibility with dataset labels and
renderers. Losses and metrics exclude that known prefix.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

TASK = "order_remaining_given_first"


def validate_given_start(target: Tensor) -> None:
    if target.ndim != 2 or target.size(1) < 2:
        raise ValueError("Given-start puzzles require at least two panels per sample")
    expected = torch.arange(target.size(1), device=target.device).expand_as(target)
    if not torch.equal(target.sort(dim=1).values, expected) or torch.any(target[:, 0] != 0):
        raise ValueError("Expected a full permutation starting with given input slot 0")


def with_given_start(item: dict[str, Any]) -> dict[str, Any]:
    """Move only the opening panel, retaining the shuffled order of all others."""
    target = item["target_order"].long()
    count = target.numel()
    if count < 2 or sorted(target.tolist()) != list(range(count)):
        raise ValueError("Expected a permutation of at least two input panels")
    first = int(target[0])
    display = torch.tensor([first] + [i for i in range(count) if i != first], device=target.device)
    result = dict(item)
    for key in ("panel_images", "layout_features", "input_panel_indices"):
        if key in item:
            result[key] = item[key].index_select(0, display.to(item[key].device))
    for key in ("panels", "dialog_texts"):
        if key in item:
            result[key] = [item[key][i] for i in display.tolist()]
    result["target_order"] = display.argsort()[target]
    if "input_panel_indices" not in item:
        result["input_panel_indices"] = display
    result["given_first_index"] = 0
    result["task"] = TASK
    return result


def shuffle_remaining_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Restore canonical order, then shuffle only the continuation in every modality."""
    target = batch["target_order"].long()
    batch_size, count = target.shape
    if count < 2:
        raise ValueError("Given-start puzzles require at least two panels")
    permutation = torch.stack([
        torch.cat([torch.zeros(1, dtype=torch.long), torch.randperm(count - 1) + 1])
        for _ in range(batch_size)
    ]).to(target.device)
    display = target.gather(1, permutation)
    result = dict(batch)
    for key in ("panel_images", "layout_features", "input_panel_indices"):
        if key in batch:
            value = batch[key]
            index = display.to(value.device).reshape(*display.shape, *([1] * (value.ndim - 2)))
            result[key] = value.gather(1, index.expand(*display.shape, *value.shape[2:]))
    for key in ("panels", "dialog_texts"):
        if key in batch:
            result[key] = [[items[i] for i in order] for items, order in zip(batch[key], display.tolist())]
    result["target_order"] = permutation.argsort(dim=1)
    result["given_first_index"] = torch.zeros(batch_size, dtype=torch.long, device=target.device)
    result["task"] = [TASK] * batch_size
    return result


def remaining_order_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    validate_given_start(target)
    if pred.shape != target.shape or torch.any(pred[:, 0] != 0):
        raise ValueError("Predictions must retain the given opening panel in slot 0")
    expected = torch.arange(pred.size(1), device=pred.device).expand_as(pred)
    if not torch.equal(pred.sort(dim=1).values, expected):
        raise ValueError("Predictions must be full permutations")
    pred, target = pred[:, 1:], target[:, 1:]
    matches = pred.eq(target)
    ranks = pred.argsort(dim=1)
    # IDs are 1..K-1, so subtract one when looking up their predicted ranks.
    target_ranks = ranks.gather(1, target - 1)
    pairs = torch.triu_indices(target.size(1), target.size(1), offset=1, device=target.device)
    pairwise = (target_ranks[:, pairs[0]] < target_ranks[:, pairs[1]]).float().mean() if pairs.size(1) else matches.float().mean()
    return {
        "exact_match": matches.all(dim=1).float().mean().item(),
        "position_accuracy": matches.float().mean().item(),
        "pairwise_accuracy": pairwise.item(),
    }
