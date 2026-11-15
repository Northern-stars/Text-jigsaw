"""One ordering task: known opening/ending panels, predict the middle only.

Targets are reading-position -> input-slot permutations, never per-slot ranks.
Images, dialogue, layout and provenance always move together.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

TASK = "order_middle_given_first_last"


def validate_order(order: Tensor) -> None:
    if order.ndim != 2 or order.size(1) < 3 or order.dtype != torch.long:
        raise ValueError("Endpoint puzzles require long permutations [B,K] with K >= 3")
    expected = torch.arange(order.size(1), device=order.device).expand_as(order)
    if not torch.equal(order.sort(dim=1).values, expected):
        raise ValueError("Expected full permutations of input slots")
    if torch.any(order[:, 0] != 0) or torch.any(order[:, -1] != order.size(1) - 1):
        raise ValueError("Expected given opening slot 0 and ending slot K-1")


def gather_batch_by_order(value: Tensor, order: Tensor) -> Tensor:
    index = order.to(value.device).reshape(*order.shape, *([1] * (value.ndim - 2)))
    return value.gather(1, index.expand(*order.shape, *value.shape[2:]))


def reorder_batch(batch: dict[str, Any], display: Tensor) -> dict[str, Any]:
    result = dict(batch)
    for key in ("panel_images", "layout_features", "input_panel_indices"):
        if key in batch:
            result[key] = gather_batch_by_order(batch[key], display)
    for key in ("panels", "dialog_texts"):
        if key in batch:
            result[key] = [[items[i] for i in row] for items, row in zip(batch[key], display.tolist())]
    result["target_order"] = display.argsort(dim=1).gather(1, batch["target_order"].to(display.device))
    return result


def with_given_endpoints(item: dict[str, Any]) -> dict[str, Any]:
    target = item["target_order"].long()
    count = target.numel()
    if count < 3 or sorted(target.tolist()) != list(range(count)):
        raise ValueError("Endpoint puzzles require a permutation of at least three panels")
    display = torch.tensor(endpoint_display_order(target.tolist()), device=target.device)
    result = dict(item)
    for key in ("panel_images", "layout_features", "input_panel_indices"):
        if key in item:
            result[key] = item[key].index_select(0, display.to(item[key].device))
    for key in ("panels", "dialog_texts"):
        if key in item:
            result[key] = [item[key][i] for i in display.tolist()]
    result["target_order"] = display.argsort()[target]
    result.setdefault("input_panel_indices", display)
    result.update(given_first_index=0, given_last_index=count - 1, task=TASK)
    return result


def endpoint_display_order(target_order, rng=None):
    """Place GT endpoints around the middle, optionally randomising the latter."""
    count = len(target_order)
    if count < 3 or sorted(target_order) != list(range(count)):
        raise ValueError("Endpoint puzzles require a permutation of at least three panels")
    first, last = target_order[0], target_order[-1]
    middle = [index for index in range(count) if index not in (first, last)]
    if rng is not None:
        rng.shuffle(middle)
    return [first, *middle, last]


def shuffle_middle_batch(batch: dict[str, Any]) -> dict[str, Any]:
    target = batch["target_order"]
    validate_order(target)
    count = target.size(1)
    positions = torch.stack([
        torch.cat([torch.tensor([0]), torch.randperm(count - 2) + 1, torch.tensor([count - 1])])
        for _ in range(target.size(0))
    ]).to(target.device)
    return reorder_batch(batch, target.gather(1, positions))


def ordering_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    validate_order(target)
    validate_order(pred)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    pred, target = pred[:, 1:-1], target[:, 1:-1]
    matches = pred.eq(target)
    ranks = pred.argsort(dim=1).gather(1, target - 1)
    pairs = torch.triu_indices(target.size(1), target.size(1), offset=1, device=target.device)
    pairwise = (ranks[:, pairs[0]] < ranks[:, pairs[1]]).float().mean() if pairs.size(1) else matches.float().mean()
    return {"exact_match": matches.all(dim=1).float().mean().item(),
            "position_accuracy": matches.float().mean().item(), "pairwise_accuracy": pairwise.item()}


class OrderingPuzzle:
    """Dispatch solver-specific training/inference through the same puzzle API."""

    def __init__(self, solver):
        self.solver = solver

    def training_loss(self, batch):
        validate_order(batch["target_order"])
        return self.solver.training_loss(batch)

    def infer(self, batch):
        # Inference deliberately exposes no GT order or original panel metadata.
        inputs = {key: batch[key] for key in ("panel_images", "dialog_texts", "layout_features") if key in batch}
        prediction = self.solver.predict(inputs)
        validate_order(prediction)
        return prediction


# Compatibility for earlier first-only studies.
FIRST_ONLY_TASK = "order_remaining_given_first"

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
    result["task"] = FIRST_ONLY_TASK
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
    result["task"] = [FIRST_ONLY_TASK] * batch_size
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
