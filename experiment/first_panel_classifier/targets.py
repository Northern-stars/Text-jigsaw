"""Label helpers for first-panel classification."""

from __future__ import annotations

import torch
from torch import Tensor

LABEL_DEFINITION = "target_order_first_input_panel_index_v2"


def validate_checkpoint_label_definition(extra: dict, label_definition: str = LABEL_DEFINITION) -> None:
    """Reject weights trained against a different first-panel target."""
    if extra.get("label_definition") != label_definition:
        raise ValueError(
            "checkpoint uses an unknown or incompatible first-panel label definition; "
            "start a new run and retrain with target_order_first_input_panel_index_v2"
        )


def first_panel_targets(target_order: Tensor) -> Tensor:
    """Return the first input-slot index in each reading-order sequence."""
    if target_order.ndim != 2:
        raise ValueError(f"target_order must be [batch, panel_count], got {tuple(target_order.shape)}")
    if target_order.size(1) < 2:
        raise ValueError("target_order must contain at least two panels")
    sorted_values = torch.sort(target_order, dim=1).values
    expected = torch.arange(target_order.size(1), device=target_order.device).expand_as(target_order)
    if not torch.equal(sorted_values, expected):
        raise ValueError("each target_order row must be a permutation of panel indices")
    return target_order[:, 0]
