"""Self-supervised Jigsaw-style solver for MangaZero panel ordering.

The model follows the classic jigsaw objective:

1. Recover the canonical panel sequence from a MangaZero puzzle label.
2. Apply one fixed candidate permutation to that canonical sequence.
3. Predict the candidate permutation id from the shuffled panels.

The encoder is shared across all panels. The classifier only predicts from a
bounded candidate bank, so its output size does not grow beyond
``max_permutations``.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_permutation_bank(
    panel_count: int,
    max_permutations: int | None = 1000,
    seed: int = 0,
) -> list[tuple[int, ...]]:
    """Build a deterministic candidate bank without materializing all K! orders."""
    if panel_count <= 1:
        raise ValueError("panel_count must be greater than 1")

    total = math.factorial(panel_count)
    limit = total if max_permutations is None else int(max_permutations)
    if limit <= 0:
        raise ValueError("max_permutations must be positive or None")
    if limit >= total:
        return list(itertools.permutations(range(panel_count)))

    rng = random.Random(seed)
    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    # Include identity so the bank has a stable canonical class.
    identity = tuple(range(panel_count))
    selected.append(identity)
    seen.add(identity)
    while len(selected) < limit:
        candidate = tuple(rng.sample(range(panel_count), panel_count))
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    return selected


def inverse_permutation(permutation: Tensor) -> Tensor:
    """Return inverse orders for a tensor shaped [..., K]."""
    if permutation.ndim < 1:
        raise ValueError("permutation must have at least one dimension")
    inverse = torch.empty_like(permutation)
    positions = torch.arange(permutation.size(-1), device=permutation.device)
    positions = positions.expand_as(permutation)
    inverse.scatter_(-1, permutation, positions)
    return inverse


class JigsawPanelCNN(nn.Module):
    """Compact shared CNN encoder for individual MangaZero panels."""

    def __init__(self, output_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.features(images))


class MangaZeroJigsawSolver(nn.Module):
    """Classify a fixed bank of panel permutations."""

    def __init__(
        self,
        panel_count: int,
        max_permutations: int | None = 1000,
        permutation_seed: int = 0,
        feature_dim: int = 256,
        hidden_dim: int = 1024,
        layout_dim: int = 10,
        use_layout: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if panel_count <= 1:
            raise ValueError("panel_count must be greater than 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.panel_count = panel_count
        self.use_layout = use_layout
        self.permutation_seed = permutation_seed
        permutations = build_permutation_bank(panel_count, max_permutations, permutation_seed)
        self.num_classes = len(permutations)
        self.feature_dim = feature_dim

        bank = torch.tensor(permutations, dtype=torch.long)
        self.register_buffer("permutation_bank", bank, persistent=True)
        self._class_index = {
            permutation: index
            for index, permutation in enumerate(permutations)
        }

        self.panel_encoder = JigsawPanelCNN(output_dim=feature_dim, dropout=dropout)
        self.layout_projection = nn.Linear(layout_dim, feature_dim) if use_layout else None
        self.panel_position_embedding = nn.Parameter(torch.zeros(1, panel_count, feature_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(panel_count * feature_dim),
            nn.Linear(panel_count * feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self._reset_parameters()

    def forward(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
    ) -> Tensor:
        if panel_images.ndim != 5:
            raise ValueError(f"panel_images must have shape [B,K,3,H,W], got {tuple(panel_images.shape)}")
        batch_size, panel_count, channels, height, width = panel_images.shape
        if panel_count != self.panel_count:
            raise ValueError(f"expected panel_count={self.panel_count}, got {panel_count}")
        flat_images = panel_images.reshape(batch_size * panel_count, channels, height, width)
        features = self.panel_encoder(flat_images).view(batch_size, panel_count, self.feature_dim)
        if self.use_layout:
            if layout_features is None:
                raise ValueError("layout_features are required when use_layout=True")
            if layout_features.shape[:2] != (batch_size, panel_count):
                raise ValueError(
                    "layout_features must have shape [B,K,D], "
                    f"got {tuple(layout_features.shape)}"
                )
            features = features + self.layout_projection(layout_features)
        features = features + self.panel_position_embedding
        return self.classifier(features.reshape(batch_size, -1))

    def target_to_class_indices(self, target_order: Tensor) -> Tensor:
        """Map MangaZero ``canonical -> input`` labels to Jigsaw class ids."""
        if target_order.ndim != 2 or target_order.size(1) != self.panel_count:
            raise ValueError(
                f"target_order must have shape [B,{self.panel_count}], "
                f"got {tuple(target_order.shape)}"
            )
        # The Jigsaw class convention is input_position -> canonical_panel.
        # MangaZero stores the inverse convention: canonical_panel -> input_position.
        jigsaw_permutations = inverse_permutation(target_order)
        result = []
        for order in jigsaw_permutations.detach().cpu().tolist():
            permutation = tuple(int(value) for value in order)
            class_id = self._class_index.get(permutation)
            if class_id is None:
                raise ValueError(
                    f"Jigsaw permutation {list(permutation)} is not in the fixed "
                    f"permutation bank of {self.num_classes} classes"
                )
            result.append(class_id)
        return torch.tensor(result, dtype=torch.long, device=target_order.device)

    @torch.no_grad()
    def predict_permutation(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
    ) -> Tensor:
        logits = self(panel_images, layout_features=layout_features)
        class_ids = logits.argmax(dim=-1)
        return self.permutation_bank[class_ids]

    @torch.no_grad()
    def predict_order(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Return MangaZero target_order, not the jigsaw input permutation."""
        predicted_permutation = self.predict_permutation(
            panel_images,
            layout_features=layout_features,
        )
        return inverse_permutation(predicted_permutation)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.panel_position_embedding, mean=0.0, std=0.02)


def jigsaw_cross_entropy(
    logits: Tensor,
    target_order: Tensor,
    model: MangaZeroJigsawSolver,
) -> Tensor:
    class_targets = model.target_to_class_indices(target_order)
    return F.cross_entropy(logits, class_targets)


def recover_canonical_batch(
    panel_images: Tensor,
    panels: list[list[dict]],
    layout_features: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, list[list[dict]] | None]:
    """Restore canonical panel order using metadata instead of labels."""
    canonical_orders = []
    for sample_panels in panels:
        if not sample_panels:
            raise ValueError("Each sample must contain at least one panel")
        if len(sample_panels) != panel_images.size(1):
            raise ValueError(
                "panel metadata count does not match panel tensor count: "
                f"{len(sample_panels)} != {panel_images.size(1)}"
            )
        indexed_panels = list(enumerate(sample_panels))
        indexed_panels.sort(key=lambda item: canonical_panel_key(item[1]))
        canonical_orders.append([input_index for input_index, _ in indexed_panels])
    order = torch.tensor(
        canonical_orders,
        dtype=torch.long,
        device=panel_images.device,
    )
    canonical_images = gather_by_order(panel_images, order)
    canonical_layout = (
        gather_by_order(layout_features, order)
        if layout_features is not None
        else None
    )
    canonical_panels = reorder_nested(panels, order)
    return canonical_images, canonical_layout, canonical_panels


def canonical_panel_key(panel: dict) -> tuple[int, int, int]:
    """Return a stable original-order key for current and older samples."""
    if "original_window_index" in panel:
        return (0, int(panel["original_window_index"]), 0)
    if "global_order" in panel:
        return (1, int(panel["global_order"]), 0)
    if "panel_index_in_page" in panel:
        return (2, int(panel["panel_index_in_page"]), 0)
    raise ValueError(
        "Panel metadata must contain original_window_index, global_order, "
        "or panel_index_in_page for self-supervised canonicalization"
    )


def gather_by_order(tensor: Tensor, order: Tensor) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims, got {tuple(tensor.shape)}")
    index = order
    for _ in range(tensor.ndim - 2):
        index = index.unsqueeze(-1)
    index = index.expand(*order.shape, *tensor.shape[2:])
    return tensor.gather(1, index)


def reorder_nested(items: list[list[dict]], order: Tensor) -> list[list[dict]]:
    return [
        [sample_items[index] for index in sample_order]
        for sample_items, sample_order in zip(items, order.tolist())
    ]
