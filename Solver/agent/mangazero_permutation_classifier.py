"""Direct permutation classifier for MangaZero panel ordering.

This solver treats each possible panel order as one hard-coded class. For a
fixed panel count K, the classifier predicts one of K! lexicographic
permutation classes, then maps that class back to a full target_order.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from Solver.agent.mangazero_set2seq_solver import (
    HashedDialogTextEncoder,
    LightweightPanelViTEncoder,
    TorchvisionPretrainedViTEncoder,
)


class MangaZeroPermutationClassifier(nn.Module):
    """Encode shuffled panels and classify the whole ordering as one class."""

    def __init__(
        self,
        panel_count: int,
        solver_type: str = "visual",
        image_feature_dim: int = 256,
        image_size: tuple[int, int] = (224, 224),
        vit_patch_size: int = 16,
        vit_layers: int = 4,
        vit_num_heads: int = 8,
        vit_backbone: str = "lightweight",
        vit_pretrained: bool = True,
        vit_freeze: bool = False,
        text_feature_dim: int = 256,
        text_vocab_size: int = 8192,
        d_model: int = 256,
        layout_dim: int = 10,
        use_layout: bool = True,
        encoder_layers: int = 4,
        num_heads: int = 8,
        classifier_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if panel_count <= 1:
            raise ValueError("panel_count must be greater than 1")
        if solver_type not in {"visual", "text", "multimodal"}:
            raise ValueError("solver_type must be one of: visual, text, multimodal")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.panel_count = panel_count
        self.solver_type = solver_type
        self.use_visual = solver_type in {"visual", "multimodal"}
        self.use_text = solver_type in {"text", "multimodal"}
        self.use_layout = use_layout
        self.num_classes = math.factorial(panel_count)

        class_orders = torch.tensor(
            list(itertools.permutations(range(panel_count))),
            dtype=torch.long,
        )
        self.register_buffer("class_orders", class_orders, persistent=False)
        self._class_index = {
            tuple(order.tolist()): index
            for index, order in enumerate(class_orders)
        }

        self.image_encoder = None
        if self.use_visual:
            if vit_backbone == "pretrained":
                self.image_encoder = TorchvisionPretrainedViTEncoder(
                    output_dim=image_feature_dim,
                    model_name="vit_b_16",
                    pretrained=vit_pretrained,
                    freeze=vit_freeze,
                )
            elif vit_backbone == "lightweight":
                self.image_encoder = LightweightPanelViTEncoder(
                    output_dim=image_feature_dim,
                    patch_size=vit_patch_size,
                    image_size=image_size,
                    num_layers=vit_layers,
                    num_heads=vit_num_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError("vit_backbone must be one of: pretrained, lightweight")
        self.image_projection = nn.Linear(image_feature_dim, d_model) if self.use_visual else None

        self.text_encoder = (
            HashedDialogTextEncoder(output_dim=text_feature_dim, vocab_size=text_vocab_size)
            if self.use_text
            else None
        )
        self.text_projection = nn.Linear(text_feature_dim, d_model) if self.use_text else None
        self.layout_projection = nn.Linear(layout_dim, d_model) if use_layout else None

        self.input_slot_embedding = nn.Parameter(torch.zeros(1, panel_count, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        hidden_dim = int(classifier_hidden_dim or d_model * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self._reset_parameters()

    def forward(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        encoded = self.encode(panel_images, layout_features=layout_features, dialog_texts=dialog_texts)
        return self.classifier(encoded[:, 0])

    def encode(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        if panel_images.ndim != 5:
            raise ValueError(f"panel_images must have shape [B,K,3,H,W], got {tuple(panel_images.shape)}")
        batch_size, panel_count, channels, height, width = panel_images.shape
        if panel_count != self.panel_count:
            raise ValueError(f"expected panel_count={self.panel_count}, got {panel_count}")

        tokens = torch.zeros(
            batch_size,
            panel_count,
            self.input_slot_embedding.size(-1),
            device=panel_images.device,
            dtype=panel_images.dtype,
        )
        if self.use_visual:
            flat_images = panel_images.reshape(batch_size * panel_count, channels, height, width)
            visual_features = self.image_encoder(flat_images).view(batch_size, panel_count, -1)
            tokens = tokens + self.image_projection(visual_features)
        if self.use_text:
            if dialog_texts is None:
                raise ValueError("dialog_texts are required when solver_type uses text")
            text_features = self.text_encoder(dialog_texts, device=panel_images.device)
            tokens = tokens + self.text_projection(text_features).to(dtype=tokens.dtype)
        if self.use_layout:
            if layout_features is None:
                raise ValueError("layout_features are required when use_layout=True")
            tokens = tokens + self.layout_projection(layout_features)

        tokens = self.token_norm(tokens + self.input_slot_embedding)
        cls = self.cls_token.expand(batch_size, -1, -1).to(dtype=tokens.dtype)
        return self.encoder(torch.cat([cls, tokens], dim=1))

    def target_to_class_indices(self, target_order: Tensor) -> Tensor:
        if target_order.ndim != 2 or target_order.size(1) != self.panel_count:
            raise ValueError(f"target_order must have shape [B,{self.panel_count}], got {tuple(target_order.shape)}")
        indices = []
        for order in target_order.detach().cpu().tolist():
            key = tuple(int(value) for value in order)
            try:
                indices.append(self._class_index[key])
            except KeyError as exc:
                raise ValueError(f"target_order is not a valid permutation: {order}") from exc
        return torch.tensor(indices, dtype=torch.long, device=target_order.device)

    @torch.no_grad()
    def predict_order(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        logits = self(
            panel_images,
            layout_features=layout_features,
            dialog_texts=dialog_texts,
        )
        class_ids = logits.argmax(dim=-1)
        return self.class_orders.to(device=class_ids.device)[class_ids]

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.input_slot_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)


def permutation_cross_entropy(logits: Tensor, target_order: Tensor, model: MangaZeroPermutationClassifier) -> Tensor:
    class_targets = model.target_to_class_indices(target_order)
    return F.cross_entropy(logits, class_targets)
