"""Set-to-sequence Transformer solver for MangaZero panel ordering."""

from __future__ import annotations

import math
import hashlib
import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TorchvisionPretrainedViTEncoder(nn.Module):
    """Torchvision ViT feature extractor with optional pretrained weights."""

    WEIGHT_ENUMS = {
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights",
    }

    def __init__(
        self,
        output_dim: int = 256,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise ImportError("torchvision is required for pretrained ViT. Use --vit-backbone lightweight to disable it.") from exc

        if not hasattr(models, model_name):
            raise ValueError(f"Unsupported torchvision ViT model: {model_name}")
        model_fn = getattr(models, model_name)
        weights = None
        if pretrained:
            weight_enum_name = self.WEIGHT_ENUMS.get(model_name)
            weight_enum = getattr(models, weight_enum_name, None) if weight_enum_name else None
            if weight_enum is None:
                raise ValueError(f"No torchvision pretrained weight enum found for {model_name}")
            weights = weight_enum.DEFAULT

        self.backbone = model_fn(weights=weights)
        backbone_dim = int(getattr(self.backbone, "hidden_dim", output_dim))
        self.input_size = int(getattr(self.backbone, "image_size", 224))
        self.backbone.heads = nn.Identity()
        self.projection = nn.Identity() if backbone_dim == output_dim else nn.Linear(backbone_dim, output_dim)
        self.output_dim = output_dim

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)
        return self.projection(self.backbone(images))


class LightweightPanelViTEncoder(nn.Module):
    """Small ViT encoder used when pretrained torchvision ViT is disabled."""

    def __init__(
        self,
        output_dim: int = 256,
        patch_size: int = 16,
        image_size: tuple[int, int] = (224, 224),
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("ViT output_dim must be divisible by num_heads")
        if patch_size <= 0:
            raise ValueError("ViT patch_size must be positive")
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
        base_grid_h = max(1, image_size[0] // patch_size)
        base_grid_w = max(1, image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + base_grid_h * base_grid_w, output_dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_dim)
        self._reset_parameters()

    def forward(self, images: Tensor) -> Tensor:
        patches = self.patch_embedding(images)
        batch_size, _, grid_h, grid_w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.dropout(tokens + self._position_embedding(grid_h, grid_w))
        return self.norm(self.encoder(tokens)[:, 0])

    def _position_embedding(self, grid_h: int, grid_w: int) -> Tensor:
        base_patch_positions = self.position_embedding[:, 1:]
        base_grid_size = int(math.sqrt(base_patch_positions.size(1)))
        if base_grid_size * base_grid_size == base_patch_positions.size(1):
            base_grid_h = base_grid_w = base_grid_size
        else:
            base_grid_h = max(1, self.position_embedding.size(1) - 1)
            base_grid_w = 1
        if (grid_h, grid_w) == (base_grid_h, base_grid_w):
            return self.position_embedding
        patch_positions = base_patch_positions.transpose(1, 2).reshape(
            1,
            self.output_dim,
            base_grid_h,
            base_grid_w,
        )
        patch_positions = F.interpolate(
            patch_positions,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_positions = patch_positions.flatten(2).transpose(1, 2)
        return torch.cat([self.position_embedding[:, :1], patch_positions], dim=1)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)


class HashedDialogTextEncoder(nn.Module):
    """Dependency-free dialog encoder for PaddleOCR text."""

    def __init__(self, output_dim: int = 256, vocab_size: int = 8192) -> None:
        super().__init__()
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, output_dim, mode="mean")
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, dialog_texts: list[list[str]], device: torch.device) -> Tensor:
        batch_size = len(dialog_texts)
        panel_count = len(dialog_texts[0]) if batch_size > 0 else 0
        flat_texts = [text for sample_texts in dialog_texts for text in sample_texts]
        token_ids: list[int] = []
        offsets: list[int] = []
        for text in flat_texts:
            offsets.append(len(token_ids))
            token_ids.extend(self._tokenize_to_ids(text))
        if not token_ids:
            token_ids = [1]
            offsets = [0]
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offsets_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        features = self.embedding(ids_tensor, offsets_tensor)
        return self.norm(features).view(batch_size, panel_count, self.output_dim)

    def _tokenize_to_ids(self, text: str) -> list[int]:
        text = str(text or "").strip().lower()
        if not text:
            return [1]
        pieces = re.findall(r"\w+|[^\s]", text)
        if not pieces:
            pieces = list(text)
        return [self._stable_hash(piece) for piece in pieces[:128]]

    def _stable_hash(self, piece: str) -> int:
        digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()
        return 2 + (int(digest[:8], 16) % (self.vocab_size - 2))


class MangaZeroSetToSequenceSolver(nn.Module):
    """Encode an unordered fixed-size panel set and pointer-decode its order."""

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
        decoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if solver_type not in {"visual", "text", "multimodal"}:
            raise ValueError("solver_type must be one of: visual, text, multimodal")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.panel_count = panel_count
        self.solver_type = solver_type
        self.use_visual = solver_type in {"visual", "multimodal"}
        self.use_text = solver_type in {"text", "multimodal"}
        self.use_layout = use_layout
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
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pointer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def forward(
        self,
        panel_images: Tensor,
        target_order: Tensor,
        layout_features: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        memory = self.encode(panel_images, layout_features=layout_features, dialog_texts=dialog_texts)
        decoder_inputs = self._teacher_forcing_inputs(memory, target_order)
        causal_mask = torch.triu(
            torch.ones(self.panel_count, self.panel_count, device=panel_images.device, dtype=torch.bool),
            diagonal=1,
        )
        decoder_states = self.pointer_decoder(decoder_inputs, memory, tgt_mask=causal_mask)
        return self._pointer_logits(decoder_states, memory, target_order=target_order)

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
            self.type_embedding.size(-1),
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
        return self.set_encoder(tokens + self.type_embedding)

    def greedy_decode(
        self,
        panel_images: Tensor,
        layout_features: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        memory = self.encode(panel_images, layout_features=layout_features, dialog_texts=dialog_texts)
        batch_size = memory.size(0)
        selected = torch.empty(batch_size, 0, dtype=torch.long, device=memory.device)

        for _ in range(self.panel_count):
            decoder_inputs = self._decode_inputs_from_selected(memory, selected)
            causal_mask = torch.triu(
                torch.ones(decoder_inputs.size(1), decoder_inputs.size(1), device=memory.device, dtype=torch.bool),
                diagonal=1,
            )
            decoder_states = self.pointer_decoder(decoder_inputs, memory, tgt_mask=causal_mask)
            logits = self._pointer_logits(decoder_states[:, -1:], memory).squeeze(1)
            if selected.numel() > 0:
                logits.scatter_(1, selected, -torch.inf)
            next_index = logits.argmax(dim=-1, keepdim=True)
            selected = torch.cat([selected, next_index], dim=1)

        return selected

    def _teacher_forcing_inputs(self, memory: Tensor, target_order: Tensor) -> Tensor:
        batch_size = memory.size(0)
        bos = self.bos_token.expand(batch_size, 1, -1)
        prefix = target_order[:, :-1]
        if prefix.numel() == 0:
            return bos
        selected_tokens = memory.gather(
            dim=1,
            index=prefix.unsqueeze(-1).expand(-1, -1, memory.size(-1)),
        )
        return torch.cat([bos, selected_tokens], dim=1)

    def _decode_inputs_from_selected(self, memory: Tensor, selected: Tensor) -> Tensor:
        batch_size = memory.size(0)
        bos = self.bos_token.expand(batch_size, 1, -1)
        if selected.numel() == 0:
            return bos
        selected_tokens = memory.gather(
            dim=1,
            index=selected.unsqueeze(-1).expand(-1, -1, memory.size(-1)),
        )
        return torch.cat([bos, selected_tokens], dim=1)

    def _pointer_logits(
        self,
        decoder_states: Tensor,
        memory: Tensor,
        target_order: Optional[Tensor] = None,
    ) -> Tensor:
        queries = self.query_projection(self.output_norm(decoder_states))
        keys = self.key_projection(memory)
        logits = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        if target_order is None:
            return logits

        selected_mask = torch.zeros_like(logits, dtype=torch.bool)
        for step in range(self.panel_count):
            if step > 0:
                previous = target_order[:, :step]
                selected_mask[:, step, :].scatter_(1, previous, True)
        return logits.masked_fill(selected_mask, -torch.inf)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.bos_token, mean=0.0, std=0.02)


def pointer_cross_entropy(pointer_logits: Tensor, target_order: Tensor) -> Tensor:
    panel_count = pointer_logits.size(1)
    return F.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.size(-1)),
        target_order[:, :panel_count].reshape(-1),
    )
