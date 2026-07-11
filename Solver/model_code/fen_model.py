"""Multimodal model with independent image and text encoders.

The model is intentionally encoder-agnostic: pass in any two ``nn.Module``
encoders, and this wrapper will call them, normalize their output shape, project
both modalities to a shared dimension, fuse them, and produce task logits.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn


EncoderInputs = Union[Tensor, Sequence[Any], Mapping[str, Any]]


class BasicTransformerTextEncoder(nn.Module):
    """A compact Transformer encoder for tokenized text.

    The forward method follows the common HuggingFace-style input names so it
    can be used with ``FenModel`` through a dict:

    ``model(image_inputs, {"input_ids": ids, "attention_mask": mask})``
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_length: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.output_dim = embed_dim
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.use_cls_token = use_cls_token

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )
        position_count = max_length + (1 if use_cls_token else 0)
        self.position_embedding = nn.Embedding(position_count, embed_dim)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        )
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

        self._reset_parameters()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encode token ids.

        Args:
            input_ids: Long tensor with shape ``[batch, seq_len]``.
            attention_mask: Optional mask with 1/True for valid tokens and
                0/False for padding.
            key_padding_mask: Optional PyTorch-style mask with True for padding.
                If provided, it takes precedence over ``attention_mask``.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {input_ids.shape}"
            )

        batch_size, seq_len = input_ids.shape
        extra_token_count = 1 if self.use_cls_token else 0
        total_length = seq_len + extra_token_count
        max_total_length = self.max_length + extra_token_count
        if total_length > max_total_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_length={self.max_length}"
            )

        hidden = self.token_embedding(input_ids)

        if self.use_cls_token:
            assert self.cls_token is not None
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            hidden = torch.cat([cls_token, hidden], dim=1)

        positions = torch.arange(total_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        padding_mask = self._build_padding_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.output_norm(hidden)

        if self.use_cls_token:
            pooled = hidden[:, 0]
        else:
            pooled = self._masked_mean_pool(hidden, padding_mask)

        return {
            "last_hidden_state": hidden,
            "pooler_output": self.pooler(pooled),
        }

    def _build_padding_mask(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        if key_padding_mask is not None:
            padding_mask = key_padding_mask.bool()
        elif attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = input_ids.eq(self.padding_idx)

        if self.use_cls_token:
            cls_padding = torch.zeros(
                padding_mask.size(0),
                1,
                dtype=torch.bool,
                device=padding_mask.device,
            )
            padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        return padding_mask

    @staticmethod
    def _masked_mean_pool(hidden: Tensor, padding_mask: Tensor) -> Tensor:
        valid_mask = ~padding_mask
        weights = valid_mask.unsqueeze(-1).type_as(hidden)
        summed = (hidden * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)


class MultimodalFenModel(nn.Module):
    """Two-branch multimodal model for image and text features.

    Args:
        image_encoder: Module that encodes image inputs.
        text_encoder: Module that encodes text inputs.
        image_feature_dim: Feature size produced by ``image_encoder``.
        text_feature_dim: Feature size produced by ``text_encoder``.
        num_outputs: Output dimension, for example class count or regression
            target count.
        hidden_dim: Shared feature dimension after modality projection.
        fusion: Fusion strategy. Supports ``"concat"``, ``"sum"``, and
            ``"gated"``.
        dropout: Dropout probability used in projection/fusion heads.
        freeze_image_encoder: If true, image encoder parameters are frozen.
        freeze_text_encoder: If true, text encoder parameters are frozen.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_feature_dim: int,
        text_feature_dim: int,
        num_outputs: int,
        hidden_dim: int = 256,
        fusion: str = "concat",
        dropout: float = 0.1,
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()

        if fusion not in {"concat", "sum", "gated"}:
            raise ValueError("fusion must be one of: 'concat', 'sum', 'gated'")

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion

        if freeze_image_encoder:
            self._set_requires_grad(self.image_encoder, False)
        if freeze_text_encoder:
            self._set_requires_grad(self.text_encoder, False)

        self.image_projection = self._projection_block(
            image_feature_dim, hidden_dim, dropout
        )
        self.text_projection = self._projection_block(
            text_feature_dim, hidden_dim, dropout
        )

        if fusion == "concat":
            fused_dim = hidden_dim * 2
            self.gate = None
        elif fusion == "gated":
            fused_dim = hidden_dim
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            fused_dim = hidden_dim
            self.gate = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(
        self,
        image_inputs: EncoderInputs,
        text_inputs: EncoderInputs,
        return_features: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Run both encoders and return logits.

        ``image_inputs`` and ``text_inputs`` can each be:
        - a Tensor, called as ``encoder(inputs)``;
        - a tuple/list, called as ``encoder(*inputs)``;
        - a dict, called as ``encoder(**inputs)``.
        """

        image_features = self._extract_features(
            self._call_encoder(self.image_encoder, image_inputs)
        )
        text_features = self._extract_features(
            self._call_encoder(self.text_encoder, text_inputs)
        )

        image_embedding = self.image_projection(image_features)
        text_embedding = self.text_projection(text_features)
        fused = self._fuse(image_embedding, text_embedding)
        logits = self.classifier(fused)

        if not return_features:
            return logits

        return {
            "logits": logits,
            "image_features": image_features,
            "text_features": text_features,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "fused_embedding": fused,
        }

    def _fuse(self, image_embedding: Tensor, text_embedding: Tensor) -> Tensor:
        if self.fusion == "concat":
            return torch.cat([image_embedding, text_embedding], dim=-1)

        if self.fusion == "sum":
            return image_embedding + text_embedding

        assert self.gate is not None
        gate = self.gate(torch.cat([image_embedding, text_embedding], dim=-1))
        return gate * image_embedding + (1.0 - gate) * text_embedding

    @staticmethod
    def _projection_block(
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _call_encoder(encoder: nn.Module, inputs: EncoderInputs) -> Any:
        if isinstance(inputs, Mapping):
            return encoder(**inputs)
        if isinstance(inputs, (tuple, list)):
            return encoder(*inputs)
        return encoder(inputs)

    @staticmethod
    def _extract_features(encoder_output: Any) -> Tensor:
        """Convert common encoder outputs into a 2D ``[batch, feature]`` tensor."""

        if isinstance(encoder_output, Tensor):
            features = encoder_output
        elif isinstance(encoder_output, Mapping):
            features = MultimodalFenModel._extract_from_mapping(encoder_output)
        elif hasattr(encoder_output, "pooler_output"):
            features = encoder_output.pooler_output
        elif hasattr(encoder_output, "last_hidden_state"):
            features = encoder_output.last_hidden_state[:, 0]
        elif isinstance(encoder_output, (tuple, list)) and encoder_output:
            features = encoder_output[0]
        else:
            raise TypeError(
                "Encoder output must be a Tensor, non-empty tuple/list, mapping, "
                "or object with pooler_output/last_hidden_state."
            )

        if not isinstance(features, Tensor):
            raise TypeError("Extracted encoder features must be a torch.Tensor.")

        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features[:, 0]
        elif features.ndim > 4:
            features = features.flatten(start_dim=1)

        if features.ndim != 2:
            raise ValueError(
                "Encoder features must be reducible to shape [batch, feature], "
                f"got {tuple(features.shape)}."
            )

        return features

    @staticmethod
    def _extract_from_mapping(encoder_output: Mapping[str, Any]) -> Tensor:
        for key in ("pooler_output", "features", "embedding", "embeddings", "logits"):
            value = encoder_output.get(key)
            if isinstance(value, Tensor):
                return value

        last_hidden_state = encoder_output.get("last_hidden_state")
        if isinstance(last_hidden_state, Tensor):
            return last_hidden_state[:, 0]

        raise KeyError(
            "Mapping encoder output must contain one of: pooler_output, features, "
            "embedding, embeddings, logits, last_hidden_state."
        )

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


# Backward-friendly alias for shorter imports.
FenModel = MultimodalFenModel
