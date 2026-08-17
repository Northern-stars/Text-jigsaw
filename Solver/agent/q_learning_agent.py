"""DQN agent for the text jigsaw swap environment."""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from Solver.model_code.fen_model import (
    attention_fen_model,
    central_fen_model,
    dualstem_fen_model,
    fen_model,
)


PIECE_COUNT = 9
ACTION_COUNT = PIECE_COUNT * PIECE_COUNT
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class CharTokenizer:
    """Simple byte-level tokenizer for dependency-free text encoding."""

    pad_token_id = 0
    unk_token_id = 1

    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length
        self.vocab_size = 258

    def encode(self, text: str) -> list[int]:
        byte_values = text.encode("utf-8", errors="replace")
        token_ids = [value + 2 for value in byte_values[: self.max_length]]
        return token_ids

    def batch_encode(self, texts: list[str], device: torch.device) -> dict[str, Tensor]:
        encoded = [self.encode(text) for text in texts]
        max_len = max((len(item) for item in encoded), default=1)
        max_len = max(1, min(max_len, self.max_length))

        input_ids = torch.zeros(
            (len(texts), max_len),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row, token_ids in enumerate(encoded):
            token_ids = token_ids[:max_len]
            if not token_ids:
                continue
            length = len(token_ids)
            input_ids[row, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class SmallCNNImageEncoder(nn.Module):
    """Small CNN that maps a composed puzzle image to a global feature."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.features(images)


class ByteTransformerTextEncoder(nn.Module):
    """Compact Transformer text encoder for byte-level OCR text tokens."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        num_heads: int,
        num_layers: int,
        max_length: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("text feature_dim must be divisible by num_heads")

        self.max_length = max_length
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, feature_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_length + 1, feature_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.embedding_norm = nn.LayerNorm(feature_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.pooler = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Tanh())
        self._reset_parameters()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(f"input length {seq_len} exceeds max_length={self.max_length}")

        token_embeddings = self.token_embedding(input_ids)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_token, token_embeddings], dim=1)

        positions = torch.arange(seq_len + 1, device=input_ids.device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        if attention_mask is None:
            padding_mask = input_ids.eq(self.padding_idx)
        else:
            padding_mask = ~attention_mask.bool()
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)
        padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.output_norm(hidden)
        return self.pooler(hidden[:, 0])

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)


class QHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class FenFeaturePolicy(nn.Module):
    """Use a FEN backbone as feature extractor, then predict swap-action Q values."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.q_head = QHead(feature_dim, hidden_dim, action_count, dropout)

    def forward(self, images: Tensor) -> Tensor:
        features = self.feature_extractor(images)
        features = self._as_feature_tensor(features)
        return self.q_head(features)

    @staticmethod
    def _as_feature_tensor(features: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(features, dict):
            for key in ("board_summary", "features", "embedding", "logits"):
                value = features.get(key)
                if isinstance(value, Tensor):
                    features = value
                    break
            else:
                raise KeyError(
                    "FEN feature extractor returned a dict without a supported "
                    "feature key: board_summary/features/embedding/logits"
                )

        if not isinstance(features, Tensor):
            raise TypeError(f"FEN features must be a tensor, got {type(features)!r}")

        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if features.ndim != 2:
            raise ValueError(f"FEN features must have shape [batch, dim], got {tuple(features.shape)}")
        return features


class TextFeaturePolicy(nn.Module):
    """Use OCR text only, then predict swap-action Q values."""

    def __init__(
        self,
        text_encoder: ByteTransformerTextEncoder,
        text_feature_dim: int,
        hidden_dim: int,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.q_head = QHead(text_feature_dim, hidden_dim, action_count, dropout)

    def forward(self, text_inputs: dict[str, Tensor]) -> Tensor:
        text_features = self.text_encoder(**text_inputs)
        return self.q_head(text_features)


class MultimodalFeaturePolicy(nn.Module):
    """Fuse FEN visual features and OCR text features before Q prediction."""

    def __init__(
        self,
        visual_feature_extractor: nn.Module,
        text_encoder: ByteTransformerTextEncoder,
        image_feature_dim: int,
        text_feature_dim: int,
        hidden_dim: int,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_feature_extractor = visual_feature_extractor
        self.text_encoder = text_encoder
        self.image_projection = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_feature_dim),
            nn.Linear(text_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.q_head = QHead(hidden_dim * 2, hidden_dim, action_count, dropout)

    def forward(self, images: Tensor, text_inputs: dict[str, Tensor]) -> Tensor:
        image_features = FenFeaturePolicy._as_feature_tensor(self.visual_feature_extractor(images))
        text_features = self.text_encoder(**text_inputs)
        fused_features = torch.cat(
            [self.image_projection(image_features), self.text_projection(text_features)],
            dim=-1,
        )
        return self.q_head(fused_features)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self.memory: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, obs: dict, action: tuple[int, int], reward: float, next_obs: dict, done: bool) -> None:
        self.memory.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNJigsawAgent:
    def __init__(
        self,
        device: str | torch.device = "cpu",
        solver_type: str = "visual",
        image_size: int = 288,
        image_feature_dim: int = 256,
        text_feature_dim: int = 256,
        hidden_dim: int = 256,
        text_num_heads: int = 8,
        text_num_layers: int = 2,
        text_max_length: int = 512,
        fen_hidden_size1: int = 512,
        fen_feature_hidden: int = 512,
        fen_model_name: str = "ef",
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 10000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        target_update_interval: int = 1000,
        grad_clip_norm: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        if solver_type not in {"visual", "text", "multimodal"}:
            raise ValueError("solver_type must be one of: visual, text, multimodal")
        self.solver_type = solver_type
        self.image_size = image_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.rng = random.Random(seed)

        self.tokenizer = CharTokenizer(max_length=text_max_length)
        self.policy_net = self._build_q_model(
            solver_type=solver_type,
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            hidden_dim=hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_length=text_max_length,
            fen_hidden_size1=fen_hidden_size1,
            fen_feature_hidden=fen_feature_hidden,
            fen_model_name=fen_model_name,
        ).to(self.device)
        self.target_net = self._build_q_model(
            solver_type=solver_type,
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            hidden_dim=hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_length=text_max_length,
            fen_hidden_size1=fen_hidden_size1,
            fen_feature_hidden=fen_feature_hidden,
            fen_model_name=fen_model_name,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    def select_action(self, obs: dict, training: bool = True) -> tuple[int, int]:
        epsilon = self.epsilon if training else 0.0
        legal_mask = np.asarray(obs["legal_action_mask"], dtype=bool)

        if training and self.rng.random() < epsilon:
            legal_actions = np.argwhere(legal_mask)
            selected = legal_actions[self.rng.randrange(len(legal_actions))]
            return int(selected[0]), int(selected[1])

        self.policy_net.eval()
        with torch.no_grad():
            batch = self._collate_observations([obs])
            q_values = self._forward_policy(self.policy_net, batch)
            q_values = q_values.view(1, PIECE_COUNT, PIECE_COUNT)
            q_values = self._mask_q_values(q_values, batch["legal_action_masks"])
            flat_action = int(q_values.view(-1).argmax().item())
        if training:
            self.policy_net.train()
        return divmod(flat_action, PIECE_COUNT)

    def optimize(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        obs_batch = [transition.obs for transition in transitions]
        next_obs_batch = [transition.next_obs for transition in transitions]

        batch = self._collate_observations(obs_batch)
        next_batch = self._collate_observations(next_obs_batch)
        actions = torch.tensor(
            [action[0] * PIECE_COUNT + action[1] for action in (t.action for t in transitions)],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [transition.done for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        self.policy_net.train()
        q_values = self._forward_policy(self.policy_net, batch)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._forward_policy(self.target_net, next_batch)
            next_q_values = next_q_values.view(-1, PIECE_COUNT, PIECE_COUNT)
            next_q_values = self._mask_q_values(
                next_q_values,
                next_batch["legal_action_masks"],
            )
            next_q_max = next_q_values.view(next_q_values.size(0), -1).max(dim=1).values
            target = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.target_update_interval == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.global_step / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def state_dict(self) -> dict:
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = int(state.get("global_step", 0))

    def _build_q_model(
        self,
        solver_type: str,
        image_feature_dim: int,
        text_feature_dim: int,
        hidden_dim: int,
        text_num_heads: int,
        text_num_layers: int,
        text_max_length: int,
        fen_hidden_size1: int,
        fen_feature_hidden: int,
        fen_model_name: str,
    ) -> nn.Module:
        if solver_type == "text":
            return TextFeaturePolicy(
                text_encoder=self._build_text_encoder(
                    text_feature_dim=text_feature_dim,
                    text_num_heads=text_num_heads,
                    text_num_layers=text_num_layers,
                    text_max_length=text_max_length,
                ),
                text_feature_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                action_count=ACTION_COUNT,
            )

        feature_extractor = self._build_fen_feature_extractor(
            model_name=fen_model_name,
            hidden_size1=fen_hidden_size1,
            feature_dim=image_feature_dim,
            feature_hidden=fen_feature_hidden,
        )

        if solver_type == "multimodal":
            return MultimodalFeaturePolicy(
                visual_feature_extractor=feature_extractor,
                text_encoder=self._build_text_encoder(
                    text_feature_dim=text_feature_dim,
                    text_num_heads=text_num_heads,
                    text_num_layers=text_num_layers,
                    text_max_length=text_max_length,
                ),
                image_feature_dim=image_feature_dim,
                text_feature_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                action_count=ACTION_COUNT,
            )

        return FenFeaturePolicy(
            feature_extractor=feature_extractor,
            feature_dim=image_feature_dim,
            hidden_dim=hidden_dim,
            action_count=ACTION_COUNT,
        )

    def _build_text_encoder(
        self,
        text_feature_dim: int,
        text_num_heads: int,
        text_num_layers: int,
        text_max_length: int,
    ) -> ByteTransformerTextEncoder:
        return ByteTransformerTextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            feature_dim=text_feature_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            max_length=text_max_length,
        )

    def _build_fen_feature_extractor(
        self,
        model_name: str,
        hidden_size1: int,
        feature_dim: int,
        feature_hidden: int,
    ) -> nn.Module:
        if model_name.startswith("dualstem_"):
            return dualstem_fen_model(
                hidden_size1=hidden_size1,
                hidden_size2=feature_dim,
                feature_hidden=feature_hidden,
                model_name=model_name.removeprefix("dualstem_"),
            )

        if model_name == "central":
            return central_fen_model(hidden_size1=hidden_size1, hidden_size2=feature_dim)

        if model_name == "attention":
            return attention_fen_model(
                embed_dim=feature_dim,
                single_output=True,
                project_hidden=feature_dim,
            )

        return fen_model(
            hidden_size1=hidden_size1,
            hidden_size2=feature_dim,
            feature_hidden=feature_hidden,
            model_name=model_name,
        )

    def _collate_observations(self, observations: list[dict]) -> dict:
        batch: dict[str, Tensor | dict[str, Tensor]] = {}
        if self.solver_type in {"visual", "multimodal"}:
            batch["images"] = torch.stack(
                [self._image_to_tensor(obs["image"]) for obs in observations],
                dim=0,
            ).to(self.device)

        if self.solver_type in {"text", "multimodal"}:
            texts = [self._join_texts(obs["texts"]) for obs in observations]
            batch["text_inputs"] = self.tokenizer.batch_encode(texts, device=self.device)

        legal_action_masks = torch.tensor(
            np.stack([np.asarray(obs["legal_action_mask"], dtype=bool) for obs in observations]),
            dtype=torch.bool,
            device=self.device,
        )
        batch["legal_action_masks"] = legal_action_masks
        return batch

    def _forward_policy(self, policy_net: nn.Module, batch: dict) -> Tensor:
        if self.solver_type == "visual":
            return policy_net(batch["images"])
        if self.solver_type == "text":
            return policy_net(batch["text_inputs"])
        return policy_net(batch["images"], batch["text_inputs"])

    def _image_to_tensor(self, image: Image.Image | np.ndarray | Tensor) -> Tensor:
        if isinstance(image, Tensor):
            tensor = image.detach().cpu().float()
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type: {type(image)!r}")
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor

    @staticmethod
    def _join_texts(texts: str | Iterable[str]) -> str:
        if isinstance(texts, str):
            return texts
        return "\n".join(str(text) for text in texts)

    @staticmethod
    def _mask_q_values(q_values: Tensor, legal_action_masks: Tensor) -> Tensor:
        return q_values.masked_fill(~legal_action_masks, -torch.inf)
