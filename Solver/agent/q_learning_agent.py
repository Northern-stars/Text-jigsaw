"""DQN agent for the text jigsaw swap environment."""

from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from Solver.model_code.fen_model import BasicTransformerTextEncoder, FenModel


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
        image_size: int = 224,
        image_feature_dim: int = 256,
        text_feature_dim: int = 256,
        hidden_dim: int = 256,
        text_num_heads: int = 8,
        text_num_layers: int = 2,
        text_max_length: int = 512,
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
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            hidden_dim=hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_length=text_max_length,
        ).to(self.device)
        self.target_net = self._build_q_model(
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            hidden_dim=hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_length=text_max_length,
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
            q_values = self.policy_net(batch["images"], batch["text_inputs"])
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
        q_values = self.policy_net(batch["images"], batch["text_inputs"])
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_batch["images"], next_batch["text_inputs"])
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
        image_feature_dim: int,
        text_feature_dim: int,
        hidden_dim: int,
        text_num_heads: int,
        text_num_layers: int,
        text_max_length: int,
    ) -> nn.Module:
        image_encoder = SmallCNNImageEncoder(output_dim=image_feature_dim)
        text_encoder = BasicTransformerTextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=text_feature_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            dim_feedforward=text_feature_dim * 4,
            max_length=text_max_length,
        )
        return FenModel(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            num_outputs=ACTION_COUNT,
            hidden_dim=hidden_dim,
            fusion="concat",
        )

    def _collate_observations(self, observations: list[dict]) -> dict:
        images = torch.stack(
            [self._image_to_tensor(obs["image"]) for obs in observations],
            dim=0,
        ).to(self.device)
        texts = [self._join_texts(obs["texts"]) for obs in observations]
        text_inputs = self.tokenizer.batch_encode(texts, device=self.device)
        legal_action_masks = torch.tensor(
            np.stack([np.asarray(obs["legal_action_mask"], dtype=bool) for obs in observations]),
            dtype=torch.bool,
            device=self.device,
        )
        return {
            "images": images,
            "text_inputs": text_inputs,
            "legal_action_masks": legal_action_masks,
        }

    def _image_to_tensor(self, image: Image.Image | np.ndarray | Tensor) -> Tensor:
        if isinstance(image, Tensor):
            tensor = image.detach().cpu().float()
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type: {type(image)!r}")
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
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
