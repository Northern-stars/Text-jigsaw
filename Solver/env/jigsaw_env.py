"""Reinforcement-learning environment for 3x3 text jigsaw puzzles."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image


GRID_SIZE = 3
PIECE_COUNT = GRID_SIZE * GRID_SIZE
SOLVED_ORDER = list(range(PIECE_COUNT))
HORIZONTAL_PAIRS = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8))
VERTICAL_PAIRS = ((0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8))


@dataclass(frozen=True)
class Piece:
    piece_id: int
    row: int
    col: int
    text: str
    segments: list[str]
    image_path: Path
    image: Image.Image


class TextJigsawEnv:
    """3x3 text jigsaw environment with swap actions.

    Observation format:
        {
            "image": PIL image or transformed image,
            "texts": list[str],
            "order": list[int],
            "legal_action_mask": np.ndarray bool shape [9, 9],
        }
    """

    def __init__(
        self,
        dataset_dir: str,
        pairwise_weight: float = 1.0,
        category_weight: float = 1.0,
        done_reward: float = 10.0,
        max_steps: int = 50,
        reconstruct_text_by_line: bool = False,
        image_transform: Optional[Callable[[Image.Image], object]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.pairwise_weight = pairwise_weight
        self.category_weight = category_weight
        self.done_reward_value = done_reward
        self.max_steps = max_steps
        self.reconstruct_text_by_line = reconstruct_text_by_line
        self.image_transform = image_transform
        self.rng = random.Random(seed)

        self.sample_paths = sorted(self.dataset_dir.rglob("*.json"))
        if not self.sample_paths:
            raise ValueError(f"No .json files found under {self.dataset_dir}")

        self.pieces: list[Piece] = []
        self.current_order: list[int] = SOLVED_ORDER.copy()
        self.step_count = 0
        self.sample_path: Optional[Path] = None
        self.legal_action_mask = self._build_legal_action_mask()

    def reset(self) -> dict:
        self.sample_path = self.rng.choice(self.sample_paths)
        self.pieces = self._load_sample(self.sample_path)
        self.current_order = self._make_initial_order()
        self.step_count = 0
        return self._get_observation()

    def step(self, action: tuple[int, int]) -> tuple[dict, float, bool, bool, dict]:
        index1, index2 = action
        self._validate_action(index1, index2)

        self.current_order[index1], self.current_order[index2] = (
            self.current_order[index2],
            self.current_order[index1],
        )
        self.step_count += 1

        reward, info = self._compute_reward()
        done = self.current_order == SOLVED_ORDER
        truncated = self.step_count >= self.max_steps and not done
        obs = self._get_observation()

        info.update(
            {
                "order": self.current_order.copy(),
                "sample_path": str(self.sample_path) if self.sample_path else "",
                "step_count": self.step_count,
            }
        )
        return obs, reward, done, truncated, info

    def render(self) -> Image.Image:
        return self._build_current_image()

    def _load_sample(self, json_path: Path) -> list[Piece]:
        with json_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if labels.get("grid_size") != GRID_SIZE:
            raise ValueError(f"{json_path} has grid_size={labels.get('grid_size')}")

        raw_pieces = labels.get("pieces", [])
        if len(raw_pieces) != PIECE_COUNT:
            raise ValueError(f"{json_path} must contain {PIECE_COUNT} pieces")

        by_id = {}
        sample_dir = json_path.parent
        base_size: Optional[tuple[int, int]] = None

        for raw_piece in raw_pieces:
            piece_id = int(raw_piece["piece_id"])
            image_path = sample_dir / raw_piece["image"]
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))

            image = Image.open(image_path).convert("RGB")
            if base_size is None:
                base_size = image.size
            elif image.size != base_size:
                image = image.resize(base_size, Image.BILINEAR)

            segments = raw_piece.get("segments")
            if not isinstance(segments, list):
                segments = self._fallback_segments(raw_piece.get("text", ""))

            by_id[piece_id] = Piece(
                piece_id=piece_id,
                row=int(raw_piece.get("row", piece_id // GRID_SIZE)),
                col=int(raw_piece.get("col", piece_id % GRID_SIZE)),
                text=str(raw_piece.get("text", "")),
                segments=[str(segment) for segment in segments],
                image_path=image_path,
                image=image,
            )

        expected_ids = set(SOLVED_ORDER)
        if set(by_id) != expected_ids:
            raise ValueError(f"{json_path} piece ids must be 0..8")

        return [by_id[piece_id] for piece_id in SOLVED_ORDER]

    def _fallback_segments(self, text: str) -> list[str]:
        if not text:
            return []
        segments = [segment for segment in text.split("<SEG>") if segment]
        return segments if segments else [text]

    def _make_initial_order(self) -> list[int]:
        order = SOLVED_ORDER.copy()
        for _ in range(100):
            self.rng.shuffle(order)
            if order != SOLVED_ORDER:
                return order.copy()

        order = SOLVED_ORDER.copy()
        order[0], order[1] = order[1], order[0]
        return order

    def _get_observation(self) -> dict:
        image = self._build_current_image()
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "texts": self._build_current_texts(),
            "order": self.current_order.copy(),
            "legal_action_mask": self.legal_action_mask.copy(),
        }

    def _build_current_image(self) -> Image.Image:
        if not self.pieces:
            raise RuntimeError("reset() must be called before building an image")

        piece_w, piece_h = self.pieces[0].image.size
        canvas = Image.new("RGB", (piece_w * GRID_SIZE, piece_h * GRID_SIZE), "white")

        for position, piece_id in enumerate(self.current_order):
            row = position // GRID_SIZE
            col = position % GRID_SIZE
            canvas.paste(self.pieces[piece_id].image, (col * piece_w, row * piece_h))

        return canvas

    def _build_current_texts(self) -> list[str]:
        if self.reconstruct_text_by_line:
            return self._build_reconstructed_line_texts()
        return self._build_piece_texts()

    def _build_piece_texts(self) -> list[str]:
        texts = []
        for position, piece_id in enumerate(self.current_order):
            piece = self.pieces[piece_id]
            piece_text = " ".join(segment.rstrip("\n") for segment in piece.segments)
            if not piece_text:
                piece_text = piece.text
            texts.append(f"<P{position}> {piece_text}")
        return texts

    def _build_reconstructed_line_texts(self) -> list[str]:
        texts = []
        for board_row in range(GRID_SIZE):
            row_positions = [board_row * GRID_SIZE + col for col in range(GRID_SIZE)]
            row_pieces = [self.pieces[self.current_order[pos]] for pos in row_positions]
            max_line_count = max((len(piece.segments) for piece in row_pieces), default=0)

            for line_id in range(max_line_count):
                line_parts = []
                for piece in row_pieces:
                    if line_id < len(piece.segments):
                        text = piece.segments[line_id].rstrip("\n")
                        if text:
                            line_parts.append(text)
                if line_parts:
                    texts.append(f"<R{board_row}-L{line_id}> {' '.join(line_parts)}")

        return texts

    def _compute_reward(self) -> tuple[float, dict]:
        correct_pairs = self._count_correct_pairs()
        correct_positions = sum(
            piece_id == position for position, piece_id in enumerate(self.current_order)
        )
        pairwise_reward = correct_pairs / 12.0
        category_reward = correct_positions / float(PIECE_COUNT)
        solved = self.current_order == SOLVED_ORDER
        done_reward = self.done_reward_value if solved else 0.0
        reward = (
            pairwise_reward * self.pairwise_weight
            + category_reward * self.category_weight
            + done_reward
        )

        return reward, {
            "pairwise_reward": pairwise_reward,
            "category_reward": category_reward,
            "done_reward": done_reward,
            "correct_pairs": correct_pairs,
            "correct_positions": correct_positions,
        }

    def _count_correct_pairs(self) -> int:
        correct_pairs = 0

        for left_pos, right_pos in HORIZONTAL_PAIRS:
            left_piece = self.current_order[left_pos]
            right_piece = self.current_order[right_pos]
            if (
                right_piece == left_piece + 1
                and left_piece // GRID_SIZE == right_piece // GRID_SIZE
            ):
                correct_pairs += 1

        for top_pos, bottom_pos in VERTICAL_PAIRS:
            top_piece = self.current_order[top_pos]
            bottom_piece = self.current_order[bottom_pos]
            if (
                bottom_piece == top_piece + GRID_SIZE
                and top_piece % GRID_SIZE == bottom_piece % GRID_SIZE
            ):
                correct_pairs += 1

        return correct_pairs

    def _validate_action(self, index1: int, index2: int) -> None:
        if not 0 <= index1 < PIECE_COUNT:
            raise ValueError(f"index1 out of range: {index1}")
        if not 0 <= index2 < PIECE_COUNT:
            raise ValueError(f"index2 out of range: {index2}")
        if index1 == index2:
            raise ValueError("index1 and index2 must be different")

    @staticmethod
    def _build_legal_action_mask() -> np.ndarray:
        mask = np.ones((PIECE_COUNT, PIECE_COUNT), dtype=bool)
        np.fill_diagonal(mask, False)
        return mask
