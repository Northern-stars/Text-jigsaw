"""Reinforcement-learning environment for 3x3 text jigsaw puzzles.

The environment can read both the original ``Data/PuzzleData`` JSON samples and
the Newspaper Navigator OCR jigsaw dataset produced by
``Data/newspaper-navigator/create_jigsaw_ocr_dataset.py``.
"""

from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image


GRID_SIZE = 3
PIECE_COUNT = GRID_SIZE * GRID_SIZE
CENTER_INDEX = PIECE_COUNT // 2
MOVABLE_POSITIONS = tuple(position for position in range(PIECE_COUNT) if position != CENTER_INDEX)
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
    chars: list[dict[str, object]]


class TextJigsawEnv:
    """3x3 text jigsaw environment with a fixed center piece and swap actions.

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

        self.sample_paths = self._discover_sample_paths(self.dataset_dir)
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

    def _discover_sample_paths(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            if dataset_path.name == "manifest.json":
                return self._sample_paths_from_manifest(dataset_path)
            return [dataset_path]

        manifest_path = dataset_path / "manifest.json"
        if manifest_path.exists():
            manifest_samples = self._sample_paths_from_manifest(manifest_path)
            if manifest_samples:
                return manifest_samples

        label_paths = sorted(dataset_path.rglob("label.json"))
        if label_paths:
            return label_paths

        return sorted(
            path
            for path in dataset_path.rglob("*.json")
            if path.name != "manifest.json"
        )

    def _sample_paths_from_manifest(self, manifest_path: Path) -> list[Path]:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        paths = []
        for puzzle in manifest.get("puzzles", []):
            label_path = puzzle.get("label_path")
            if not label_path:
                continue
            paths.append(self._resolve_path(str(label_path), manifest_path, must_exist=False))

        return [path for path in paths if path.exists()]

    def _load_sample(self, json_path: Path) -> list[Piece]:
        with json_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        rows, cols = self._read_grid_shape(labels, json_path)
        if rows != GRID_SIZE or cols != GRID_SIZE:
            raise ValueError(
                f"{json_path} has grid shape {rows}x{cols}; "
                f"this agent environment expects {GRID_SIZE}x{GRID_SIZE}"
            )

        raw_pieces = labels.get("pieces", [])
        if len(raw_pieces) != PIECE_COUNT:
            raise ValueError(f"{json_path} must contain {PIECE_COUNT} pieces")

        by_id = {}
        base_size: Optional[tuple[int, int]] = None

        for raw_piece in raw_pieces:
            piece_id = self._piece_index(raw_piece, rows, cols)
            row = int(raw_piece.get("row", piece_id // cols))
            col = int(raw_piece.get("col", piece_id % cols))
            image_path = self._resolve_piece_image_path(raw_piece, json_path)

            image = Image.open(image_path).convert("RGB")
            if base_size is None:
                base_size = image.size
            elif image.size != base_size:
                image = image.resize(base_size, Image.BILINEAR)

            segments = raw_piece.get("segments")
            if not isinstance(segments, list):
                chars = raw_piece.get("chars", [])
                if self.reconstruct_text_by_line and isinstance(chars, list):
                    segments = self._segments_from_chars(chars, str(raw_piece.get("text", "")))
                else:
                    segments = self._fallback_segments(str(raw_piece.get("text", "")))
            chars = raw_piece.get("chars", [])
            if not isinstance(chars, list):
                chars = []

            by_id[piece_id] = Piece(
                piece_id=piece_id,
                row=row,
                col=col,
                text=str(raw_piece.get("text", "")),
                segments=[str(segment) for segment in segments],
                image_path=image_path,
                image=image,
                chars=[dict(char) for char in chars if isinstance(char, dict)],
            )

        expected_ids = set(SOLVED_ORDER)
        if set(by_id) != expected_ids:
            raise ValueError(f"{json_path} piece ids must be 0..8")

        return [by_id[piece_id] for piece_id in SOLVED_ORDER]

    def _read_grid_shape(self, labels: dict, json_path: Path) -> tuple[int, int]:
        meta = labels.get("meta", {})
        if isinstance(meta, dict) and "rows" in meta and "cols" in meta:
            return int(meta["rows"]), int(meta["cols"])

        grid_size = labels.get("grid_size")
        if grid_size is not None:
            grid_size = int(grid_size)
            return grid_size, grid_size

        raise ValueError(f"{json_path} does not declare a supported grid shape")

    def _piece_index(self, raw_piece: dict, rows: int, cols: int) -> int:
        raw_piece_id = raw_piece.get("piece_id")
        if isinstance(raw_piece_id, int):
            return raw_piece_id

        if isinstance(raw_piece_id, str):
            if raw_piece_id.isdigit():
                return int(raw_piece_id)

            match = re.fullmatch(r"r(\d+)_c(\d+)", raw_piece_id)
            if match is not None:
                row = int(match.group(1))
                col = int(match.group(2))
                return row * cols + col

        row = int(raw_piece["row"])
        col = int(raw_piece["col"])
        return row * cols + col

    def _resolve_piece_image_path(self, raw_piece: dict, json_path: Path) -> Path:
        raw_path = raw_piece.get("piece_path", raw_piece.get("image"))
        if not raw_path:
            raise ValueError(f"{json_path} contains a piece without piece_path/image")
        return self._resolve_path(str(raw_path), json_path, must_exist=True)

    def _resolve_path(self, raw_path: str, json_path: Path, must_exist: bool) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            if must_exist and not path.exists():
                raise FileNotFoundError(str(path))
            return path

        bases: list[Path] = [json_path.parent]
        if self.dataset_dir.is_dir():
            bases.append(self.dataset_dir)
        else:
            bases.append(self.dataset_dir.parent)
        bases.extend(json_path.parents)
        bases.append(Path.cwd())

        seen = set()
        for base in bases:
            try:
                candidate = (base / path).resolve()
            except OSError:
                candidate = base / path
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate

        fallback = (json_path.parent / path).resolve()
        if must_exist:
            raise FileNotFoundError(str(fallback))
        return fallback

    def _fallback_segments(self, text: str) -> list[str]:
        if not text:
            return []
        segments = [segment for segment in text.split("<SEG>") if segment]
        return segments if segments else [text]

    def _segments_from_chars(self, chars: list[dict[str, object]], fallback_text: str) -> list[str]:
        char_items = [char for char in chars if self._char_bbox(char) is not None]
        if not char_items:
            return self._fallback_segments(fallback_text)

        lines: list[list[dict[str, object]]] = []
        line_centers: list[float] = []

        for char in sorted(char_items, key=lambda item: (self._char_center_y(item), self._char_x1(item))):
            center_y = self._char_center_y(char)
            height = self._char_height(char)
            threshold = max(4.0, height * 0.75)

            line_index = None
            for index, line_center in enumerate(line_centers):
                if abs(center_y - line_center) <= threshold:
                    line_index = index
                    break

            if line_index is None:
                lines.append([char])
                line_centers.append(center_y)
            else:
                lines[line_index].append(char)
                line = lines[line_index]
                line_centers[line_index] = sum(self._char_center_y(item) for item in line) / len(line)

        segments = []
        for line in sorted(lines, key=lambda item: min(self._char_center_y(char) for char in item)):
            text = self._text_from_chars(sorted(line, key=self._char_x1))
            if text:
                segments.append(text)

        return segments or self._fallback_segments(fallback_text)

    def _text_from_chars(self, chars: list[dict[str, object]]) -> str:
        words: list[str] = []
        current_word_id: object | None = None
        current_chars: list[str] = []

        for char_info in chars:
            word_id = char_info.get("word_id")
            if current_chars and word_id != current_word_id:
                words.append("".join(current_chars))
                current_chars = []
            current_word_id = word_id
            current_chars.append(str(char_info.get("char", "")))

        if current_chars:
            words.append("".join(current_chars))

        return " ".join(word for word in words if word)

    @staticmethod
    def _char_bbox(char: dict[str, object]) -> Optional[list[float]]:
        bbox = char.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            return [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None

    def _char_x1(self, char: dict[str, object]) -> float:
        bbox = self._char_bbox(char)
        return bbox[0] if bbox is not None else 0.0

    def _char_center_y(self, char: dict[str, object]) -> float:
        bbox = self._char_bbox(char)
        return ((bbox[1] + bbox[3]) / 2.0) if bbox is not None else 0.0

    def _char_height(self, char: dict[str, object]) -> float:
        bbox = self._char_bbox(char)
        return (bbox[3] - bbox[1]) if bbox is not None else 0.0

    def _make_initial_order(self) -> list[int]:
        order = SOLVED_ORDER.copy()
        movable_piece_ids = list(MOVABLE_POSITIONS)
        for _ in range(100):
            self.rng.shuffle(movable_piece_ids)
            for position, piece_id in zip(MOVABLE_POSITIONS, movable_piece_ids):
                order[position] = piece_id
            order[CENTER_INDEX] = CENTER_INDEX
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
            self.current_order[position] == position for position in MOVABLE_POSITIONS
        )
        pairwise_reward = correct_pairs / 12.0
        category_reward = correct_positions / float(len(MOVABLE_POSITIONS))
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
        if index1 == CENTER_INDEX or index2 == CENTER_INDEX:
            raise ValueError(f"center position {CENTER_INDEX} is fixed and cannot be swapped")

    @staticmethod
    def _build_legal_action_mask() -> np.ndarray:
        mask = np.ones((PIECE_COUNT, PIECE_COUNT), dtype=bool)
        np.fill_diagonal(mask, False)
        mask[CENTER_INDEX, :] = False
        mask[:, CENTER_INDEX] = False
        return mask
