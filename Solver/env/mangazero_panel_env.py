"""Dataset environment for fixed-count MangaZero panel ordering."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MangaZeroPanelSample:
    sequence_id: str
    manga_id: str
    chapter_id: str
    panel_count: int
    target_order: list[int]
    panels: list[dict[str, Any]]


class MangaZeroPanelOrderingDataset(Dataset):
    """Read fixed-count panel-ordering samples produced by Data/Mangazero/main.py."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] | None = None,
        use_layout: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.use_layout = use_layout
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        panel_images = []
        layout_features = []
        dialog_texts = []

        for panel in sample.panels:
            image = self._load_panel_image(panel)
            panel_images.append(image)
            layout_features.append(self._layout_features(panel))
            dialog_texts.append(str(panel.get("dialog_text", "")))

        item: dict[str, Any] = {
            "sequence_id": sample.sequence_id,
            "manga_id": sample.manga_id,
            "chapter_id": sample.chapter_id,
            "panel_images": torch.stack(panel_images, dim=0),
            "target_order": torch.tensor(sample.target_order, dtype=torch.long),
            "dialog_texts": dialog_texts,
            "panels": sample.panels,
        }
        if self.use_layout:
            item["layout_features"] = torch.tensor(layout_features, dtype=torch.float32)
        return item

    def _load_samples(self) -> list[MangaZeroPanelSample]:
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if manifest_path.exists():
            return self._load_samples_from_manifest(manifest_path)
        sample_paths = sorted(self.dataset_dir.glob("[0-9]*/sample.json"))
        if sample_paths:
            return [self._sample_from_json(json.loads(path.read_text(encoding="utf-8"))) for path in sample_paths]
        split_jsonl_path = self.dataset_dir / f"{self.split}.jsonl"
        if split_jsonl_path.exists():
            return self._load_samples_from_jsonl(split_jsonl_path)
        raise FileNotFoundError(f"No manifest.jsonl, numbered sample.json directories, or {self.split}.jsonl in {self.dataset_dir}")

    def _load_samples_from_manifest(self, manifest_path: Path) -> list[MangaZeroPanelSample]:
        samples = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sample_path = self.dataset_dir / str(entry["sample_path"])
                samples.append(self._sample_from_json(json.loads(sample_path.read_text(encoding="utf-8"))))
        return samples

    def _load_samples_from_jsonl(self, jsonl_path: Path) -> list[MangaZeroPanelSample]:
        samples = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                samples.append(self._sample_from_json(json.loads(line)))
        return samples

    @staticmethod
    def _sample_from_json(raw: dict[str, Any]) -> MangaZeroPanelSample:
        return MangaZeroPanelSample(
            sequence_id=str(raw["sequence_id"]),
            manga_id=str(raw.get("manga_id", "")),
            chapter_id=str(raw.get("chapter_id", "")),
            panel_count=int(raw["panel_count"]),
            target_order=[int(value) for value in raw["target_order"]],
            panels=list(raw["panels"]),
        )

    def _load_panel_image(self, panel: dict[str, Any]) -> torch.Tensor:
        image_path = self._resolve_path(str(panel["padded_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None and image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        candidates = [
            self.dataset_dir / path,
            self.dataset_dir.parent / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _layout_features(panel: dict[str, Any]) -> list[float]:
        bbox = [float(value) for value in panel.get("bbox", [0, 0, 1, 1])]
        x1, y1, x2, y2 = bbox
        page_size = panel.get("page_size", [x2, y2])
        page_w = max(1.0, float(page_size[0]))
        page_h = max(1.0, float(page_size[1]))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        page_index = float(panel.get("page_index", 0))
        return [
            x1 / page_w,
            y1 / page_h,
            x2 / page_w,
            y2 / page_h,
            center_x / page_w,
            center_y / page_h,
            width / page_w,
            height / page_h,
            (width * height) / max(1.0, page_w * page_h),
            page_index / 10000.0,
        ]


class MangaZeroPanelOrderingEnv:
    """Sequential panel-ordering environment backed by MangaZeroPanelOrderingDataset.

    The environment exposes a fixed puzzle as an episode. Each action selects
    one input panel index as the next panel in the predicted order.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] | None = None,
        use_layout: bool = True,
        seed: int | None = None,
        invalid_action_penalty: float = -1.0,
    ) -> None:
        self.dataset = MangaZeroPanelOrderingDataset(
            dataset_dir=dataset_dir,
            split=split,
            image_size=image_size,
            use_layout=use_layout,
        )
        self.rng = random.Random(seed)
        self.invalid_action_penalty = invalid_action_penalty
        self.current_sample: dict[str, Any] | None = None
        self.selected_order: list[int] = []
        self.done = False

    def reset(self, index: int | None = None) -> dict[str, Any]:
        if index is None:
            index = self.rng.randrange(len(self.dataset))
        self.current_sample = self.dataset[index]
        self.selected_order = []
        self.done = False
        return self._observation()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.current_sample is None:
            raise RuntimeError("reset() must be called before step()")
        if self.done:
            return self._observation(), 0.0, True, self._episode_info()

        panel_count = int(self.current_sample["target_order"].numel())
        action = int(action)
        invalid = action < 0 or action >= panel_count or action in self.selected_order
        if invalid:
            self.done = True
            info = self._episode_info(invalid_action=action)
            return self._observation(), self.invalid_action_penalty, True, info

        self.selected_order.append(action)
        self.done = len(self.selected_order) == panel_count
        if not self.done:
            return self._observation(), 0.0, False, self._episode_info()

        info = self._episode_info()
        reward = info["pairwise_accuracy"]
        if info["exact_match"] == 1.0:
            reward += 1.0
        return self._observation(), float(reward), True, info

    def _observation(self) -> dict[str, Any]:
        if self.current_sample is None:
            raise RuntimeError("reset() must be called before requesting an observation")
        panel_count = int(self.current_sample["target_order"].numel())
        selected_mask = torch.zeros(panel_count, dtype=torch.bool)
        if self.selected_order:
            selected_mask[torch.tensor(self.selected_order, dtype=torch.long)] = True
        observation = {
            "sequence_id": self.current_sample["sequence_id"],
            "panel_images": self.current_sample["panel_images"],
            "dialog_texts": self.current_sample["dialog_texts"],
            "selected_order": torch.tensor(self.selected_order, dtype=torch.long),
            "selected_mask": selected_mask,
            "done": self.done,
        }
        if "layout_features" in self.current_sample:
            observation["layout_features"] = self.current_sample["layout_features"]
        return observation

    def _episode_info(self, invalid_action: int | None = None) -> dict[str, Any]:
        if self.current_sample is None:
            return {}
        target = self.current_sample["target_order"].tolist()
        pred = self.selected_order.copy()
        complete = len(pred) == len(target) and invalid_action is None
        info = {
            "target_order": target,
            "pred_order": pred,
            "invalid_action": invalid_action,
            "complete": complete,
            "exact_match": 0.0,
            "position_accuracy": 0.0,
            "pairwise_accuracy": 0.0,
        }
        if pred:
            compare_len = min(len(pred), len(target))
            info["position_accuracy"] = sum(
                int(pred[i] == target[i]) for i in range(compare_len)
            ) / len(target)
        if complete:
            info["exact_match"] = float(pred == target)
            info["pairwise_accuracy"] = pairwise_accuracy(pred, target)
        return info


def pairwise_accuracy(pred_order: list[int], target_order: list[int]) -> float:
    pred_rank = {panel: rank for rank, panel in enumerate(pred_order)}
    target_rank = {panel: rank for rank, panel in enumerate(target_order)}
    correct = 0
    total = 0
    for i, panel_i in enumerate(target_order):
        for panel_j in target_order[i + 1:]:
            if panel_i not in pred_rank or panel_j not in pred_rank:
                continue
            total += 1
            if pred_rank[panel_i] < pred_rank[panel_j]:
                correct += 1
    return correct / total if total else 0.0


def collate_panel_ordering_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sequence_id": [item["sequence_id"] for item in batch],
        "manga_id": [item["manga_id"] for item in batch],
        "chapter_id": [item["chapter_id"] for item in batch],
        "panel_images": torch.stack([item["panel_images"] for item in batch], dim=0),
        "target_order": torch.stack([item["target_order"] for item in batch], dim=0),
        "dialog_texts": [item["dialog_texts"] for item in batch],
        "panels": [item["panels"] for item in batch],
    }
    if "layout_features" in batch[0]:
        result["layout_features"] = torch.stack([item["layout_features"] for item in batch], dim=0)
    return result
