"""Dataset environment for fixed-count MangaZero panel ordering."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from Solver.data.panel_ordering_dataset import (
    MangaZeroPanelOrderingDataset,
    MangaZeroPanelSample,
    collate_panel_ordering_batch,
    load_panel_ordering_splits,
    pairwise_accuracy,
)


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
