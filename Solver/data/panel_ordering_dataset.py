"""Shared reader and split loader for fixed-count panel-ordering puzzles."""

from __future__ import annotations

import json
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
    # Experiment metadata is optional for backward compatibility with older
    # sample.json files.  RQ2 requires style/work_id and validates them in its
    # runner before constructing a training split.
    dataset: str = ""
    style: str = ""
    work_id: str = ""
    page_key: str = ""
    window_id: str = ""
    split: str = ""
    source_dir: str = ""


class MangaZeroPanelOrderingDataset(Dataset):
    """Read fixed-count ordering samples from a shared-format dataset directory."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] | None = None,
        use_layout: bool = True,
        allow_empty: bool = False,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = self._normalize_split(split)
        self.image_size = image_size
        self.use_layout = use_layout
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.samples = self._load_samples()
        if not self.samples and not allow_empty:
            raise ValueError(f"No samples found in {self.dataset_dir} for split {self.split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        panel_images = []
        layout_features = []
        dialog_texts = []

        for panel in sample.panels:
            image = self._load_panel_image(panel, sample.source_dir or str(self.dataset_dir))
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
            "dataset": sample.dataset,
            "style": sample.style,
            "work_id": sample.work_id,
            "page_key": sample.page_key,
            "window_id": sample.window_id,
            "split": sample.split or self.split,
        }
        if self.use_layout:
            item["layout_features"] = torch.tensor(layout_features, dtype=torch.float32)
        return item

    def _load_samples(self) -> list[MangaZeroPanelSample]:
        split_jsonl_path = self.manifest_path or (self.dataset_dir / f"{self.split}.jsonl")
        if self.split != "all" and split_jsonl_path.exists():
            return self._load_samples_from_jsonl(
                split_jsonl_path,
                self.split if self.manifest_path is not None else None,
            )

        manifest_path = self.manifest_path or (self.dataset_dir / "manifest.jsonl")
        if manifest_path.exists():
            return self._load_samples_from_manifest(manifest_path, self.split)

        if self.split != "all":
            raise FileNotFoundError(
                f"No {self.split}.jsonl or split-aware manifest.jsonl in {self.dataset_dir}. "
                "Generate train/val/test split manifests or explicitly use split='all'."
            )

        sample_paths = sorted(self.dataset_dir.glob("[0-9]*/sample.json"))
        if sample_paths:
            return [self._sample_from_json(json.loads(path.read_text(encoding="utf-8"))) for path in sample_paths]
        raise FileNotFoundError(f"No manifest.jsonl or numbered sample.json directories in {self.dataset_dir}")

    def _load_samples_from_manifest(
        self,
        manifest_path: Path,
        split: str,
    ) -> list[MangaZeroPanelSample]:
        samples = []
        entries_without_split = 0
        entries_seen = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entries_seen += 1
                if split != "all":
                    entry_split = entry.get("split")
                    if entry_split is None:
                        entries_without_split += 1
                        continue
                    if self._normalize_split(str(entry_split)) != split:
                        continue
                samples.append(self._sample_from_manifest_entry(entry))
        if split != "all" and entries_seen > 0 and entries_without_split == entries_seen:
            raise ValueError(
                f"{manifest_path} has no split metadata. Refusing to use the full dataset as "
                f"the {split!r} split; generate split manifests first or explicitly use split='all'."
            )
        if split != "all" and entries_without_split:
            raise ValueError(
                f"{manifest_path} has {entries_without_split} entries without split metadata; "
                "refusing a potentially leaking partial split."
            )
        return samples

    def _load_samples_from_jsonl(self, jsonl_path: Path, split: str | None = None) -> list[MangaZeroPanelSample]:
        samples = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if split is not None and entry.get("split") != split:
                    continue
                samples.append(self._sample_from_manifest_entry(entry))
        return samples

    def _sample_from_manifest_entry(self, entry: dict[str, Any]) -> MangaZeroPanelSample:
        if "panels" in entry:
            sample = entry
        else:
            source_dir = Path(str(entry.get("source_dir", self.dataset_dir)))
            sample_path = source_dir / str(entry["sample_path"])
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
        # Manifest metadata takes precedence over fields in sample.json.  This
        # permits one experiment manifest to annotate legacy samples without
        # rewriting the source dataset.
        merged = dict(sample)
        for key in ("dataset", "style", "work_id", "page_key", "window_id", "split"):
            if key in entry:
                merged[key] = entry[key]
        merged["source_dir"] = str(entry.get("source_dir", self.dataset_dir))
        return self._sample_from_json(merged)

    @staticmethod
    def _normalize_split(split: str) -> str:
        normalized = str(split).strip().lower()
        if normalized == "valid":
            normalized = "val"
        if normalized not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split {split!r}; expected train, val, test, or all")
        return normalized

    @staticmethod
    def _sample_from_json(raw: dict[str, Any]) -> MangaZeroPanelSample:
        return MangaZeroPanelSample(
            sequence_id=str(raw["sequence_id"]),
            manga_id=str(raw.get("manga_id", "")),
            chapter_id=str(raw.get("chapter_id", "")),
            panel_count=int(raw["panel_count"]),
            target_order=[int(value) for value in raw["target_order"]],
            panels=list(raw["panels"]),
            dataset=str(raw.get("dataset", "")),
            style=str(raw.get("style", "")),
            work_id=str(raw.get("work_id", raw.get("manga_id", ""))),
            page_key=str(raw.get("page_key", "")),
            window_id=str(raw.get("window_id", raw.get("sequence_id", ""))),
            split=str(raw.get("split", "")),
            source_dir=str(raw.get("source_dir", "")),
        )

    def _load_panel_image(self, panel: dict[str, Any], source_dir: str | Path | None = None) -> torch.Tensor:
        image_path = self._resolve_path(str(panel["padded_path"]), source_dir)
        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None and image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _resolve_path(self, raw_path: str, source_dir: str | Path | None = None) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        base_dir = Path(source_dir) if source_dir is not None else self.dataset_dir
        candidates = [
            base_dir / path,
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


def load_panel_ordering_splits(
    dataset_dir: str | Path,
    image_size: tuple[int, int] | None = None,
    use_layout: bool = True,
) -> tuple[MangaZeroPanelOrderingDataset, MangaZeroPanelOrderingDataset, MangaZeroPanelOrderingDataset]:
    """Load train, validation, and test sets recorded in the dataset manifests."""
    return tuple(
        MangaZeroPanelOrderingDataset(
            dataset_dir=dataset_dir,
            split=split,
            image_size=image_size,
            use_layout=use_layout,
            allow_empty=split != "train",
        )
        for split in ("train", "val", "test")
    )


def pairwise_accuracy(pred_order: list[int], target_order: list[int]) -> float:
    pred_rank = {panel: rank for rank, panel in enumerate(pred_order)}
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
        "dataset": [item.get("dataset", "") for item in batch],
        "style": [item.get("style", "") for item in batch],
        "work_id": [item.get("work_id", "") for item in batch],
        "page_key": [item.get("page_key", "") for item in batch],
        "window_id": [item.get("window_id", "") for item in batch],
        "split": [item.get("split", "") for item in batch],
    }
    if "layout_features" in batch[0]:
        result["layout_features"] = torch.stack([item["layout_features"] for item in batch], dim=0)
    return result
