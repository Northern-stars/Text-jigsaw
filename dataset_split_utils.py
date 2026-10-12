"""Shared no-leak split scheduling for ordering puzzle generation.

Each source page (manga page / comix page) is assigned to exactly one split.
Puzzles are generated from a page only while that split still has quota;
when the quota is consumed, the next page belongs to the next split. This
guarantees train/val/test never share a page (or a window made from it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPLIT_ORDER = ("train", "val", "test")


class SplitBudget:
    """Track per-split puzzle quotas and the active split for the next page."""

    def __init__(self, train: int = 0, val: int = 0, test: int = 0) -> None:
        self.quotas = {"train": int(train), "val": int(val), "test": int(test)}
        self.counts = {"train": 0, "val": 0, "test": 0}
        self.current: str | None = self._first_nonzero()

    def _first_nonzero(self) -> str | None:
        for name in SPLIT_ORDER:
            if self.quotas[name] > 0:
                return name
        return None

    @property
    def active_split(self) -> str | None:
        return self.current

    @property
    def remaining(self) -> int:
        if self.current is None:
            return 0
        return self.quotas[self.current] - self.counts[self.current]

    @property
    def done(self) -> bool:
        return self.current is None

    def add(self, count: int) -> None:
        if self.current is None:
            return
        self.counts[self.current] += int(count)
        if self.remaining <= 0:
            self._advance()

    def _advance(self) -> None:
        if self.current is None:
            return
        start = SPLIT_ORDER.index(self.current)
        for name in SPLIT_ORDER[start + 1:]:
            if self.quotas[name] > self.counts[name]:
                self.current = name
                return
        self.current = None

    @property
    def totals(self) -> dict[str, int]:
        return dict(self.counts)


class SplitManifests:
    """Write per-split JSONL files plus a combined manifest.jsonl.

    Entries are compatible with MangaZeroPanelOrderingDataset:
    each line includes sample_path and puzzle_index.
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {name: output_dir / f"{name}.jsonl" for name in SPLIT_ORDER}
        self.handles = {
            name: path.open("w", encoding="utf-8")
            for name, path in self.paths.items()
        }
        self.combined = (output_dir / "manifest.jsonl").open("w", encoding="utf-8")

    def write(self, split: str, entry: dict[str, Any]) -> None:
        if split not in self.handles:
            raise ValueError(f"unknown split: {split}")
        record = dict(entry)
        record["split"] = split
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.handles[split].write(line)
        self.combined.write(line)

    def flush(self) -> None:
        for handle in self.handles.values():
            handle.flush()
        self.combined.flush()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.combined.close()


def resolve_split_counts(
    args: Any, default_puzzle_num: int
) -> tuple[int, int, int]:
    """Resolve explicit train/val/test counts, falling back to puzzle_num as train."""
    train = int(getattr(args, "train_puzzles", 0) or 0)
    val = int(getattr(args, "val_puzzles", 0) or 0)
    test = int(getattr(args, "test_puzzles", 0) or 0)
    if train <= 0 and val <= 0 and test <= 0:
        train = int(getattr(args, "puzzle_num", default_puzzle_num) or 0)
    if train < 0 or val < 0 or test < 0:
        raise ValueError("split puzzle counts must be non-negative")
    return train, val, test