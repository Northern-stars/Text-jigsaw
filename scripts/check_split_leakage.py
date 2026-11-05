"""Detect train/val/test overlap in panel-ordering puzzle datasets.

The check reads ``sample.json`` annotations only; puzzle images are never
opened. Three overlap levels are reported for every split pair:

- ``page``: the same source page appears in both splits. This is the hard
  leakage criterion: a page is used to build several sliding windows, so
  sharing a page means the model has seen part of the test content.
- ``panel``: the same panel (source page + panel index + bbox) appears in
  both splits.
- ``puzzle``: the exact same set of panels appears in both splits.

Input layouts:

- A dataset directory with ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``.
- A dataset directory with a combined ``manifest.jsonl`` whose lines carry a
  ``split`` field.
- Explicit split files via ``--train`` / ``--val`` / ``--test``.

Each JSONL line may either reference a ``sample_path`` or contain the full
sample (the inline layout used by ``MangaZeroPanelOrderingDataset``).

Examples:

    python scripts/check_split_leakage.py Data/Manga109/ordering_output
    python scripts/check_split_leakage.py Data/comix/6panel-ordering --json
    python scripts/check_split_leakage.py --train a/train.jsonl --test a/test.jsonl

Exit code is 1 when cross-split page or panel overlap is found, 0 otherwise
(use ``--no-strict`` for a report-only run).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SPLIT_ORDER = ("train", "val", "test")


@dataclass
class SampleRecord:
    split: str
    sequence_id: str
    source: str
    page_key: str
    panel_keys: tuple[tuple[Any, ...], ...]

    @property
    def puzzle_key(self) -> tuple[str, tuple[tuple[Any, ...], ...]]:
        return self.page_key, tuple(sorted(self.panel_keys))


@dataclass
class SplitStats:
    samples: int = 0
    pages: set[str] = field(default_factory=set)
    panels: set[tuple[Any, ...]] = field(default_factory=set)
    puzzles: set[tuple[str, tuple[tuple[Any, ...], ...]]] = field(default_factory=set)
    # Reusing one page for several windows inside a split is expected; the
    # counters only flag exact duplicate windows and repeated pages.
    repeated_page_samples: int = 0
    duplicate_window_samples: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect page/panel/puzzle overlap between ordering data splits."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        help="Dataset directory containing train/val/test.jsonl or manifest.jsonl.",
    )
    for split in SPLIT_ORDER:
        parser.add_argument(
            f"--{split}",
            type=Path,
            default=None,
            help=f"Explicit {split}.jsonl (overrides discovery in dataset_dir).",
        )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON report.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum conflicting keys to print per pair (default: 10).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Always exit 0, even when overlap is found.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            entries.append(entry)
    return entries


def _resolve_sample_path(entry: dict[str, Any], jsonl_dir: Path) -> Path:
    raw_path = entry.get("sample_path") or entry.get("path")
    if not raw_path:
        raise ValueError("manifest entry has neither sample_path nor inline panels")
    sample_path = Path(str(raw_path))
    candidates = [sample_path] if sample_path.is_absolute() else [
        jsonl_dir / sample_path,
        Path.cwd() / sample_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"sample_path {raw_path!r} not found (looked in: "
        + ", ".join(str(c) for c in candidates)
        + ")"
    )


def load_split_samples(
    split: str,
    entries: Iterable[dict[str, Any]],
    jsonl_dir: Path,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for entry in entries:
        if isinstance(entry.get("panels"), list):
            sample = entry
            source = f"{split}.jsonl:inline"
        else:
            sample_path = _resolve_sample_path(entry, jsonl_dir)
            try:
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sample_path}: invalid JSON: {exc}") from exc
            source = str(sample_path)
        if not isinstance(sample, dict) or not isinstance(sample.get("panels"), list):
            raise ValueError(f"{source}: sample has no panels list")
        records.append(_build_record(split, sample, source))
    return records


def _build_record(split: str, sample: dict[str, Any], source: str) -> SampleRecord:
    panels = [panel for panel in sample["panels"] if isinstance(panel, dict)]
    if not panels:
        raise ValueError(f"{source}: sample has an empty panels list")
    page_key = _page_key(sample, panels)
    panel_keys = tuple(_panel_key(page_key, panel) for panel in panels)
    return SampleRecord(
        split=split,
        sequence_id=str(sample.get("sequence_id", source)),
        source=source,
        page_key=page_key,
        panel_keys=panel_keys,
    )


def _page_key(sample: dict[str, Any], panels: list[dict[str, Any]]) -> str:
    page_key = sample.get("page_key")
    if page_key:
        return str(page_key)

    group_key = sample.get("group_key")
    page_id = sample.get("page_id")
    if group_key and page_id:
        group_text = str(group_key)
        page_text = str(page_id)
        # Manga109 already stores the full page path in group_key.
        if group_text == page_text or group_text.endswith("/" + page_text):
            return group_text
        return f"{group_text}/{page_text}"

    manga_id = sample.get("manga_id")
    if manga_id or page_id:
        chapter_id = sample.get("chapter_id", "default")
        return f"{manga_id or 'unknown'}/{chapter_id or 'default'}/{page_id or 'unknown'}"

    # COMICS legacy layout: comic_no / page_no instead of manga_id / page_id.
    if sample.get("comic_no") is not None:
        return f"comics/{sample.get('comic_no')}/{sample.get('page_no')}"

    # Last resort: the source image path is a stable page identity.
    panel = panels[0]
    source_path = panel.get("source_image_path") or panel.get("source_path")
    if source_path:
        return str(source_path)
    raise ValueError(f"cannot derive a page key for sample {sample.get('sequence_id')!r}")


def _panel_key(page_key: str, panel: dict[str, Any]) -> tuple[Any, ...]:
    panel_index = panel.get("panel_index_in_page")
    if panel_index is None:
        panel_index = panel.get("panel_no")
    source_path = panel.get("source_image_path") or panel.get("source_path")
    bbox = panel.get("bbox")
    bbox_key: tuple[Any, ...] = ()
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        bbox_key = tuple(_number(value) for value in bbox[:4])
    return (page_key, str(source_path or ""), panel_index, bbox_key)


def _number(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else round(number, 3)


def discover_splits(dataset_dir: Path) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    sources: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for split in SPLIT_ORDER:
        split_path = dataset_dir / f"{split}.jsonl"
        if split_path.exists():
            sources[split] = (split_path, _read_jsonl(split_path))
    if sources:
        return sources

    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No train/val/test.jsonl or manifest.jsonl in {dataset_dir}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in _read_jsonl(manifest_path):
        split = entry.get("split")
        if split:
            grouped[str(split)].append(entry)
    if not grouped:
        raise ValueError(
            f"{manifest_path} has no split field; use --train/--val/--test instead"
        )
    return {
        split: (manifest_path, entries)
        for split, entries in grouped.items()
        if split in SPLIT_ORDER
    }


def build_stats(records: list[SampleRecord]) -> SplitStats:
    stats = SplitStats(samples=len(records))
    seen_pages: set[str] = set()
    seen_puzzles: set[tuple[str, tuple[tuple[Any, ...], ...]]] = set()
    for record in records:
        if record.page_key in seen_pages:
            stats.repeated_page_samples += 1
        seen_pages.add(record.page_key)
        puzzle_key = record.puzzle_key
        if puzzle_key in seen_puzzles:
            stats.duplicate_window_samples += 1
        seen_puzzles.add(puzzle_key)
        stats.pages.add(record.page_key)
        stats.panels.update(record.panel_keys)
        stats.puzzles.add(puzzle_key)
    return stats


def _index(records: list[SampleRecord], key_name: str) -> dict[Any, list[SampleRecord]]:
    index: dict[Any, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if key_name == "page":
            index[record.page_key].append(record)
        elif key_name == "puzzle":
            index[record.puzzle_key].append(record)
        else:
            for panel_key in record.panel_keys:
                index[panel_key].append(record)
    return index


def compare_pair(
    left: str,
    right: str,
    left_records: list[SampleRecord],
    right_records: list[SampleRecord],
    max_examples: int = 10,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_name in ("page", "panel", "puzzle"):
        left_index = _index(left_records, key_name)
        right_index = _index(right_records, key_name)
        overlap = set(left_index) & set(right_index)
        result[key_name] = {
            "overlap": len(overlap),
            "left_only": len(set(left_index) - set(right_index)),
            "right_only": len(set(right_index) - set(left_index)),
            "examples": [
                {
                    "key": _jsonable(key),
                    "left": sorted({r.sequence_id for r in left_index[key]})[:3],
                    "right": sorted({r.sequence_id for r in right_index[key]})[:3],
                }
                for key in sorted(overlap, key=str)[:max_examples]
            ],
        }
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def main() -> None:
    args = parse_args()
    explicit = {
        split: getattr(args, split)
        for split in SPLIT_ORDER
        if getattr(args, split) is not None
    }
    if explicit:
        split_sources = {
            split: (path, _read_jsonl(path)) for split, path in explicit.items()
        }
    else:
        if args.dataset_dir is None:
            raise SystemExit("provide DATASET_DIR or at least one of --train/--val/--test")
        dataset_dir = args.dataset_dir.resolve()
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {dataset_dir}")
        split_sources = discover_splits(dataset_dir)

    if len(split_sources) < 2:
        raise SystemExit(
            f"need at least two splits to compare, found: {sorted(split_sources)}"
        )

    records_by_split: dict[str, list[SampleRecord]] = {}
    for split, (path, entries) in split_sources.items():
        records_by_split[split] = load_split_samples(split, entries, path.parent)

    stats = {split: build_stats(records) for split, records in records_by_split.items()}
    pairs: dict[str, Any] = {}
    leaks = 0
    names = [name for name in SPLIT_ORDER if name in records_by_split]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pair_report = compare_pair(
                left,
                right,
                records_by_split[left],
                records_by_split[right],
                max_examples=args.max_examples,
            )
            pairs[f"{left}__{right}"] = pair_report
            leaks += (
                pair_report["page"]["overlap"]
                + pair_report["panel"]["overlap"]
                + pair_report["puzzle"]["overlap"]
            )

    report = {
        "splits": {
            split: {
                "samples": value.samples,
                "pages": len(value.pages),
                "panels": len(value.panels),
                "puzzles": len(value.puzzles),
                "repeated_page_samples": value.repeated_page_samples,
                "duplicate_window_samples": value.duplicate_window_samples,
            }
            for split, value in stats.items()
        },
        "pairs": pairs,
        "leak": leaks > 0,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"splits: {', '.join(names)}")
        for split in names:
            value = stats[split]
            extra = ""
            if value.repeated_page_samples or value.duplicate_window_samples:
                extra = (
                    f" repeated_page_samples={value.repeated_page_samples}"
                    f" duplicate_window_samples={value.duplicate_window_samples}"
                )
            print(
                f"  {split}: samples={value.samples} pages={len(value.pages)} "
                f"panels={len(value.panels)} puzzles={len(value.puzzles)}{extra}"
            )
        for pair_name, pair_report in pairs.items():
            left, right = pair_name.split("__", 1)
            print(f"{left} vs {right}:")
            for key_name in ("page", "panel", "puzzle"):
                report_item = pair_report[key_name]
                print(
                    f"  {key_name}_overlap={report_item['overlap']} "
                    f"(left_only={report_item['left_only']}, "
                    f"right_only={report_item['right_only']})"
                )
                for example in report_item["examples"][: args.max_examples]:
                    print(
                        f"    {example['key']} "
                        f"{left}={example['left']} {right}={example['right']}"
                    )
        print("LEAK DETECTED" if leaks else "OK: no cross-split page/panel overlap")

    if leaks and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
