"""Build a work-level manifest for RQ2."""

from __future__ import annotations
# Resolve the project config package when this file is run directly.
import sys as _cli_sys
from pathlib import Path as _CliPath
_cli_root = next((p for p in _CliPath(__file__).resolve().parents if (p / "config" / "cli.py").is_file()), None)
if _cli_root is not None and str(_cli_root) not in _cli_sys.path:
    _cli_sys.path.insert(0, str(_cli_root))
from config.cli import parse_args_with_config

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="STYLE:DATASET:DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parse_args_with_config(parser, script_file=__file__)


def _work_id(sample: dict[str, Any]) -> str:
    value = sample.get("manga_id") or sample.get("book_id") or sample.get("comic_no")
    if not value:
        raise ValueError(f"Missing work ID in sample {sample.get('sequence_id', '<unknown>')}")
    return str(value)


def _window_id(sample: dict[str, Any]) -> str:
    panel_ids = sorted(str(p.get("panel_id", p.get("padded_path", ""))) for p in sample.get("panels", []))
    digest = hashlib.sha1("\n".join(panel_ids).encode("utf-8")).hexdigest()[:16]
    return "/".join((_work_id(sample), str(sample.get("chapter_id", "")), str(sample.get("page_id", sample.get("page_index", ""))), digest))


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train/val ratios must be positive and sum to less than 1")
    rows: list[dict[str, Any]] = []
    works: dict[str, set[str]] = {"manga": set(), "comics": set()}
    sources = []
    for spec in args.source:
        try:
            style, dataset_name, raw_dir = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"Invalid --source {spec!r}; expected STYLE:DATASET:DIR") from exc
        style = style.lower()
        if style not in works:
            raise ValueError(f"Unsupported style {style!r}; expected manga or comics")
        source_dir = Path(raw_dir).resolve()
        manifest = source_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        source_id = f"{style}:{dataset_name}"
        sources.append({"source_id": source_id, "source_dir": str(source_dir)})
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            sample = json.loads((source_dir / str(entry["sample_path"])).read_text(encoding="utf-8"))
            work = f"{source_id}:{_work_id(sample)}"
            works[style].add(work)
            window = _window_id(sample)
            rows.append({"source_id": source_id, "source_dir": str(source_dir), "sample_path": str(entry["sample_path"]).replace("\\", "/"), "sequence_id": str(sample["sequence_id"]), "dataset": dataset_name, "style": style, "work_id": _work_id(sample), "canonical_work_id": work, "page_key": window.rsplit("/", 1)[0], "window_id": window, "panel_count": int(sample["panel_count"])})
    if not works["manga"] or not works["comics"]:
        raise ValueError("RQ2 requires both manga and comics works")
    rng = random.Random(args.seed)
    assignments: dict[str, str] = {}
    for style, work_ids in works.items():
        ordered = sorted(work_ids)
        rng.shuffle(ordered)
        train_end = int(len(ordered) * args.train_ratio)
        val_end = train_end + int(len(ordered) * args.val_ratio)
        for index, work in enumerate(ordered):
            assignments[work] = "train" if index < train_end else "val" if index < val_end else "test"
    for style, work_ids in works.items():
        if any(not any(assignments[work] == split for work in work_ids) for split in ("train", "val", "test")):
            raise ValueError(f"Not enough distinct {style} works to populate all splits; add works or change ratios")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row["split"] = assignments[row["canonical_work_id"]]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_suffix(".meta.json").write_text(json.dumps({"seed": args.seed, "sources": sources, "work_counts": {style: {split: sum(assignments[w] == split for w in work_ids) for split in ("train", "val", "test")} for style, work_ids in works.items()}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
