"""Build leakage-safe panel-ordering puzzles from downloaded comic strips."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cli import parse_args_with_config
from dataset_split_utils import SplitBudget, SplitManifests, resolve_split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/strip/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/strip/ordering"))
    parser.add_argument("--panel-count", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shuffle-per-window", type=int, default=1)
    parser.add_argument("--puzzle-num", type=int, default=500,
                        help="Total puzzle budget; defaults to 80%% train, 10%% val, 10%% test.")
    parser.add_argument("--train-puzzles", type=int, default=0,
                        help="Exact train count; setting any split count enables explicit quotas.")
    parser.add_argument("--val-puzzles", type=int, default=0,
                        help="Exact validation count; setting any split count enables explicit quotas.")
    parser.add_argument("--test-puzzles", type=int, default=0,
                        help="Exact test count; setting any split count enables explicit quotas.")
    parser.add_argument("--grid-rows", type=int, default=0,
                        help="Grid rows used to split a full strip image; 0 infers a grid.")
    parser.add_argument("--grid-cols", type=int, default=0,
                        help="Grid columns used to split a full strip image; 0 infers a grid.")
    parser.add_argument("--reading-order", choices=("ltr", "rtl"), default="ltr")
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--seed", type=int, default=0)
    return parse_args_with_config(parser, script_file=__file__)


def main() -> None:
    args = parse_args()
    validate_args(args)
    counts = build_ordering_dataset(args)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    for name in ("panel_count", "stride", "shuffle_per_window", "target_width", "target_height"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.puzzle_num <= 0:
        raise ValueError("--puzzle-num must be positive")
    if args.grid_rows < 0 or args.grid_cols < 0:
        raise ValueError("grid dimensions cannot be negative")
    if (args.grid_rows == 0) != (args.grid_cols == 0):
        raise ValueError("specify both --grid-rows and --grid-cols, or leave both at 0")


def build_ordering_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir
    manifest_path = raw_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw manifest not found: {manifest_path}; run strip/download_raw.py first")

    raw_entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_entries.sort(key=lambda row: str(row.get("strip_id", "")))
    train_q, val_q, test_q = resolve_split_counts(args, args.puzzle_num)
    budget = SplitBudget(train=train_q, val=val_q, test=test_q)
    manifests = SplitManifests(output_dir)
    rng = random.Random(args.seed)
    puzzle_index = 0
    strips_used = {"train": set(), "val": set(), "test": set()}
    strips_skipped = 0

    try:
        for entry in raw_entries:
            if budget.done:
                break
            split = budget.active_split
            if split is None:
                break
            strip_id = str(entry.get("strip_id", ""))
            if not strip_id:
                raise ValueError("raw manifest entry is missing strip_id")
            image_path = resolve_raw_path(raw_dir, str(entry["image_path"]))
            annotation_path = resolve_raw_path(raw_dir, str(entry["annotation_path"]))
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            with Image.open(image_path) as image_file:
                strip_image = image_file.convert("RGB")
            panels = extract_panels(strip_image, annotation, args.grid_rows, args.grid_cols, args.reading_order)
            if len(panels) < args.panel_count:
                strips_skipped += 1
                continue

            strip_split = split
            strips_used[split].add(strip_id)
            page_key = f"strip/{strip_id}"
            for start in range(0, len(panels) - args.panel_count + 1, args.stride):
                if budget.done or budget.active_split != strip_split:
                    break
                window = panels[start:start + args.panel_count]
                for window_variant in range(args.shuffle_per_window):
                    if budget.done or budget.active_split != strip_split or budget.remaining <= 0:
                        break
                    sample = write_puzzle(
                        puzzle_index, strip_id, page_key, window, output_dir,
                        args.target_width, args.target_height, args.image_format,
                        args.reading_order, rng, window_variant, list(strip_image.size),
                    )
                    entry_out = {
                        "puzzle_index": puzzle_index,
                        "sequence_id": sample["sequence_id"],
                        "sample_path": f"{puzzle_index:06d}/sample.json",
                        "page_key": page_key,
                        "strip_id": strip_id,
                    }
                    manifests.write(strip_split, entry_out)
                    manifests.flush()
                    budget.add(1)
                    puzzle_index += 1
                    if budget.active_split != strip_split:
                        break
    finally:
        manifests.close()

    result = {
        "dataset_name": "bghira/comicstrips-gpt4o-blip3",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "puzzle_num": args.puzzle_num,
        "panel_count": args.panel_count,
        "stride": args.stride,
        "shuffle_per_window": args.shuffle_per_window,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "reading_order": args.reading_order,
        "split_counts": budget.totals,
        "split_target_counts": {"train": train_q, "val": val_q, "test": test_q},
        "strip_counts": {name: len(values) for name, values in strips_used.items()},
        "strips_skipped_for_panel_count": strips_skipped,
        "split_policy": "Each source strip is assigned to one split; all windows and shuffles from it remain in that split.",
        "seed": args.seed,
    }
    (output_dir / "meta.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def resolve_raw_path(raw_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else raw_dir / path


def extract_panels(
    strip_image: Image.Image,
    annotation: dict[str, Any],
    grid_rows: int,
    grid_cols: int,
    reading_order: str,
) -> list[dict[str, Any]]:
    width, height = strip_image.size
    panel_images = annotation.get("panels") or annotation.get("panel_images")
    if isinstance(panel_images, list) and panel_images:
        boxes = annotation.get("panel_boxes") or annotation.get("bboxes") or []
        result = []
        for index, item in enumerate(panel_images):
            if isinstance(item, dict):
                box = parse_bbox(item.get("bbox") or item.get("box"))
                image_value = item.get("image")
                text = str(item.get("dialog_text", item.get("text", "")) or "")
            else:
                box = parse_bbox(boxes[index]) if index < len(boxes) else None
                image_value = item
                text = ""
            if isinstance(image_value, Image.Image):
                panel_image = image_value.convert("RGB")
                bbox = box or [0, 0, panel_image.width, panel_image.height]
            elif box is not None:
                panel_image = strip_image.crop(tuple(box))
                bbox = box
            else:
                continue
            result.append({"image": panel_image, "bbox": bbox, "panel_index": index, "dialog_text": text})
        if result:
            return result

    boxes = annotation.get("panel_boxes") or annotation.get("bboxes") or annotation.get("panels_bbox")
    if isinstance(boxes, list) and boxes:
        result = []
        for index, raw_box in enumerate(boxes):
            bbox = parse_bbox(raw_box)
            if bbox is None:
                continue
            x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            result.append({"image": strip_image.crop((x1, y1, x2, y2)), "bbox": [x1, y1, x2, y2], "panel_index": index, "dialog_text": ""})
        if result:
            return sort_panels(result, reading_order)

    rows, cols = infer_grid(width, height, grid_rows, grid_cols)
    result = []
    for row in range(rows):
        for col in range(cols):
            x1, x2 = round(col * width / cols), round((col + 1) * width / cols)
            y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
            result.append({
                "image": strip_image.crop((x1, y1, x2, y2)),
                "bbox": [x1, y1, x2, y2],
                "panel_index": row * cols + col,
                "dialog_text": "",
            })
    if reading_order == "rtl":
        result = [panel for row in range(rows) for panel in reversed(result[row * cols:(row + 1) * cols])]
    return result


def infer_grid(width: int, height: int, rows: int, cols: int) -> tuple[int, int]:
    if rows and cols:
        return rows, cols
    aspect = width / max(height, 1)
    if aspect >= 2.4:
        return 1, 4
    if aspect >= 1.25:
        return 2, 2
    if aspect <= 0.42:
        return 4, 1
    if aspect <= 0.8:
        return 2, 2
    return 2, 2


def parse_bbox(value: Any) -> list[int] | None:
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [round(float(number)) for number in value[:4]]
    except (TypeError, ValueError):
        return None


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]


def sort_panels(panels: list[dict[str, Any]], reading_order: str) -> list[dict[str, Any]]:
    return sorted(panels, key=lambda panel: (panel["bbox"][1], -panel["bbox"][0] if reading_order == "rtl" else panel["bbox"][0]))


def write_puzzle(
    puzzle_index: int,
    strip_id: str,
    page_key: str,
    panels: list[dict[str, Any]],
    output_dir: Path,
    target_width: int,
    target_height: int,
    image_format: str,
    reading_order: str,
    rng: random.Random,
    variant: int,
    page_size: list[int],
) -> dict[str, Any]:
    puzzle_dir = output_dir / f"{puzzle_index:06d}"
    padded_dir = puzzle_dir / "panels_padded"
    padded_dir.mkdir(parents=True, exist_ok=True)
    count = len(panels)
    input_order = list(range(count))
    rng.shuffle(input_order)
    target_order = [input_order.index(index) for index in range(count)]
    panel_rows = []
    for input_index, original_index in enumerate(input_order):
        panel = panels[original_index]
        panel_image = panel["image"].convert("RGB")
        raw_size = list(panel_image.size)
        padded, pad = pad_image(panel_image, (target_width, target_height))
        filename = f"input_{input_index:02d}.{image_format}"
        padded.save(padded_dir / filename)
        panel_rows.append({
            "panel_id": f"{strip_id}_panel_{panel['panel_index']:04d}",
            "comic_no": strip_id,
            "page_no": 0,
            "panel_no": panel["panel_index"],
            "panel_index_in_page": panel["panel_index"],
            "source_path": f"raw/strips/{strip_id}/image",
            "padded_path": f"{puzzle_index:06d}/panels_padded/{filename}",
            "raw_size": raw_size,
            "padded_size": [target_width, target_height],
            "bbox": list(panel["bbox"]),
            "page_size": page_size,
            "dialog_bboxes": [],
            "dialog_texts": [panel["dialog_text"]] if panel["dialog_text"] else [],
            "dialog_text": panel["dialog_text"],
            "text_source": "dataset annotation" if panel["dialog_text"] else "none",
            "pad": pad,
        })
    sample = {
        "sequence_id": f"strip_{strip_id}_window_{puzzle_index:06d}_variant_{variant:03d}",
        "puzzle_index": puzzle_index,
        "manga_id": "comicstrips-gpt4o-blip3",
        "chapter_id": strip_id,
        "page_id": strip_id,
        "page_key": page_key,
        "group_key": f"comicstrips/{strip_id}",
        "panel_count": count,
        "target_panel_size": [target_width, target_height],
        "panels": panel_rows,
        "input_order": input_order,
        "target_order": target_order,
        "reading_order": reading_order,
    }
    (puzzle_dir / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


def pad_image(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, list[int]]:
    target_w, target_h = target_size
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    left = (target_w - resized.width) // 2
    top = (target_h - resized.height) // 2
    canvas = Image.new("RGB", target_size, "black")
    canvas.paste(resized, (left, top))
    return canvas, [left, top, target_w - left - resized.width, target_h - top - resized.height]


if __name__ == "__main__":
    main()
