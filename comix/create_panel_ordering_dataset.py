"""Build fixed-count panel-ordering puzzles from the COMICS dataset.

The script expects the COMICS data that ``comics/setup.sh`` downloads:

    comics/data/raw_panel_images/<comic_no>/<page_no>_<panel_no>.jpg
    comics/data/COMICS_ocr_file.csv
    comics/data/predadpages.txt

The OCR CSV uses the official COMICS columns:
``comic_no,page_no,panel_no,textbox_no,dialog_or_narration,text,x1,y1,x2,y2``.

Output is written in the same layout as ``Mangazero/build_dataset.py`` so the
existing MangaZero training scripts can consume it directly:

    comics/panel_ordering_dataset/
      manifest.jsonl
      meta.json
      000000/sample.json
      000000/panels_padded/input_00.jpg ...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


OCR_FIELDS = [
    "comic_no",
    "page_no",
    "panel_no",
    "textbox_no",
    "dialog_or_narration",
    "text",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass
class TextBox:
    textbox_no: int
    dialog_or_narration: int
    text: str
    bbox: list[float]


@dataclass
class Panel:
    comic_no: int
    page_no: int
    panel_no: int
    source_path: Path
    text_boxes: list[TextBox] = field(default_factory=list)

    @property
    def dialog_text(self) -> str:
        texts = [box.text.strip() for box in self.text_boxes if box.text.strip()]
        return " ".join(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create COMICS panel-ordering puzzles.")
    parser.add_argument("--panels-dir", type=Path, default=Path("comics/data/raw_panel_images"))
    parser.add_argument("--ocr-csv", type=Path, default=Path("comics/data/COMICS_ocr_file.csv"))
    parser.add_argument("--ad-pages", type=Path, default=Path("comics/data/predadpages.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("comics/panel_ordering_dataset"))
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--shuffle-per-window", type=int, default=1)
    parser.add_argument("--puzzle-num", type=int, default=500)
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.panel_count <= 0:
        raise ValueError("--panel-count must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.shuffle_per_window <= 0:
        raise ValueError("--shuffle-per-window must be positive")
    if args.puzzle_num < 0:
        raise ValueError("--puzzle-num must be non-negative")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("target size must be positive")


def load_ad_pages(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    }


def load_panels(
    panels_dir: Path,
    ocr_csv: Path,
    ad_pages: set[str],
) -> list[Panel]:
    if not panels_dir.is_dir():
        raise FileNotFoundError(f"COMICS panel image directory not found: {panels_dir}")
    if not ocr_csv.exists():
        raise FileNotFoundError(f"COMICS OCR CSV not found: {ocr_csv}")

    existing_panels: set[tuple[int, int, int]] = set()
    for image_path in panels_dir.rglob("*.jpg"):
        try:
            comic_no = int(image_path.parent.name)
            page_part, panel_part = image_path.stem.split("_", 1)
            page_no = int(page_part)
            panel_no = int(panel_part)
        except (ValueError, TypeError):
            continue
        existing_panels.add((comic_no, page_no, panel_no))

    by_key: dict[tuple[int, int, int], Panel] = {}
    missing_images = 0
    for row in csv.DictReader(ocr_csv.open("r", encoding="utf-8", errors="replace")):
        if any(field_name not in row for field_name in OCR_FIELDS):
            raise ValueError(
                "COMICS OCR CSV is missing required columns. Expected: "
                + ", ".join(OCR_FIELDS)
            )
        comic_no = int(row["comic_no"])
        page_no = int(row["page_no"])
        panel_no = int(row["panel_no"])
        if f"{comic_no}---{page_no}" in ad_pages:
            continue
        if (comic_no, page_no, panel_no) not in existing_panels:
            missing_images += 1
            continue
        key = (comic_no, page_no, panel_no)
        stem = f"{page_no}_{panel_no}"
        source_path = panels_dir / str(comic_no) / f"{stem}.jpg"
        panel = by_key.setdefault(
            key,
            Panel(
                comic_no=comic_no,
                page_no=page_no,
                panel_no=panel_no,
                source_path=source_path,
            ),
        )
        try:
            bbox = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        except (TypeError, ValueError):
            bbox = [0.0, 0.0, 0.0, 0.0]
        panel.text_boxes.append(
            TextBox(
                textbox_no=int(row["textbox_no"]) if str(row["textbox_no"]).strip() else 0,
                dialog_or_narration=(
                    int(row["dialog_or_narration"])
                    if str(row["dialog_or_narration"]).strip()
                    else 0
                ),
                text=row["text"],
                bbox=bbox,
            )
        )

    if missing_images:
        print(f"Skipped {missing_images} OCR rows without a matching panel image.", flush=True)
    return list(by_key.values())


def group_by_page(panels: list[Panel]) -> dict[tuple[int, int], list[Panel]]:
    grouped: dict[tuple[int, int], list[Panel]] = {}
    for panel in panels:
        key = (panel.comic_no, panel.page_no)
        grouped.setdefault(key, []).append(panel)
    for page_panels in grouped.values():
        page_panels.sort(key=lambda panel: panel.panel_no)
    return grouped


def pad_image(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, list[int]]:
    target_w, target_h = target_size
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", target_size, "black")
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, [left, top, target_w - left - new_w, target_h - top - new_h]


def panel_to_json(
    panel: Panel,
    padded_rel_path: str,
    panel_index: int,
    target_size: tuple[int, int],
    raw_size: tuple[int, int],
    pad: list[int],
) -> dict[str, Any]:
    width, height = raw_size
    return {
        "panel_id": f"{panel.comic_no}_{panel.page_no}_panel_{panel.panel_no:03d}",
        "comic_no": panel.comic_no,
        "page_no": panel.page_no,
        "panel_no": panel.panel_no,
        "panel_index_in_page": panel.panel_no,
        "page_index": panel.page_no,
        "source_path": str(panel.source_path),
        "padded_path": padded_rel_path,
        "raw_size": [width, height],
        "padded_size": list(target_size),
        "bbox": [0, 0, width, height],
        "page_size": [width, height],
        "pad": pad,
        "dialog_bboxes": [box.bbox for box in panel.text_boxes],
        "dialog_texts": [box.text for box in panel.text_boxes],
        "dialog_text": panel.dialog_text,
        "text_source": "COMICS OCR",
    }


def write_puzzle_directory(
    puzzle_index: int,
    page_key: str,
    window: list[Panel],
    output_dir: Path,
    target_size: tuple[int, int],
    image_format: str,
    rng: random.Random,
) -> dict[str, Any]:
    puzzle_dir = output_dir / f"{puzzle_index:06d}"
    padded_dir = puzzle_dir / "panels_padded"
    padded_dir.mkdir(parents=True, exist_ok=True)

    panel_count = len(window)
    input_order = list(range(panel_count))
    rng.shuffle(input_order)
    target_order = [input_order.index(index) for index in range(panel_count)]

    shuffled_panels = []
    for input_index, original_index in enumerate(input_order):
        panel = window[original_index]
        padded_name = f"input_{input_index:02d}.{image_format}"
        padded_path = padded_dir / padded_name
        source_image = Image.open(panel.source_path).convert("RGB")
        raw_size = source_image.size
        padded, pad = pad_image(source_image, target_size)
        padded.save(padded_path)
        shuffled_panels.append(
            panel_to_json(
                panel=panel,
                padded_rel_path=f"{puzzle_index:06d}/panels_padded/{padded_name}",
                panel_index=original_index,
                target_size=target_size,
                raw_size=raw_size,
                pad=pad,
            )
        )

    sample = {
        "sequence_id": f"comics_{page_key}_puzzle_{puzzle_index:06d}",
        "puzzle_index": puzzle_index,
        "comic_no": window[0].comic_no,
        "page_no": window[0].page_no,
        "page_key": page_key,
        "panel_count": panel_count,
        "target_panel_size": list(target_size),
        "panels": shuffled_panels,
        "input_order": input_order,
        "target_order": target_order,
        "text_source": "COMICS OCR",
    }
    with (puzzle_dir / "sample.json").open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)
    return sample


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ad_pages = load_ad_pages(args.ad_pages)
    panels = load_panels(args.panels_dir, args.ocr_csv, ad_pages)
    if not panels:
        raise RuntimeError("No COMICS panels found after reading OCR CSV and image directory.")

    page_groups = group_by_page(panels)
    page_group_count = len(page_groups)
    panel_count_total = len(panels)
    puzzle_count = 0
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for (comic_no, page_no), page_panels in sorted(page_groups.items()):
            page_key = f"{comic_no}_{page_no}"
            for start in range(0, len(page_panels), args.stride):
                if puzzle_count >= args.puzzle_num:
                    break
                if start + args.panel_count > len(page_panels):
                    continue
                window = page_panels[start:start + args.panel_count]
                for _ in range(args.shuffle_per_window):
                    if puzzle_count >= args.puzzle_num:
                        break
                    sample_json = write_puzzle_directory(
                        puzzle_index=puzzle_count,
                        page_key=page_key,
                        window=window,
                        output_dir=output_dir,
                        target_size=(args.target_width, args.target_height),
                        image_format=args.image_format,
                        rng=rng,
                    )
                    manifest_handle.write(
                        json.dumps(
                            {
                                "puzzle_index": puzzle_count,
                                "sequence_id": sample_json["sequence_id"],
                                "sample_path": f"{puzzle_count:06d}/sample.json",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    manifest_handle.flush()
                    puzzle_count += 1
            if puzzle_count >= args.puzzle_num:
                break

    meta = {
        "source": "COMICS",
        "panels_dir": str(args.panels_dir),
        "ocr_csv": str(args.ocr_csv),
        "ad_pages": str(args.ad_pages),
        "panel_count": args.panel_count,
        "stride": args.stride,
        "shuffle_per_window": args.shuffle_per_window,
        "puzzle_num": args.puzzle_num,
        "target_panel_size": [args.target_width, args.target_height],
        "page_group_count": page_group_count,
        "panel_count_total": panel_count_total,
        "puzzle_count": puzzle_count,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
