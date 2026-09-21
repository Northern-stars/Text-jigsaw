"""Create Manga109 panel-ordering datasets for the MangaZero training pipeline.

Reads Manga109 page images and XML annotations (frame bboxes), extracts panel
images from each page, sorts them in reading order, shuffles into input order,
and writes the output in the exact format consumed by
``Solver/env/mangazero_panel_env.py`` (MangaZeroPanelOrderingDataset).

Default output layout::

    output/
      manifest.jsonl
      <puzzle_index>/
        sample.json
        panels_padded/input_00.jpg
        ...

Example::

    python Manga109/create_ordering.py \\
        --manga109-dir Data/Manga109 \\
        --output-dir Data/Manga109/ordering_output \\
        --panel-count 6 --puzzle-num 200

The tool reads <frame> elements from each page's annotation, computes their
natural reading order (top-to-bottom, right-to-left for manga), and produces
fixed-length sliding-window samples.  Pages with too few frames are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_book_pages(xml_path: Path) -> list[dict[str, Any]]:
    """Parse all page annotations from one Manga109 book XML.

    Returns list of dicts with keys: index, width, height, frames, texts.
    Each frame/text has id, xmin, ymin, xmax, ymax, content (for text).
    """
    root = ElementTree.parse(str(xml_path)).getroot()
    pages_el = None
    for child in root:
        if _local_tag(child.tag) == "pages":
            pages_el = child
            break
    if pages_el is None:
        return []

    pages: list[dict[str, Any]] = []
    for page_el in pages_el:
        if _local_tag(page_el.tag) != "page":
            continue
        page: dict[str, Any] = {
            "index": int(page_el.attrib.get("index", -1)),
            "width": int(page_el.attrib.get("width", 0)),
            "height": int(page_el.attrib.get("height", 0)),
            "frames": [],
            "texts": [],
        }
        for child in page_el:
            ctag = _local_tag(child.tag)
            if ctag == "frame":
                try:
                    xmin = float(child.attrib["xmin"])
                    ymin = float(child.attrib["ymin"])
                    xmax = float(child.attrib["xmax"])
                    ymax = float(child.attrib["ymax"])
                except (KeyError, ValueError):
                    continue
                page["frames"].append({
                    "id": child.attrib.get("id", ""),
                    "xmin": xmin, "ymin": ymin,
                    "xmax": xmax, "ymax": ymax,
                })
            elif ctag == "text":
                content = (child.text or "").strip()
                if not content:
                    continue
                try:
                    xmin = float(child.attrib["xmin"])
                    ymin = float(child.attrib["ymin"])
                    xmax = float(child.attrib["xmax"])
                    ymax = float(child.attrib["ymax"])
                except (KeyError, ValueError):
                    continue
                page["texts"].append({
                    "id": child.attrib.get("id", ""),
                    "xmin": xmin, "ymin": ymin,
                    "xmax": xmax, "ymax": ymax,
                    "content": content,
                })
        pages.append(page)
    return pages


# ---------------------------------------------------------------------------
# Reading-order sort (manga: right-to-left)
# ---------------------------------------------------------------------------

def sort_frames_reading_order(
    frames: list[dict[str, Any]],
    page_height: int,
    rtl: bool = True,
) -> list[dict[str, Any]]:
    """Sort frame bboxes in manga reading order (top-to-bottom, right-to-left).

    Uses vertical band detection to group frames into rows, then sorts
    within each row from right to left.
    """
    if not frames:
        return []

    # Group into vertical bands
    y_centers = [(f["ymin"] + f["ymax"]) / 2.0 for f in frames]
    heights = [f["ymax"] - f["ymin"] for f in frames]
    avg_h = sum(heights) / len(heights) if heights else page_height / 4
    band_threshold = max(avg_h * 0.5, 10.0)

    # Simple row assignment: group frames whose y-centers are close
    rows: list[list[int]] = []
    row_centers: list[float] = []
    for i, yc in enumerate(y_centers):
        assigned = False
        for r_idx, rc in enumerate(row_centers):
            if abs(yc - rc) <= band_threshold:
                rows[r_idx].append(i)
                row_centers[r_idx] = sum(
                    y_centers[j] for j in rows[r_idx]
                ) / len(rows[r_idx])
                assigned = True
                break
        if not assigned:
            rows.append([i])
            row_centers.append(yc)

    # Sort rows top-to-bottom
    row_order = sorted(range(len(rows)), key=lambda r: row_centers[r])

    sorted_frames: list[dict[str, Any]] = []
    for r_idx in row_order:
        row_indices = rows[r_idx]
        if rtl:
            row_indices.sort(key=lambda i: -frames[i]["xmin"])
        else:
            row_indices.sort(key=lambda i: frames[i]["xmin"])
        for i in row_indices:
            sorted_frames.append(frames[i])

    return sorted_frames


# ---------------------------------------------------------------------------
# Text assignment to frames
# ---------------------------------------------------------------------------

def _bbox_overlap_ratio(
    tb: tuple[float, float, float, float],
    fb: tuple[float, float, float, float],
) -> float:
    ix0 = max(tb[0], fb[0])
    iy0 = max(tb[1], fb[1])
    ix1 = min(tb[2], fb[2])
    iy1 = min(tb[3], fb[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    text_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
    return inter / text_area if text_area > 0 else 0.0


def assign_texts_to_frames(
    texts: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    min_overlap: float = 0.3,
) -> dict[str, str]:
    """Assign text annotations to frames. Returns frame_id -> joined text."""
    result: dict[str, str] = {f["id"]: "" for f in frames}
    for ta in texts:
        tb = (ta["xmin"], ta["ymin"], ta["xmax"], ta["ymax"])
        best_overlap = 0.0
        best_id: str | None = None
        for frame in frames:
            fb = (frame["xmin"], frame["ymin"], frame["xmax"], frame["ymax"])
            ratio = _bbox_overlap_ratio(tb, fb)
            if ratio > best_overlap:
                best_overlap = ratio
                best_id = frame["id"]
        if best_id is not None and best_overlap >= min_overlap:
            existing = result[best_id]
            if existing:
                result[best_id] += " " + ta["content"]
            else:
                result[best_id] = ta["content"]
    return result


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _find_page_image(book_images_dir: Path, page_index: int) -> Path | None:
    for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
        for fmt in [f"{page_index:03d}", str(page_index)]:
            candidate = book_images_dir / f"{fmt}.{ext}"
            if candidate.exists():
                return candidate
    return None


def pad_image_to_target(
    image: Image.Image,
    target_size: tuple[int, int],
) -> tuple[Image.Image, list[int]]:
    """Resize image to fit inside target_size, pad with black, return canvas + pad."""
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


# ---------------------------------------------------------------------------
# Main dataset creation
# ---------------------------------------------------------------------------

def create_manga109_ordering(
    manga109_dir: Path,
    output_dir: Path,
    panel_count: int = 6,
    puzzle_num: int = 200,
    target_width: int = 224,
    target_height: int = 224,
    min_frames: int = 3,
    image_format: str = "jpg",
    min_text_overlap: float = 0.3,
    rtl: bool = True,
    books: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Create panel-ordering puzzle samples from Manga109.

    Output format is identical to Mangazero/build_dataset.py so that
    MangaZeroPanelOrderingDataset can consume it without changes.
    """
    annotations_dir = manga109_dir / "annotations"
    images_dir = manga109_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if books:
        book_names = books
    else:
        book_names = sorted(p.stem for p in annotations_dir.glob("*.xml"))

    # Collect all (book, page) pairs with enough frames
    page_candidates: list[tuple[str, dict[str, Any], Path]] = []
    for book_name in book_names:
        xml_path = annotations_dir / f"{book_name}.xml"
        book_images_dir = images_dir / book_name
        if not xml_path.exists() or not book_images_dir.exists():
            continue
        pages = parse_book_pages(xml_path)
        for page_info in pages:
            if len(page_info["frames"]) < min_frames:
                continue
            page_idx = int(page_info["index"])
            image_path = _find_page_image(book_images_dir, page_idx)
            if image_path is None:
                continue
            page_candidates.append((book_name, page_info, image_path))

    print(f"Found {len(page_candidates)} pages with >= {min_frames} frames across {len(book_names)} books")

    # Generate puzzle samples via sliding window
    puzzle_count = 0
    manifest_entries: list[dict[str, Any]] = []
    skipped_windows = 0

    page_iter = page_candidates
    if tqdm is not None:
        page_iter = tqdm(page_candidates, desc="generating puzzles", unit="page", dynamic_ncols=True)

    for book_name, page_info, image_path in page_iter:
        if puzzle_count >= puzzle_num:
            break

        frames = list(page_info["frames"])
        texts = list(page_info["texts"])
        page_w = int(page_info["width"])
        page_h = int(page_info["height"])
        page_idx = int(page_info["index"])

        # Assign text to frames
        text_map = assign_texts_to_frames(texts, frames, min_text_overlap)

        # Sort frames in reading order
        ordered_frames = sort_frames_reading_order(frames, page_h, rtl=rtl)

        # Read the page image once
        with Image.open(image_path) as full_img:
            full_img = full_img.convert("RGB")

        # Sliding window
        for start in range(0, len(ordered_frames) - panel_count + 1):
            if puzzle_count >= puzzle_num:
                break

            window = ordered_frames[start : start + panel_count]

            # Extract and pad each panel image
            puzzle_dir = output_dir / f"{puzzle_count:06d}"
            padded_dir = puzzle_dir / "panels_padded"
            padded_dir.mkdir(parents=True, exist_ok=True)

            input_order = list(range(panel_count))
            rng.shuffle(input_order)
            target_order = [input_order.index(i) for i in range(panel_count)]

            shuffled_panels: list[dict[str, Any]] = []
            for input_idx, original_idx in enumerate(input_order):
                frame = window[original_idx]
                # Clamp bbox to page bounds
                x1 = max(0, min(page_w, round(frame["xmin"])))
                y1 = max(0, min(page_h, round(frame["ymin"])))
                x2 = max(0, min(page_w, round(frame["xmax"])))
                y2 = max(0, min(page_h, round(frame["ymax"])))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = full_img.crop((x1, y1, x2, y2))
                padded_name = f"input_{input_idx:02d}.{image_format}"
                padded_img, pad = pad_image_to_target(crop, (target_width, target_height))
                padded_img.save(padded_dir / padded_name)

                dialog_text = text_map.get(frame["id"], "")

                shuffled_panels.append({
                    "panel_id": f"p{page_idx:04d}_f{original_idx}",
                    "manga_id": book_name,
                    "chapter_id": "default",
                    "source_image_path": str(image_path.relative_to(manga109_dir)),
                    "page_id": f"{book_name}_p{page_idx:04d}",
                    "page_index": page_idx,
                    "panel_index_in_page": original_idx,
                    "global_order": original_idx,
                    "bbox": [x1, y1, x2, y2],
                    "page_size": [page_w, page_h],
                    "padded_path": f"{puzzle_count:06d}/panels_padded/{padded_name}",
                    "raw_size": [x2 - x1, y2 - y1],
                    "padded_size": [target_width, target_height],
                    "pad": pad,
                    "caption": "",
                    "dialog_bboxes": [],
                    "dialog_texts": [dialog_text] if dialog_text else [],
                    "dialog_text": dialog_text,
                    "character_ids": [],
                    "character_bboxes": [],
                    "character_types": [],
                })

            if len(shuffled_panels) < panel_count:
                skipped_windows += 1
                continue

            sample = {
                "sequence_id": f"{book_name}_p{page_idx:04d}_w{start:02d}",
                "puzzle_index": puzzle_count,
                "manga_id": book_name,
                "chapter_id": "default",
                "page_id": f"{book_name}_p{page_idx:04d}",
                "page_index": page_idx,
                "group_key": f"{book_name}/default/{book_name}_p{page_idx:04d}",
                "page_key": f"{book_name}/default/{book_name}_p{page_idx:04d}",
                "panel_count": panel_count,
                "target_panel_size": [target_width, target_height],
                "panels": shuffled_panels,
                "input_order": input_order,
                "target_order": target_order,
                "text_source": "dialog_text",
            }

            with (puzzle_dir / "sample.json").open("w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)

            manifest_entries.append({
                "sequence_id": sample["sequence_id"],
                "sample_path": f"{puzzle_count:06d}/sample.json",
                "manga_id": book_name,
                "page_index": page_idx,
                "panel_count": panel_count,
            })

            puzzle_count += 1

    # Write manifest.jsonl
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats = {
        "puzzle_count": puzzle_count,
        "skipped_windows": skipped_windows,
        "pages_with_frames": len(page_candidates),
        "books_scanned": len(book_names),
    }
    print(
        f"Done. Puzzles: {puzzle_count}. Skipped windows: {skipped_windows}. "
        f"Pages with frames: {len(page_candidates)}."
    )
    print(f"Output written to: {output_dir}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Manga109 panel-ordering puzzle dataset."
    )
    p.add_argument(
        "--manga109-dir", type=Path, default=Path("Data/Manga109"),
        help="Root directory of Manga109 (containing images/ and annotations/).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("Data/Manga109/ordering_output"),
        help="Output directory for puzzles and manifest.jsonl.",
    )
    p.add_argument("--panel-count", type=int, default=6, help="Panels per sample (sliding window size).")
    p.add_argument("--puzzle-num", type=int, default=200, help="Maximum number of puzzle samples.")
    p.add_argument("--target-width", type=int, default=224, help="Padded panel image width.")
    p.add_argument("--target-height", type=int, default=224, help="Padded panel image height.")
    p.add_argument("--min-frames", type=int, default=3, help="Minimum frames on a page to include it.")
    p.add_argument("--min-text-overlap", type=float, default=0.3, help="Min overlap to assign text to a frame.")
    p.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    p.add_argument("--rtl", action="store_true", default=True, help="Right-to-left reading order (manga).")
    p.add_argument("--no-rtl", action="store_false", dest="rtl")
    p.add_argument("--books", nargs="*", help="Process only these book titles.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_manga109_ordering(
        manga109_dir=args.manga109_dir,
        output_dir=args.output_dir,
        panel_count=args.panel_count,
        puzzle_num=args.puzzle_num,
        target_width=args.target_width,
        target_height=args.target_height,
        min_frames=args.min_frames,
        image_format=args.image_format,
        min_text_overlap=args.min_text_overlap,
        rtl=args.rtl,
        books=args.books,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
