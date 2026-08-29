"""Create a jigsaw OCR dataset from newspaper page images and ALTO XML.

The script divides each available newspaper image into grid cells, optionally
selects a floating square crop inside every cell as the final piece, reads OCR
word coordinates from the paired ALTO XML, estimates character soft boxes inside
each word, and records which characters/text belong to every final piece.

Default output layout:

    output/
      manifest.json
      puzzles/<page_id>/
        pieces/r000_c000.jpg
        label.json

Example:
    python create_jigsaw_ocr_dataset.py --dataset-dir ocr_dataset --output-dir output --rows 4 --cols 4 --limit 10
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from PIL import Image

from create_ocr_dataset import create_image_xml_pairs

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PIECE_RESOLUTION = 384


@dataclass(frozen=True)
class CharBox:
    char: str
    word: str
    word_id: str
    confidence: float | None
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PieceSpec:
    row: int
    col: int
    grid_box: tuple[int, int, int, int]
    piece_box: tuple[int, int, int, int]
    piece_path: Path | None = None

    def piece_name(self, image_format: str) -> str:
        return f"r{self.row:03d}_c{self.col:03d}.{image_format.lower()}"

    def with_piece_path(self, piece_path: Path) -> "PieceSpec":
        return PieceSpec(
            row=self.row,
            col=self.col,
            grid_box=self.grid_box,
            piece_box=self.piece_box,
            piece_path=piece_path,
        )


def parse_alto_char_boxes(xml_path: Path, image_size: tuple[int, int]) -> list[CharBox]:
    """Read ALTO word boxes and estimate per-character boxes in image pixels."""

    image_width, image_height = image_size
    xml_width, xml_height = read_alto_page_size(xml_path)
    scale_x = image_width / xml_width if xml_width else 1.0
    scale_y = image_height / xml_height if xml_height else 1.0

    chars: list[CharBox] = []
    xml_iter = progress(
        ElementTree.iterparse(xml_path, events=("end",)),
        desc=f"Parsing XML {xml_path.name}",
        unit="node",
        leave=False,
    )
    for _, element in xml_iter:
        if local_name(element.tag) != "String":
            element.clear()
            continue

        content = element.attrib.get("CONTENT", "")
        if not content:
            element.clear()
            continue

        try:
            x = float(element.attrib["HPOS"]) * scale_x
            y = float(element.attrib["VPOS"]) * scale_y
            width = float(element.attrib["WIDTH"]) * scale_x
            height = float(element.attrib["HEIGHT"]) * scale_y
        except (KeyError, ValueError):
            element.clear()
            continue

        confidence = parse_float(element.attrib.get("WC"))
        word_id = element.attrib.get("ID", "")
        chars.extend(split_word_into_char_boxes(content, word_id, confidence, x, y, width, height))
        element.clear()

    return chars


def read_alto_page_size(xml_path: Path) -> tuple[float | None, float | None]:
    try:
        xml_iter = progress(
            ElementTree.iterparse(xml_path, events=("end",)),
            desc=f"Reading page size {xml_path.name}",
            unit="node",
            leave=False,
        )
        for _, element in xml_iter:
            if local_name(element.tag) == "Page":
                width = parse_float(element.attrib.get("WIDTH"))
                height = parse_float(element.attrib.get("HEIGHT"))
                return width, height
            element.clear()
    except ElementTree.ParseError:
        return None, None
    return None, None


def split_word_into_char_boxes(
    word: str,
    word_id: str,
    confidence: float | None,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[CharBox]:
    if not word:
        return []

    char_width = width / len(word)
    chars: list[CharBox] = []
    for char_index, char in enumerate(word):
        char_x1 = x + char_index * char_width
        char_x2 = x + (char_index + 1) * char_width
        chars.append(
            CharBox(
                char=char,
                word=word,
                word_id=word_id,
                confidence=confidence,
                bbox=(char_x1, y, char_x2, y + height),
            )
        )
    return chars


def create_jigsaw_dataset(
    dataset_dir: Path = Path("ocr_dataset"),
    output_dir: Path = Path("output"),
    rows: int = 3,
    cols: int = 3,
    limit: int | None = None,
    max_images: int | None = None,
    image_format: str = "jpg",
    min_char_overlap: float = 0.5,
    piece_resolution: int = PIECE_RESOLUTION,
    seed: int | None = 0,
) -> dict[str, object]:
    """Cut paired newspaper pages into one labeled puzzle per page."""

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    if not 0 < min_char_overlap <= 1:
        raise ValueError("min_char_overlap must be in the range (0, 1]")
    if piece_resolution == 0 or piece_resolution < -1:
        raise ValueError("piece_resolution must be a positive integer or -1 for native grid pieces")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be a positive integer")

    puzzles_dir = output_dir / "puzzles"
    puzzles_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    pairs = create_image_xml_pairs(dataset_dir)
    if limit is not None:
        pairs = pairs[:limit]

    puzzle_summaries: list[dict[str, object]] = []
    total_piece_count = 0
    skipped_pages: list[dict[str, object]] = []

    pages_iter = progress(pairs, desc="Processing pages", unit="page", total=len(pairs))
    for page_index, (image_path, xml_path) in enumerate(pages_iter):
        if max_images is not None and len(puzzle_summaries) >= max_images:
            break

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            page_id = image_path.stem
            try:
                piece_specs = build_piece_specs(
                    image_width=image_width,
                    image_height=image_height,
                    rows=rows,
                    cols=cols,
                    piece_resolution=piece_resolution,
                    rng=rng,
                    page_id=page_id,
                )
            except ValueError as exc:
                skipped_pages.append(
                    {
                        "page_index": page_index,
                        "page_id": page_id,
                        "image_path": path_to_json(image_path),
                        "reason": str(exc),
                    }
                )
                continue

            puzzle_dir = puzzles_dir / page_id
            pieces_dir = puzzle_dir / "pieces"
            pieces_dir.mkdir(parents=True, exist_ok=True)
            piece_specs = [
                piece_spec.with_piece_path(pieces_dir / piece_spec.piece_name(image_format))
                for piece_spec in piece_specs
            ]
            char_boxes = parse_alto_char_boxes(xml_path, image.size)

            page_info = {
                "page_index": page_index,
                "page_id": page_id,
                "image_path": path_to_json(image_path),
                "xml_path": path_to_json(xml_path),
                "width": image_width,
                "height": image_height,
                "char_count": len(char_boxes),
            }
            piece_labels: list[dict[str, object]] = []

            pieces_iter = progress(
                piece_specs,
                desc=f"Writing pieces {page_id}",
                unit="piece",
                total=len(piece_specs),
                leave=False,
            )
            for piece_spec in pieces_iter:
                if piece_spec.piece_path is None:
                    raise RuntimeError("piece_path must be assigned before writing pieces")

                image.crop(piece_spec.piece_box).save(piece_spec.piece_path)
                piece_chars = chars_in_piece(char_boxes, piece_spec.piece_box, min_char_overlap)

                piece_labels.append(
                    {
                        "piece_id": piece_spec.piece_path.stem,
                        "piece_path": path_to_json(piece_spec.piece_path),
                        "page_id": page_id,
                        "page_index": page_index,
                        "row": piece_spec.row,
                        "col": piece_spec.col,
                        "grid_bbox": list(piece_spec.grid_box),
                        "bbox": list(piece_spec.piece_box),
                        "width": piece_spec.piece_box[2] - piece_spec.piece_box[0],
                        "height": piece_spec.piece_box[3] - piece_spec.piece_box[1],
                        "text": text_from_piece_chars(piece_chars),
                        "chars": piece_chars,
                    }
                )

            puzzle = {
                "meta": {
                    "dataset_dir": path_to_json(dataset_dir),
                    "output_dir": path_to_json(output_dir),
                    "puzzle_dir": path_to_json(puzzle_dir),
                    "rows": rows,
                    "cols": cols,
                    "piece_count": len(piece_labels),
                    "piece_resolution": piece_resolution,
                    "min_char_overlap": min_char_overlap,
                    "coordinate_note": build_coordinate_note(piece_resolution),
                },
                "page": page_info,
                "pieces": piece_labels,
            }

            label_path = puzzle_dir / "label.json"
            with label_path.open("w", encoding="utf-8") as handle:
                json.dump(puzzle, handle, ensure_ascii=False, indent=2)

            total_piece_count += len(piece_labels)
            puzzle_summaries.append(
                {
                    "page_index": page_index,
                    "page_id": page_id,
                    "puzzle_dir": path_to_json(puzzle_dir),
                    "label_path": path_to_json(label_path),
                    "piece_count": len(piece_labels),
                    "char_count": len(char_boxes),
                }
            )

    dataset = {
        "meta": {
            "dataset_dir": path_to_json(dataset_dir),
            "output_dir": path_to_json(output_dir),
            "rows": rows,
            "cols": cols,
            "input_limit": limit,
            "max_images": max_images,
            "piece_count": total_piece_count,
            "piece_resolution": piece_resolution,
            "page_count": len(puzzle_summaries),
            "skipped_page_count": len(skipped_pages),
            "min_char_overlap": min_char_overlap,
            "label_note": "Each puzzle has its own label.json under output_dir/puzzles/<page_id>/.",
        },
        "puzzles": puzzle_summaries,
        "skipped_pages": skipped_pages,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    return dataset


def grid_piece_box(width: int, height: int, rows: int, cols: int, row: int, col: int) -> tuple[int, int, int, int]:
    left = round(col * width / cols)
    upper = round(row * height / rows)
    right = round((col + 1) * width / cols)
    lower = round((row + 1) * height / rows)
    return left, upper, right, lower


def build_piece_specs(
    image_width: int,
    image_height: int,
    rows: int,
    cols: int,
    piece_resolution: int,
    rng: random.Random,
    page_id: str,
) -> list[PieceSpec]:
    specs: list[PieceSpec] = []
    for row in range(rows):
        for col in range(cols):
            grid_box = grid_piece_box(image_width, image_height, rows, cols, row, col)
            if piece_resolution == -1:
                piece_box = grid_box
            else:
                piece_box = square_piece_box(grid_box, piece_resolution, rng, page_id, row, col)
            specs.append(
                PieceSpec(
                    row=row,
                    col=col,
                    grid_box=grid_box,
                    piece_box=piece_box,
                )
            )
    return specs


def build_coordinate_note(piece_resolution: int) -> str:
    if piece_resolution == -1:
        return (
            "Native grid mode: grid_bbox and bbox are identical page-pixel grid cells; "
            "char boxes are relative to that full cell piece."
        )
    return (
        "Square crop mode: grid_bbox is the original grid cell in page pixels; "
        "bbox is the final square piece in page pixels; char boxes are relative to the final piece."
    )


def square_piece_box(
    grid_box: tuple[int, int, int, int],
    piece_resolution: int,
    rng: random.Random,
    page_id: str,
    row: int,
    col: int,
) -> tuple[int, int, int, int]:
    left, upper, right, lower = grid_box
    grid_width = right - left
    grid_height = lower - upper
    if piece_resolution > grid_width or piece_resolution > grid_height:
        raise ValueError(
            f"grid cell too small for piece_resolution={piece_resolution} "
            f"{row},{col} on page {page_id}: cell size is {grid_width}x{grid_height}"
        )

    max_left = right - piece_resolution
    max_upper = lower - piece_resolution
    piece_left = rng.randint(left, max_left)
    piece_upper = rng.randint(upper, max_upper)
    return (
        piece_left,
        piece_upper,
        piece_left + piece_resolution,
        piece_upper + piece_resolution,
    )


def chars_in_piece(
    char_boxes: Iterable[CharBox],
    piece_box: tuple[int, int, int, int],
    min_overlap: float,
) -> list[dict[str, object]]:
    piece_left, piece_top, piece_right, piece_bottom = piece_box
    chars: list[dict[str, object]] = []

    for char_box in char_boxes:
        overlap = bbox_overlap_ratio(char_box.bbox, piece_box)
        if overlap < min_overlap:
            continue

        x1, y1, x2, y2 = char_box.bbox
        chars.append(
            {
                "char": char_box.char,
                "word": char_box.word,
                "word_id": char_box.word_id,
                "confidence": char_box.confidence,
                "overlap": round(overlap, 6),
                "bbox": [
                    round(x1 - piece_left, 3),
                    round(y1 - piece_top, 3),
                    round(x2 - piece_left, 3),
                    round(y2 - piece_top, 3),
                ],
                "page_bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            }
        )

    chars.sort(key=lambda item: (item["page_bbox"][1], item["page_bbox"][0]))
    return chars


def text_from_piece_chars(chars: list[dict[str, object]]) -> str:
    words: list[str] = []
    current_word_id: object | None = None
    current_chars: list[str] = []

    for char_info in chars:
        word_id = char_info["word_id"]
        if current_chars and word_id != current_word_id:
            words.append("".join(current_chars))
            current_chars = []
        current_word_id = word_id
        current_chars.append(str(char_info["char"]))

    if current_chars:
        words.append("".join(current_chars))

    return " ".join(words)


def bbox_overlap_ratio(
    bbox: tuple[float, float, float, float],
    piece_box: tuple[int, int, int, int],
) -> float:
    left = max(bbox[0], piece_box[0])
    top = max(bbox[1], piece_box[1])
    right = min(bbox[2], piece_box[2])
    bottom = min(bbox[3], piece_box[3])

    if right <= left or bottom <= top:
        return 0.0

    overlap_area = (right - left) * (bottom - top)
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return overlap_area / bbox_area if bbox_area > 0 else 0.0


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def path_to_json(path: Path) -> str:
    return path.as_posix()


def progress(
    items: Iterable,
    desc: str,
    unit: str,
    total: int | None = None,
    leave: bool = True,
) -> Iterable:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit, total=total, leave=leave)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create piece images and OCR character labels for jigsaw training.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("ocr_dataset"), help="Input OCR dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory for puzzles and manifest.json.")
    parser.add_argument("--rows", type=int, default=3, help="Number of grid rows per page.")
    parser.add_argument("--cols", type=int, default=3, help="Number of grid columns per page.")
    parser.add_argument("--limit", type=int, help="Maximum number of input image/XML pairs to scan.")
    parser.add_argument("--max-images", type=int, help="Maximum number of puzzle images to generate after skips.")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"), help="Piece image format.")
    parser.add_argument(
        "--piece-resolution",
        type=int,
        default=PIECE_RESOLUTION,
        help="Final square piece side length in source image pixels; use -1 to keep native grid cells without square crops.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for floating square crop offsets.")
    parser.add_argument(
        "--min-char-overlap",
        type=float,
        default=0.5,
        help="Minimum fraction of a character box that must overlap a piece.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = create_jigsaw_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        limit=args.limit,
        max_images=args.max_images,
        image_format=args.image_format,
        min_char_overlap=args.min_char_overlap,
        piece_resolution=args.piece_resolution,
        seed=args.seed,
    )
    meta = dataset["meta"]
    print(
        f"Done. Pages: {meta['page_count']}. "
        f"Skipped: {meta['skipped_page_count']}. Pieces: {meta['piece_count']}."
    )
    print(f"Dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
