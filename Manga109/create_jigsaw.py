"""Create 3x3 text jigsaw puzzles from Manga109 page images and annotations.

Reads every book's XML annotations (frames, text bboxes), cuts each page into
a 3x3 grid, assigns dialogue text to grid cells via bbox overlap, and writes
the output in the exact format consumed by ``Solver/env/jigsaw_env.py``
(TextJigsawEnv).

Default output layout::

    output/
      manifest.json
      puzzles/<page_id>/
        label.json
        pieces/r000_c000.jpg ...

Example::

    python Manga109/create_jigsaw.py \\
        --manga109-dir Data/Manga109 \\
        --output-dir Data/Manga109/jigsaw_output \\
        --rows 3 --cols 3 \\
        --piece-resolution 384 \\
        --max-pages 500

The script skips pages whose text annotations are fewer than
``--min-texts`` (default 3) or whose grid cells are too small for
the requested piece resolution.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def _bbox_overlap_ratio(
    tb: tuple[float, float, float, float],
    pb: tuple[int, int, int, int],
) -> float:
    """Fraction of text bbox that overlaps the piece box."""
    ix0 = max(tb[0], pb[0])
    iy0 = max(tb[1], pb[1])
    ix1 = min(tb[2], pb[2])
    iy1 = min(tb[3], pb[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    text_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
    return inter / text_area if text_area > 0 else 0.0


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_book_pages(xml_path: Path) -> list[dict[str, Any]]:
    """Parse all page annotations from one Manga109 book XML.

    Returns a list of dicts with keys index, width, height, texts. Each text
    has id/xmin/ymin/xmax/ymax/content. We parse the full document once per
    book because iterparse + element.clear() removes page children before the
    page end event and would lose text nodes.
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
            "texts": [],
        }
        for child in page_el:
            if _local_tag(child.tag) != "text":
                continue
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
# Grid / piece geometry
# ---------------------------------------------------------------------------

def grid_cell_box(
    img_w: int, img_h: int,
    rows: int, cols: int,
    row: int, col: int,
) -> tuple[int, int, int, int]:
    left = round(col * img_w / cols)
    upper = round(row * img_h / rows)
    right = round((col + 1) * img_w / cols)
    lower = round((row + 1) * img_h / rows)
    return left, upper, right, lower


def square_crop_box(
    cell_box: tuple[int, int, int, int],
    piece_res: int,
    rng: random.Random,
    page_id: str,
    row: int, col: int,
) -> tuple[int, int, int, int]:
    left, upper, right, lower = cell_box
    cw, ch = right - left, lower - upper
    if piece_res > cw or piece_res > ch:
        raise ValueError(
            f"Cell {row},{col} on page {page_id} is {cw}x{ch}, "
            f"too small for piece_resolution={piece_res}"
        )
    max_left = right - piece_res
    max_upper = lower - piece_res
    pl = rng.randint(left, max_left)
    pu = rng.randint(upper, max_upper)
    return pl, pu, pl + piece_res, pu + piece_res


def build_piece_specs(
    img_w: int, img_h: int,
    rows: int, cols: int,
    piece_res: int,
    rng: random.Random,
    page_id: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            cell = grid_cell_box(img_w, img_h, rows, cols, row, col)
            if piece_res == -1:
                crop = cell
            else:
                crop = square_crop_box(cell, piece_res, rng, page_id, row, col)
            specs.append({"row": row, "col": col, "grid_box": cell, "piece_box": crop})
    return specs


# ---------------------------------------------------------------------------
# Text -> piece assignment
# ---------------------------------------------------------------------------

def assign_texts_to_pieces(
    texts: list[dict[str, Any]],
    piece_specs: list[dict[str, Any]],
    min_overlap: float = 0.3,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    assignment: dict[tuple[int, int], list[dict[str, Any]]] = {
        (s["row"], s["col"]): [] for s in piece_specs
    }
    for ta in texts:
        tb = (ta["xmin"], ta["ymin"], ta["xmax"], ta["ymax"])
        best_overlap = 0.0
        best_cell: tuple[int, int] | None = None
        for spec in piece_specs:
            ratio = _bbox_overlap_ratio(tb, spec["grid_box"])
            if ratio > best_overlap:
                best_overlap = ratio
                best_cell = (spec["row"], spec["col"])
        if best_cell is not None and best_overlap >= min_overlap:
            assignment[best_cell].append(ta)
    return assignment


def build_piece_text_bundle(
    cell_texts: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    sorted_texts = sorted(cell_texts, key=lambda t: (t["ymin"], t["xmin"]))
    segments: list[str] = [t["content"] for t in sorted_texts]
    full_text = "\n".join(segments)
    chars: list[dict[str, Any]] = []
    for t in sorted_texts:
        chars.append({
            "char": t["content"],
            "word": t["content"],
            "word_id": t["id"],
            "confidence": 1.0,
            "overlap": 1.0,
            "bbox": [
                round(t["xmin"], 3), round(t["ymin"], 3),
                round(t["xmax"], 3), round(t["ymax"], 3),
            ],
            "page_bbox": [
                round(t["xmin"], 3), round(t["ymin"], 3),
                round(t["xmax"], 3), round(t["ymax"], 3),
            ],
        })
    return full_text, segments, chars


# ---------------------------------------------------------------------------
# Image file discovery
# ---------------------------------------------------------------------------

def _find_page_image(book_images_dir: Path, page_index: int) -> Path | None:
    for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
        for fmt in [f"{page_index:03d}", str(page_index)]:
            candidate = book_images_dir / f"{fmt}.{ext}"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Main dataset creation
# ---------------------------------------------------------------------------

def create_manga109_jigsaw(
    manga109_dir: Path,
    output_dir: Path,
    rows: int = 3,
    cols: int = 3,
    piece_resolution: int = 384,
    max_pages: int | None = None,
    min_texts: int = 3,
    image_format: str = "jpg",
    min_text_overlap: float = 0.3,
    books: list[str] | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Cut Manga109 pages into labeled jigsaw puzzles.

    Output format is identical to the newspaper navigator jigsaw so that
    ``TextJigsawEnv`` can consume it without changes.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if piece_resolution == 0 or piece_resolution < -1:
        raise ValueError("piece_resolution must be a positive int or -1")

    annotations_dir = manga109_dir / "annotations"
    images_dir = manga109_dir / "images"
    puzzles_dir = output_dir / "puzzles"
    puzzles_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # discover books
    if books:
        book_names = books
    else:
        book_names = sorted(p.stem for p in annotations_dir.glob("*.xml"))

    puzzle_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_piece_count = 0
    pages_iterated = 0

    book_iter = book_names
    if tqdm is not None:
        book_iter = tqdm(book_names, desc="books", unit="book", dynamic_ncols=True)

    for book_name in book_iter:
        xml_path = annotations_dir / f"{book_name}.xml"
        if not xml_path.exists():
            skipped.append({"book": book_name, "reason": "annotation XML not found"})
            continue

        book_images_dir = images_dir / book_name
        if not book_images_dir.exists():
            skipped.append({"book": book_name, "reason": "image directory not found"})
            continue

        pages = parse_book_pages(xml_path)
        for page_info in pages:
            if max_pages is not None and pages_iterated >= max_pages:
                break

            page_index = int(page_info["index"])
            page_id = f"{book_name}_p{page_index:04d}"

            image_path = _find_page_image(book_images_dir, page_index)
            if image_path is None:
                skipped.append({"page_id": page_id, "reason": "image not found"})
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

            texts = list(page_info["texts"])
            if len(texts) < min_texts:
                skipped.append({
                    "page_id": page_id,
                    "reason": f"only {len(texts)} texts (min {min_texts})",
                })
                continue

            try:
                piece_specs = build_piece_specs(
                    img_w, img_h, rows, cols,
                    piece_resolution, rng, page_id,
                )
            except ValueError as exc:
                skipped.append({"page_id": page_id, "reason": str(exc)})
                continue

            assignment = assign_texts_to_pieces(texts, piece_specs, min_text_overlap)

            nonempty = sum(1 for v in assignment.values() if v)
            if nonempty < rows * cols // 2:
                skipped.append({
                    "page_id": page_id,
                    "reason": f"only {nonempty}/{rows*cols} cells have text",
                })
                continue

            # write pieces
            puzzle_dir = puzzles_dir / page_id
            pieces_dir = puzzle_dir / "pieces"
            pieces_dir.mkdir(parents=True, exist_ok=True)

            piece_labels: list[dict[str, Any]] = []
            for spec in piece_specs:
                piece_path = pieces_dir / f"r{spec['row']:03d}_c{spec['col']:03d}.{image_format}"
                crop = img.crop(spec["piece_box"])
                crop.save(piece_path)

                cell_texts_list = assignment.get((spec["row"], spec["col"]), [])
                full_text, segments, chars = build_piece_text_bundle(cell_texts_list)

                piece_labels.append({
                    "piece_id": piece_path.stem,
                    "piece_path": str(piece_path.relative_to(output_dir)),
                    "page_id": page_id,
                    "page_index": page_index,
                    "book": book_name,
                    "row": spec["row"],
                    "col": spec["col"],
                    "grid_bbox": list(spec["grid_box"]),
                    "bbox": list(spec["piece_box"]),
                    "width": spec["piece_box"][2] - spec["piece_box"][0],
                    "height": spec["piece_box"][3] - spec["piece_box"][1],
                    "text": full_text,
                    "segments": segments,
                    "chars": chars,
                })

            label = {
                "meta": {
                    "dataset": "manga109",
                    "source_dir": str(manga109_dir),
                    "output_dir": str(output_dir),
                    "puzzle_dir": str(puzzle_dir.relative_to(output_dir)),
                    "rows": rows,
                    "cols": cols,
                    "piece_count": len(piece_labels),
                    "piece_resolution": piece_resolution,
                    "min_text_overlap": min_text_overlap,
                },
                "page": {
                    "page_id": page_id,
                    "book": book_name,
                    "page_index": page_index,
                    "image_path": str(image_path.relative_to(manga109_dir)),
                    "xml_path": str(xml_path.relative_to(manga109_dir)),
                    "width": img_w,
                    "height": img_h,
                    "text_count": len(texts),
                },
                "pieces": piece_labels,
            }
            label_path = puzzle_dir / "label.json"
            with label_path.open("w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            total_piece_count += len(piece_labels)
            pages_iterated += 1
            puzzle_summaries.append({
                "page_index": len(puzzle_summaries),
                "page_id": page_id,
                "book": book_name,
                "puzzle_dir": str(puzzle_dir.relative_to(output_dir)),
                "label_path": str(label_path.relative_to(output_dir)),
                "piece_count": len(piece_labels),
                "text_count": len(texts),
            })

            if max_pages is not None and pages_iterated >= max_pages:
                break

    dataset = {
        "meta": {
            "dataset": "manga109",
            "manga109_dir": str(manga109_dir),
            "output_dir": str(output_dir),
            "rows": rows,
            "cols": cols,
            "piece_resolution": piece_resolution,
            "min_texts": min_texts,
            "min_text_overlap": min_text_overlap,
            "piece_count": total_piece_count,
            "page_count": len(puzzle_summaries),
            "skipped_count": len(skipped),
            "books_scanned": len(book_names),
            "label_note": (
                "Each puzzle has its own label.json under "
                "output_dir/puzzles/<page_id>/"
            ),
        },
        "puzzles": puzzle_summaries,
        "skipped_pages": skipped,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create 3x3 text jigsaw puzzles from Manga109."
    )
    p.add_argument(
        "--manga109-dir", type=Path, default=Path("Data/Manga109"),
        help="Root directory of Manga109 dataset (containing images/ and annotations/).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("Data/Manga109/jigsaw_output"),
        help="Output directory for generated puzzles and manifest.json.",
    )
    p.add_argument("--rows", type=int, default=3, help="Grid rows per page.")
    p.add_argument("--cols", type=int, default=3, help="Grid columns per page.")
    p.add_argument(
        "--piece-resolution", type=int, default=384,
        help="Square piece side length in source pixels. Use -1 for full grid cells.",
    )
    p.add_argument("--max-pages", type=int, help="Maximum pages to generate.")
    p.add_argument(
        "--min-texts", type=int, default=3,
        help="Minimum text annotations on a page to keep it.",
    )
    p.add_argument(
        "--min-text-overlap", type=float, default=0.3,
        help="Minimum fraction of a text bbox overlapping a cell to assign it.",
    )
    p.add_argument(
        "--image-format", default="jpg", choices=("jpg", "png"),
        help="Output image format for piece images.",
    )
    p.add_argument("--books", nargs="*", help="Process only these book titles.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = create_manga109_jigsaw(
        manga109_dir=args.manga109_dir,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        piece_resolution=args.piece_resolution,
        max_pages=args.max_pages,
        min_texts=args.min_texts,
        image_format=args.image_format,
        min_text_overlap=args.min_text_overlap,
        books=args.books,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    meta = dataset["meta"]
    print(
        f"Done. Books scanned: {meta['books_scanned']}. "
        f"Pages generated: {meta['page_count']}. "
        f"Pages skipped: {meta['skipped_count']}. "
        f"Total pieces: {meta['piece_count']}."
    )
    print(f"Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()
