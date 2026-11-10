"""Count pages and panels in raw dataset directories.

Supported raw layouts:

- MangaZero / COMIX style: a directory containing ``manifest.jsonl`` plus
  ``pages/NNNNNN/annotation.json``. Each manifest line is one page; for
  MangaZero each annotation has a ``frames`` list. For COMIX the annotation
  nests panels at ``metadata.detections.fasterrcnn.panels``; the legacy
  ``frames`` field is also accepted as a fallback.
- Manga109 style: annotations are per-book XML files (``<book><pages><page>``)
  read with ``Manga109/create_ordering.py::parse_book_pages`` so only frames
  with valid bbox attributes count. Two views are reported:

  * ``raw``: every ``<page>`` in the XMLs and every valid ``<frame>``.
  * ``generatable``: the pages that ``create_ordering.py`` can actually use,
    i.e. at least ``--min-frames`` frames plus an existing book image dir and
    page image. ``panels`` is then the frame count of those pages.

Examples:

    python scripts/count_raw_stats.py Mangazero/raw
    python scripts/count_raw_stats.py Data/Manga109 --source manga109
    python scripts/count_raw_stats.py Mangazero/raw Data/Manga109
"""

from __future__ import annotations
# Resolve the project config package when this file is run directly.
import sys as _cli_sys
from pathlib import Path as _CliPath
_cli_root = next((p for p in _CliPath(__file__).resolve().parents if (p / "config" / "cli.py").is_file()), None)
if _cli_root is not None and str(_cli_root) not in _cli_sys.path:
    _cli_sys.path.insert(0, str(_cli_root))
from config.cli import parse_args_with_config

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_manga109_reader():
    """Import the generator's XML reader lazily (it pulls in numpy/PIL)."""
    from Manga109.create_ordering import _find_page_image, parse_book_pages

    return _find_page_image, parse_book_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count pages and panels in raw dataset directories."
    )
    parser.add_argument("raw_dirs", nargs="+", type=Path, help="One or more raw directories.")
    parser.add_argument(
        "--source",
        choices=("auto", "mangazero", "manga109"),
        default="auto",
        help="Override source detection (auto detects by directory layout).",
    )
    parser.add_argument(
        "--pattern",
        default="*.xml",
        help="Glob pattern for Manga109 annotations (default: *.xml).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Minimum frames per page for the Manga109 generatable count (create_ordering default: 3).",
    )
    return parse_args_with_config(parser, script_file=__file__)


def count_manifest_layout(raw_dir: Path) -> tuple[int, int, int]:
    manifest_path = raw_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {raw_dir}")

    pages = 0
    panels = 0
    missing = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            annotation_path = raw_dir / str(entry.get("annotation_path", ""))
            pages += 1
            if not annotation_path.exists():
                missing += 1
                continue
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            panels += count_annotation_panels(annotation)

    return pages, panels, missing


def count_annotation_panels(annotation: dict[str, Any]) -> int:
    frames = annotation.get("frames")
    if isinstance(frames, list):
        return len(frames)

    metadata = annotation.get("metadata", {})
    if isinstance(metadata, dict):
        detections = metadata.get("detections", {})
        if isinstance(detections, dict):
            fasterrcnn = detections.get("fasterrcnn", {})
            if isinstance(fasterrcnn, dict):
                panel_items = fasterrcnn.get("panels") or fasterrcnn.get("panels_list") or []
                if isinstance(panel_items, list):
                    return len(panel_items)
    return 0


def resolve_manga109_dirs(raw_dir: Path) -> tuple[Path, Path]:
    """Return ``(annotations_dir, images_dir)`` for a Manga109 path.

    Accept the dataset root (``<root>/annotations`` + ``<root>/images``) used by
    ``create_ordering.py``. Also accept the annotations directory itself and the
    versioned layout shipped in the release zip
    (``annotations.v2020.12.18/``), so the same call works either way.
    """
    if (raw_dir / "annotations").is_dir():
        return raw_dir / "annotations", raw_dir / "images"

    candidates = [raw_dir, *sorted(raw_dir.glob("annotations*"))]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.xml")):
            images_dir = candidate.parent / "images"
            return candidate, images_dir

    raise FileNotFoundError(
        f"No Manga109 annotations found in {raw_dir} (expected annotations/*.xml)"
    )


def count_manga109_layout(
    raw_dir: Path,
    pattern: str,
    min_frames: int = 3,
) -> dict[str, int]:
    """Count Manga109 pages/panels using the generator's own XML reader.

    ``raw`` counts every page and valid frame. ``generatable`` only counts the
    pages that ``create_ordering.py`` would keep: enough frames plus an existing
    book image directory and page image.
    """
    find_page_image, parse_book_pages = _load_manga109_reader()
    annotations_dir, images_dir = resolve_manga109_dirs(raw_dir)
    annotation_files = sorted(annotations_dir.glob(pattern))
    if not annotation_files:
        raise FileNotFoundError(f"No annotations matching {pattern!r} in {annotations_dir}")

    raw_pages = 0
    raw_panels = 0
    gen_pages = 0
    gen_panels = 0
    books_with_images = 0

    for xml_path in annotation_files:
        try:
            pages = parse_book_pages(xml_path)
        except Exception as exc:
            raise ValueError(f"Failed to parse {xml_path}: {exc}") from exc

        book_name = xml_path.stem
        book_images_dir = images_dir / book_name
        has_images = book_images_dir.is_dir()
        if has_images:
            books_with_images += 1

        for page_info in pages:
            frame_count = len(page_info["frames"])
            raw_pages += 1
            raw_panels += frame_count

            if frame_count < min_frames or not has_images:
                continue
            if find_page_image(book_images_dir, int(page_info["index"])) is None:
                continue
            gen_pages += 1
            gen_panels += frame_count

    return {
        "annotation_files": len(annotation_files),
        "books_with_images": books_with_images,
        "raw_pages": raw_pages,
        "raw_panels": raw_panels,
        "generatable_pages": gen_pages,
        "generatable_panels": gen_panels,
    }


def detect_source(raw_dir: Path) -> str:
    if (raw_dir / "manifest.jsonl").exists():
        return "mangazero"
    if any(raw_dir.glob("*.xml")) or any(raw_dir.glob("annotations*/*.xml")):
        return "manga109"
    raise ValueError(
        f"Cannot detect raw layout in {raw_dir}: expected manifest.jsonl or .xml annotations"
    )


def main() -> None:
    args = parse_args()
    counts: list[tuple[Path, int, int, str, dict[str, int] | None]] = []

    for raw_dir in args.raw_dirs:
        raw_dir = raw_dir.resolve()
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {raw_dir}")

        source = args.source if args.source != "auto" else detect_source(raw_dir)
        if source == "manga109":
            stats = count_manga109_layout(raw_dir, args.pattern, args.min_frames)
            pages = stats["raw_pages"]
            panels = stats["raw_panels"]
        else:
            stats = None
            pages, panels, missing = count_manifest_layout(raw_dir)
            if missing:
                print(
                    f"  warning: {missing} annotations referenced by manifest are missing",
                    file=sys.stderr,
                )

        counts.append((raw_dir, pages, panels, source, stats))
        if stats is None:
            print(f"{raw_dir} [{source}]: pages={pages} panels={panels}")
        else:
            print(
                f"{raw_dir} [{source}]: "
                f"raw_pages={stats['raw_pages']} raw_panels={stats['raw_panels']} "
                f"generatable_pages={stats['generatable_pages']} "
                f"generatable_panels={stats['generatable_panels']} "
                f"(books_with_images={stats['books_with_images']}, "
                f"min_frames={args.min_frames})"
            )

    if len(counts) > 1:
        total_pages = sum(item[1] for item in counts)
        total_panels = sum(item[2] for item in counts)
        print(f"total: pages={total_pages} panels={total_panels}")
        m109 = [item[4] for item in counts if item[4] is not None]
        if m109:
            print(
                "total manga109: "
                f"raw_pages={sum(s['raw_pages'] for s in m109)} "
                f"raw_panels={sum(s['raw_panels'] for s in m109)} "
                f"generatable_pages={sum(s['generatable_pages'] for s in m109)} "
                f"generatable_panels={sum(s['generatable_panels'] for s in m109)}"
            )


if __name__ == "__main__":
    main()
