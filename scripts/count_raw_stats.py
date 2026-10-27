"""Count pages and panels in raw dataset directories.

Supported raw layouts:

- MangaZero / COMIX style: a directory containing ``manifest.jsonl`` plus
  ``pages/NNNNNN/annotation.json``. Each manifest line is one page; for
  MangaZero each annotation has a ``frames`` list. For COMIX the annotation
  nests panels at ``metadata.detections.fasterrcnn.panels``; the legacy
  ``frames`` field is also accepted as a fallback.
- Manga109 style: annotations are per-book XML files containing ``<page>``
  elements, with each ``<frame>`` element counting as one panel.

Examples:

    python scripts/count_raw_stats.py Mangazero/raw
    python scripts/count_raw_stats.py Data/Manga109/annotations --source manga109
    python scripts/count_raw_stats.py Mangazero/raw Data/Manga109/annotations
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


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
    return parser.parse_args()


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


def count_manga109_layout(raw_dir: Path, pattern: str) -> tuple[int, int]:
    annotation_files = sorted(raw_dir.glob(pattern))
    if not annotation_files:
        raise FileNotFoundError(f"No annotations matching {pattern!r} in {raw_dir}")

    pages = 0
    panels = 0
    for xml_path in annotation_files:
        try:
            tree = ET.parse(xml_path, parser=ET.XMLParser(encoding="utf-8"))
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse {xml_path}: {exc}") from exc
        root = tree.getroot()
        for page in root.iter("page"):
            pages += 1
            panels += sum(1 for _ in page.iter("frame"))

    return pages, panels


def detect_source(raw_dir: Path) -> str:
    if (raw_dir / "manifest.jsonl").exists():
        return "mangazero"
    if any(raw_dir.glob("*.xml")):
        return "manga109"
    raise ValueError(
        f"Cannot detect raw layout in {raw_dir}: expected manifest.jsonl or .xml annotations"
    )


def main() -> None:
    args = parse_args()
    counts: list[tuple[Path, int, int, str]] = []

    for raw_dir in args.raw_dirs:
        raw_dir = raw_dir.resolve()
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {raw_dir}")

        source = args.source if args.source != "auto" else detect_source(raw_dir)
        if source == "manga109":
            pages, panels = count_manga109_layout(raw_dir, args.pattern)
        else:
            pages, panels, missing = count_manifest_layout(raw_dir)
            if missing:
                print(
                    f"  warning: {missing} annotations referenced by manifest are missing",
                    file=sys.stderr,
                )

        counts.append((raw_dir, pages, panels, source))
        print(f"{raw_dir} [{source}]: pages={pages} panels={panels}")

    if len(counts) > 1:
        total_pages = sum(item[1] for item in counts)
        total_panels = sum(item[2] for item in counts)
        print(f"total: pages={total_pages} panels={total_panels}")


if __name__ == "__main__":
    main()