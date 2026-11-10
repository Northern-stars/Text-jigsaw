"""Create an RQ3 visual-only manifest by masking dialog boxes in panel images.

The source samples must contain ``dialog_bboxes`` in page coordinates and the
panel geometry fields emitted by ``Mangazero/build_dataset.py``. The output is
self-contained, so later evaluation does not depend on mutable source images.
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
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--fill", default="black", help="PIL fill colour for text masks")
    return parse_args_with_config(parser, script_file=__file__)


def resolve_path(raw: str, source_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else source_dir / path


def mask_panel(panel: dict[str, Any], source_dir: Path, output_path: Path, fill: str) -> None:
    source = resolve_path(str(panel["padded_path"]), source_dir)
    image = Image.open(source).convert("RGB")
    panel_bbox = [float(v) for v in panel.get("bbox", [0, 0, image.width, image.height])]
    raw_w, raw_h = [float(v) for v in panel.get("raw_size", [panel_bbox[2] - panel_bbox[0], panel_bbox[3] - panel_bbox[1]])]
    padded_w, padded_h = [float(v) for v in panel.get("padded_size", [image.width, image.height])]
    pad_left, pad_top = [float(v) for v in panel.get("pad", [0, 0])[:2]]
    scale_x = max(1e-6, (padded_w - 2 * pad_left) / raw_w)
    scale_y = max(1e-6, (padded_h - 2 * pad_top) / raw_h)
    draw = ImageDraw.Draw(image)
    for bbox in panel.get("dialog_bboxes", []):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        local = (
            pad_left + (x1 - panel_bbox[0]) * scale_x,
            pad_top + (y1 - panel_bbox[1]) * scale_y,
            pad_left + (x2 - panel_bbox[0]) * scale_x,
            pad_top + (y2 - panel_bbox[1]) * scale_y,
        )
        draw.rectangle(local, fill=fill)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    output_dir = args.output_dir
    output_manifest = args.output_manifest or (output_dir / "manifest.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        source_dir = Path(str(entry.get("source_dir", args.manifest.parent)))
        source_sample_path = source_dir / str(entry["sample_path"])
        sample = json.loads(source_sample_path.read_text(encoding="utf-8"))
        sample_id = str(sample.get("sequence_id", entry.get("sequence_id", len(rows))))
        digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
        for panel_index, panel in enumerate(sample.get("panels", [])):
            image_name = f"{digest}_{panel_index:03d}.png"
            mask_panel(panel, source_dir, output_dir / "images" / image_name, args.fill)
            panel["padded_path"] = f"images/{image_name}"
        sample_path = output_dir / "samples" / f"{digest}.json"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        new_entry = dict(entry)
        new_entry["source_dir"] = str(output_dir.resolve())
        new_entry["sample_path"] = str(sample_path.relative_to(output_dir)).replace("\\", "/")
        new_entry["visual_only_mask"] = "dialog_bboxes_black_fill"
        rows.append(new_entry)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {output_manifest}")


if __name__ == "__main__":
    main()
