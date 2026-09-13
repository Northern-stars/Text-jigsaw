"""Stream the COMIX dataset from Hugging Face and build an ordering dataset.

This script combines two steps:

1. Stream raw page images and metadata from the official COMIX Hugging Face
   dataset (``emanuelevivoli/comix-v0_1-pages``).  Each sample uses the
   official schema:

       page["json"]  -> metadata dict (book_id, page_number, page_class,
                        detections.fasterrcnn.panels/characters/faces/textboxes)
       page["jpg"]   -> page image as PIL.Image

   Samples are saved as a local annotation JSON + page image, following the
   same layout as ``Mangazero/download_raw.py``.

2. Reuse ``Mangazero/build_dataset.py`` processing logic to convert the raw
   samples into fixed-count shuffled ordering puzzles (manifest.jsonl,
   sample.json, panels_padded/).

Usage examples:

    # full pipeline: stream + build
    python comics/stream_comix.py --dataset-name emanuelevivoli/comix-v0_1-pages --max-pages 100 --puzzle-num 500

    # stream only (no ordering build)
    python comics/stream_comix.py --dataset-name emanuelevivoli/comix-v0_1-pages --stream-only

    # build only (skip download, use existing raw dir)
    python comics/stream_comix.py --build-only --raw-dir comics/raw_comix
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Mangazero.build_dataset import (
    PanelRecord,
    PreparedPanel,
    clamp_bbox,
    extract_ocr_texts,
    group_by_page,
    handle_ocr_error,
    pad_image,
    parse_source_path,
    recognize_dialog_text,
    set_progress_status,
    write_puzzle_directory,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import requests
except ImportError:
    requests = None


class NetworkDownloadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream COMIX from Hugging Face and build an ordering dataset."
    )
    # HF source
    parser.add_argument(
        "--dataset-name",
        default="emanuelevivoli/comix-v0_1-pages",
        help=(
            "Hugging Face dataset identifier. Defaults to the official "
            "COMIX pages dataset: emanuelevivoli/comix-v0_1-pages."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)

    # Raw download target
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("comics/raw_comix"),
        help="Where to store downloaded raw panel images and annotations.",
    )
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if annotation JSON already exists.",
    )
    parser.add_argument(
        "--skip-network-errors",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-skip-network-errors",
        action="store_false",
        dest="skip_network_errors",
    )

    # Ordering build
    parser.add_argument("--output-dir", type=Path, default=Path("comics/ordering_dataset"))
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--shuffle-per-window", type=int, default=1)
    parser.add_argument("--puzzle-num", type=int, default=500)
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--seed", type=int, default=0)

    # OCR (optional re-ocr of panel crops)
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--ocr-lang", default="ch")
    parser.add_argument("--ocr-version", default=None,
                        choices=("PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"))
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--ocr-mode", default="subprocess",
                        choices=("subprocess", "inline"))
    parser.add_argument("--ocr-timeout", type=float, default=30.0)
    parser.add_argument("--ocr-max-restarts", type=int, default=3)
    parser.add_argument("--ocr-use-angle-cls", action="store_true")
    parser.add_argument("--ocr-enable-pir", action="store_true")
    parser.add_argument("--ocr-enable-mkldnn", action="store_true")

    # Flow control
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Only download raw data; do not build the ordering dataset.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build ordering dataset from existing --raw-dir (skip download).",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.panel_count <= 0:
        raise ValueError("--panel-count must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.puzzle_num < 0:
        raise ValueError("--puzzle-num must be non-negative")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("target size must be positive")


# ---------------------------------------------------------------------------
# HF dataset loading
# ---------------------------------------------------------------------------


def load_hf_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def iter_dataset(dataset) -> Iterable[dict[str, Any]]:
    for sample in dataset:
        yield sample


# ---------------------------------------------------------------------------
# Image download helpers
# ---------------------------------------------------------------------------


def download_image(
    url: str,
    sample_dir: Path,
    stem: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Path:
    suffix = Path(url.split("?", 1)[0]).suffix or ".jpg"
    output_path = sample_dir / f"{stem}{suffix}"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    if requests is None:
        raise ImportError("requests is required to download page images.")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return output_path
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries and retry_sleep > 0:
                import time
                time.sleep(retry_sleep)
    raise NetworkDownloadError(f"failed to download {url}: {last_error}")


def download_pil_image(
    hf_sample: dict[str, Any],
    sample_dir: Path,
    stem: str,
) -> Path:
    """Save the official ``jpg`` PIL image (or a compatible fallback)."""
    image_field = hf_sample.get("jpg")
    if image_field is None:
        image_field = hf_sample.get("image")
    if image_field is None:
        image_field = hf_sample.get("page_image")
    if image_field is None:
        for key, value in hf_sample.items():
            if isinstance(value, Image.Image):
                image_field = value
                break
    if image_field is None:
        raise RuntimeError(f"No image found in sample keys: {list(hf_sample.keys())}")
    img = image_field if isinstance(image_field, Image.Image) else Image.open(image_field)
    ext = ".jpg"
    output_path = sample_dir / f"{stem}{ext}"
    img.convert("RGB").save(output_path)
    return output_path


def download_image_smart(
    url_or_none: str | None,
    hf_sample: dict[str, Any],
    sample_dir: Path,
    stem: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Path:
    if url_or_none:
        return download_image(url_or_none, sample_dir, stem, timeout, retries, retry_sleep)
    return download_pil_image(hf_sample, sample_dir, stem)


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return to_jsonable(item())
    return str(value)


def write_manifest_entry(
    manifest: Any,
    raw_dir: Path,
    sample_index: int,
    annotation_path: Path,
    annotation: dict[str, Any],
) -> None:
    local_pages = annotation.get("_local_pages", {})
    entry = {
        "sample_index": sample_index,
        "annotation_path": relative_to(raw_dir, annotation_path),
        "url1_path": local_pages.get("url1_path", ""),
        "url2_path": local_pages.get("url2_path", ""),
    }
    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Stream + download
# ---------------------------------------------------------------------------


def stream_raw_samples(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir
    pages_dir = raw_dir / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset(args)
    iterable = iter_dataset(dataset)
    if args.max_pages is not None:
        iterable = islice(iterable, args.max_pages)

    total = args.max_pages
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(iterable, total=total, desc="streaming comix", dynamic_ncols=True)
        iterable = progress_bar

    downloaded = 0
    skipped = 0
    sample_keys_seen: list[str] = []
    manifest_path = raw_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, hf_sample in enumerate(iterable):
            sample_dir = pages_dir / f"{index:06d}"
            annotation_path = sample_dir / "annotation.json"

            if annotation_path.exists() and not args.overwrite:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                write_manifest_entry(manifest, raw_dir, index, annotation_path, annotation)
                downloaded += 1
                continue

            sample_dir.mkdir(parents=True, exist_ok=True)
            if not sample_keys_seen:
                sample_keys_seen = list(hf_sample.keys())
                print(f"HF sample keys: {sample_keys_seen}", flush=True)

            # Official COMIX schema: page["json"] is metadata, page["jpg"] is PIL.
            metadata = hf_sample.get("json", {})
            if isinstance(metadata, dict):
                metadata_jsonable = to_jsonable(metadata)
            else:
                metadata_jsonable = to_jsonable(metadata)

            annotation = {
                "_hf_keys": sample_keys_seen,
                "metadata": metadata_jsonable,
                "_local_pages": {},
            }

            image_field = hf_sample.get("jpg")
            if image_field is None:
                image_field = hf_sample.get("image") or hf_sample.get("page_image")
            has_pil = isinstance(image_field, Image.Image)

            url1 = None
            url2 = None
            url1_path = ""
            url2_path = ""

            # Try metadata URL fields if present (not needed for the official
            # pages dataset, but keeps compatibility with older raw layouts).
            if isinstance(metadata_jsonable, dict):
                url1 = metadata_jsonable.get("url1") or metadata_jsonable.get("url") or metadata_jsonable.get("image_url")
                url2 = metadata_jsonable.get("url2")
            if not url1:
                url1 = hf_sample.get("url1") or hf_sample.get("image_url") or hf_sample.get("url")
            if not url2:
                url2 = hf_sample.get("url2")

            try:
                if has_pil:
                    # Official dataset: page["jpg"] is the full page image.
                    path1 = download_pil_image(hf_sample, sample_dir, "page")
                    url1_path = relative_to(raw_dir, path1)
                    annotation["_local_pages"]["url1_path"] = url1_path
                    annotation["_local_pages"]["source"] = "hf_jpg_field"
                elif url1:
                    path1 = download_image_smart(
                        url1, hf_sample, sample_dir, "page_left",
                        args.download_timeout, args.download_retries, args.retry_sleep,
                    )
                    url1_path = relative_to(raw_dir, path1)
                    annotation["_local_pages"]["url1_path"] = url1_path
                    if url1:
                        annotation["_local_pages"]["url1"] = url1
                if url2 and not has_pil:
                    path2 = download_image_smart(
                        url2, hf_sample, sample_dir, "page_right",
                        args.download_timeout, args.download_retries, args.retry_sleep,
                    )
                    url2_path = relative_to(raw_dir, path2)
                    annotation["_local_pages"]["url2_path"] = url2_path
                    annotation["_local_pages"]["url2"] = url2
            except (NetworkDownloadError, ImportError, RuntimeError) as exc:
                if not args.skip_network_errors:
                    raise
                skipped += 1
                print(f"Skip sample {index}: {exc}", flush=True)
                continue

            with annotation_path.open("w", encoding="utf-8") as handle:
                json.dump(annotation, handle, ensure_ascii=False, indent=2)
            write_manifest_entry(manifest, raw_dir, index, annotation_path, annotation)
            manifest.flush()
            downloaded += 1

            if progress_bar is not None:
                progress_bar.set_postfix_str(f"ok={downloaded} skip={skipped}", refresh=True)

    meta = {
        "dataset_name": args.dataset_name,
        "source_split": args.split,
        "raw_dir": str(raw_dir),
        "schema": "comix-v0_1-pages (json/jpg/detections.fasterrcnn)",
        "downloaded_count": downloaded,
        "skipped_count": skipped,
        "manifest_path": "manifest.jsonl",
        "sample_keys": sample_keys_seen,
    }
    with (raw_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Build ordering dataset from local raw dir
#
# Reuses Mangazero/build_dataset.py logic:
#   - load_raw_samples()
#   - group_by_page() -> group panels by source page
#   - write_puzzle_directory() -> write manifest + sample.json + padded panels
# ---------------------------------------------------------------------------


def load_raw_samples_from_dir(raw_dir: Path) -> list[dict[str, Any]]:
    """Read raw annotation JSONs from a manifest, identical to build_dataset."""
    manifest_path = raw_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {raw_dir}")
    samples: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            annotation_path = raw_dir / str(entry["annotation_path"])
            if annotation_path.exists():
                samples.append(json.loads(annotation_path.read_text(encoding="utf-8")))
    return samples


def process_comix_sample(
    sample: dict[str, Any],
    raw_dir: Path,
    target_size: tuple[int, int],
    ocr,
    skip_ocr_errors: bool,
    skip_network_errors: bool,
) -> list[PreparedPanel]:
    """Process one raw COMIX sample into PreparedPanels.

    Official ``emanuelevivoli/comix-v0_1-pages`` layout (as saved by
    ``stream_raw_samples``): ``sample["metadata"]`` holds the HF ``json``
    field (book_id, page_number, page_class, detections.fasterrcnn) and
    ``sample["_local_pages"]["url1_path"]`` points to the saved page image.
    """
    local_pages = sample.get("_local_pages", {})
    source_image_path = local_pages.get("url1_path", "")
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    book_id = str(metadata.get("book_id", "") or "")
    page_number = int(metadata.get("page_number", 0) or 0)
    page_class = str(metadata.get("page_class", "") or "")
    manga_id = book_id or "unknown_book"
    chapter_id = page_class or "default"
    page_index = page_number
    page_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_image_path).strip("_").replace(".", "_")
    if not page_id:
        page_id = f"{manga_id}_{page_index}"

    page_image = None
    if source_image_path:
        left_full = raw_dir / source_image_path
        if left_full.exists():
            page_image = Image.open(left_full).convert("RGB")

    if page_image is None:
        return []

    # Official path: metadata -> detections -> fasterrcnn -> panels/textboxes
    detections_root = metadata.get("detections", {})
    fasterrcnn = detections_root.get("fasterrcnn", {}) if isinstance(detections_root, dict) else {}
    panel_items = fasterrcnn.get("panels", []) if isinstance(fasterrcnn, dict) else []
    all_textboxes = fasterrcnn.get("textboxes", []) if isinstance(fasterrcnn, dict) else []
    all_characters = fasterrcnn.get("characters", []) if isinstance(fasterrcnn, dict) else []

    if not panel_items:
        return []

    page_width, page_height = page_image.size
    records: list[PreparedPanel] = []

    def _extract_bbox(item: Any, pw: int, ph: int) -> tuple[list[int], bool]:
        """Return ([x1, y1, x2, y2], valid) from one detection item."""
        bbox_raw = None
        if isinstance(item, dict):
            bbox_raw = item.get("bbox") or item.get("box") or item.get("region")
        if bbox_raw and isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
            bbox = clamp_bbox([int(v) for v in bbox_raw[:4]], pw, ph)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return bbox, True
        return [0, 0, pw, ph], False

    def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        return inter / area_a

    for panel_index, panel_det in enumerate(panel_items):
        bbox, _ = _extract_bbox(panel_det, page_width, page_height)
        panel_id = f"{page_id}_panel_{panel_index:03d}"
        raw_crop = page_image.crop(bbox).convert("RGB")
        padded, pad = pad_image(raw_crop, target_size)

        # Assign textboxes overlapping this panel; prefer official content.
        dialog_texts: list[str] = []
        panel_dialog_bboxes: list[list[int]] = []
        for tb in all_textboxes:
            tb_bbox, tb_ok = _extract_bbox(tb, page_width, page_height)
            if not tb_ok:
                continue
            if _bbox_overlap_ratio(tb_bbox, bbox) < 0.30:
                continue
            panel_dialog_bboxes.append(tb_bbox)
            content = ""
            if isinstance(tb, dict):
                content = str(tb.get("content") or tb.get("text") or "")
            if not content and ocr is not None:
                try:
                    crop = page_image.crop(tb_bbox).convert("RGB")
                    content = recognize_dialog_text(
                        crop, ocr,
                        skip_errors=skip_ocr_errors,
                        context=f"{panel_id} textbox",
                    )
                except Exception as exc:
                    content = handle_ocr_error(exc, skip_ocr_errors, f"{panel_id} textbox")
            if content.strip():
                dialog_texts.append(content.strip())

        dialog_text = " ".join(dialog_texts).strip()

        # Characters overlapping this panel.
        characters = [
            ch for ch in all_characters
            if isinstance(ch, dict)
            and _bbox_overlap_ratio(_extract_bbox(ch, page_width, page_height)[0], bbox) >= 0.30
        ]
        character_ids = [str(c.get("id", "")) for c in characters if isinstance(c, dict)]
        character_bboxes = [
            [int(v) for v in c.get("bbox", [0, 0, 0, 0])[:4]]
            for c in characters
            if isinstance(c, dict)
        ]
        character_types = [
            int(c.get("type", 0)) for c in characters if isinstance(c, dict)
        ]

        records.append(
            PreparedPanel(
                record=PanelRecord(
                    manga_id=manga_id,
                    chapter_id=chapter_id,
                    source_image_path=source_image_path,
                    page_id=page_id,
                    page_index=page_index,
                    panel_index_in_page=panel_index,
                    global_order=page_index * 10000 + panel_index,
                    bbox=bbox,
                    page_size=[page_width, page_height],
                    padded_path="",
                    raw_size=[raw_crop.size[0], raw_crop.size[1]],
                    padded_size=[target_size[0], target_size[1]],
                    pad=pad,
                    caption=page_class,
                    dialog_bboxes=panel_dialog_bboxes,
                    dialog_texts=dialog_texts,
                    dialog_text=dialog_text,
                    character_ids=character_ids,
                    character_bboxes=character_bboxes,
                    character_types=character_types,
                ),
                padded_image=padded,
            )
        )

    return records


def build_ordering_from_raw(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    rng = random.Random(args.seed)
    target_size = (args.target_width, args.target_height)

    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = None if args.disable_ocr else _build_ocr_engine(args)

    raw_samples = load_raw_samples_from_dir(raw_dir)
    if not raw_samples:
        raise RuntimeError(f"No raw samples found in {raw_dir}")

    print(f"Loaded {len(raw_samples)} raw samples from {raw_dir}", flush=True)

    # Process all raw samples into PreparedPanels, grouped by page
    grouped_panels: dict[str, list[PreparedPanel]] = {}
    for sample in raw_samples:
        panels = process_comix_sample(
            sample=sample,
            raw_dir=raw_dir,
            target_size=target_size,
            ocr=ocr,
            skip_ocr_errors=True,
            skip_network_errors=True,
        )
        if not panels:
            continue
        group_key = f"{panels[0].record.manga_id}/{panels[0].record.chapter_id}"
        page_groups = group_by_page(panels)
        for page_key, page_panels in page_groups.items():
            page_panels.sort(key=lambda p: p.record.panel_index_in_page)
            full_key = f"{group_key}/{page_key}"
            grouped_panels.setdefault(full_key, []).extend(page_panels)

    if not grouped_panels:
        raise RuntimeError("No valid panels found after processing raw samples.")

    print(f"Grouped {sum(len(v) for v in grouped_panels.values())} panels into {len(grouped_panels)} pages", flush=True)

    # Write a manifest + numbered puzzle directories, compatible with
    # MangaZeroPanelOrderingDataset and the MangaZero training scripts.
    manifest_path = output_dir / "manifest.jsonl"
    puzzle_count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for group_key, page_panels in sorted(grouped_panels.items()):
            page_panels.sort(key=lambda p: p.record.panel_index_in_page)
            group_name, page_key = _split_group_key(group_key)
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
                        group_key=group_name,
                        page_key=page_key,
                        window=window,
                        output_dir=output_dir,
                        target_size=target_size,
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
        "source": "comix via HuggingFace",
        "dataset_name": args.dataset_name,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "panel_count": args.panel_count,
        "stride": args.stride,
        "shuffle_per_window": args.shuffle_per_window,
        "puzzle_num": args.puzzle_num,
        "target_panel_size": [args.target_width, args.target_height],
        "group_count": len(grouped_panels),
        "puzzle_count": puzzle_count,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def _split_group_key(group_key: str) -> tuple[str, str]:
    """Split a full group key like ``manga/chapter/page_id`` into (group, page)."""
    parts = group_key.rsplit("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return group_key, group_key


def _build_ocr_engine(args: argparse.Namespace):
    """Build PaddleOCR engine (same as Mangazero/build_dataset.py)."""
    import os
    if not args.ocr_enable_pir:
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
    if not args.ocr_enable_mkldnn:
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_onednn"] = "0"
        os.environ["FLAGS_tracer_onednn_ops_on"] = ""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is required unless --disable-ocr is set."
        ) from exc
    return PaddleOCR(
        lang=args.ocr_lang,
        ocr_version=args.ocr_version,
        device=args.ocr_device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=args.ocr_use_angle_cls,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.build_only:
        print("=== Build only (skip download) ===", flush=True)
        build_ordering_from_raw(args)
        return

    if not args.stream_only:
        print("=== Step 1/2: Stream raw samples from Hugging Face ===", flush=True)
    stream_raw_samples(args)

    if not args.stream_only:
        print("\n=== Step 2/2: Build ordering dataset ===", flush=True)
        build_ordering_from_raw(args)


if __name__ == "__main__":
    main()
