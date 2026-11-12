"""Download bghira/comicstrips-gpt4o-blip3 into a reusable local directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cli import parse_args_with_config


DATASET_NAME = "bghira/comicstrips-gpt4o-blip3"
IMAGE_COLUMNS = ("image", "strip", "comic", "comic_strip", "picture", "img")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/strip/raw"))
    parser.add_argument("--max-strips", type=int, default=None)
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parse_args_with_config(parser, script_file=__file__)


def main() -> None:
    args = parse_args()
    if args.max_strips is not None and args.max_strips <= 0:
        raise ValueError("--max-strips must be positive")
    if args.download_retries < 0 or args.retry_sleep < 0:
        raise ValueError("download retries and retry sleep must be non-negative")

    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(**kwargs)
    iterable = iter(dataset)
    if args.max_strips is not None:
        iterable = islice(iterable, args.max_strips)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    strips_dir = args.raw_dir / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.raw_dir / "manifest.jsonl"
    saved = skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(iterable):
            strip_id = f"{index:08d}"
            strip_dir = strips_dir / strip_id
            annotation_path = strip_dir / "annotation.json"
            image_path = next(strip_dir.glob("image.*"), None) if strip_dir.exists() else None
            if annotation_path.exists() and image_path is not None and not args.overwrite:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            else:
                image_key = resolve_image_column(row, args.image_column)
                image_path = save_image(
                    row[image_key], strip_dir,
                    retries=args.download_retries,
                    retry_sleep=args.retry_sleep,
                )
                annotation = {str(key): to_jsonable(value) for key, value in row.items() if key != image_key}
                annotation["_image_column"] = image_key
                annotation["_image_path"] = image_path.relative_to(args.raw_dir).as_posix()
                annotation_path.write_text(
                    json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                saved += 1

            record = {
                "strip_id": strip_id,
                "image_path": annotation["_image_path"],
                "annotation_path": annotation_path.relative_to(args.raw_dir).as_posix(),
                "source_split": args.split,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

    meta = {
        "dataset_name": args.dataset_name,
        "source_split": args.split,
        "raw_dir": str(args.raw_dir),
        "saved_count": saved,
        "skipped_count": skipped,
        "manifest_path": "manifest.jsonl",
        "image_column": args.image_column,
    }
    (args.raw_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def resolve_image_column(row: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        if requested not in row:
            raise KeyError(f"image column {requested!r} not found; row has {sorted(row)}")
        return requested
    for name in IMAGE_COLUMNS:
        if name in row and is_image_value(row[name]):
            return name
    candidates = [name for name, value in row.items() if is_image_value(value)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not identify a unique image column. "
        f"Set --image-column explicitly; image-like columns: {candidates}"
    )


def is_image_value(value: Any) -> bool:
    return isinstance(value, (Image.Image, bytes, bytearray, memoryview)) or (
        isinstance(value, dict) and any(key in value for key in ("path", "bytes"))
    ) or (isinstance(value, str) and value.lower().startswith(("http://", "https://")))


def save_image(value: Any, strip_dir: Path, retries: int = 3, retry_sleep: float = 2.0) -> Path:
    strip_dir.mkdir(parents=True, exist_ok=True)
    image: Image.Image
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            from io import BytesIO
            image = Image.open(BytesIO(raw)).convert("RGB")
        elif value.get("path"):
            image = Image.open(value["path"]).convert("RGB")
        else:
            raise ValueError("image object must contain bytes or path")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        from io import BytesIO
        image = Image.open(BytesIO(value)).convert("RGB")
    elif isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required for URL-backed image columns") from exc
        suffix = Path(urlparse(value).path).suffix.lower() or ".jpg"
        target = strip_dir / f"image{suffix}"
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = requests.get(value, timeout=30)
                response.raise_for_status()
                target.write_bytes(response.content)
                with Image.open(target) as check:
                    check.verify()
                return target
            except Exception as exc:
                last_error = exc
                if attempt < retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
        raise RuntimeError(f"could not download strip image {value}: {last_error}")
    else:
        raise TypeError(f"unsupported image value: {type(value).__name__}")

    target = strip_dir / "image.png"
    image.save(target)
    return target


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


if __name__ == "__main__":
    main()
