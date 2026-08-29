"""Download raw MangaZero pages and annotations.

This stage is intentionally network-only: it reads Hugging Face MangaZero
samples, downloads the two page images referenced by each sample, and writes a
local annotation JSON that points to those downloaded images. Later processing
should use build_dataset.py and should not need Hugging Face/network access.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

DATASET_NAME = "jianzongwu/MangaZero"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class NetworkDownloadError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw MangaZero pages and annotations.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/Mangazero/raw"))
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-network-errors", action="store_true", default=True)
    parser.add_argument("--no-skip-network-errors", action="store_false", dest="skip_network_errors")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    raw_dir = args.raw_dir
    pages_dir = raw_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_mangazero_dataset(args)
    iterable = iter_dataset(dataset)
    if args.max_pages is not None:
        iterable = islice(iterable, args.max_pages)
    if tqdm is not None:
        iterable = tqdm(iterable, total=args.max_pages, desc="downloading raw pages", dynamic_ncols=True)

    downloaded = 0
    skipped = 0
    manifest_path = raw_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, sample in enumerate(iterable):
            progress = iterable if tqdm is not None else None
            sample_dir = pages_dir / f"{index:06d}"
            annotation_path = sample_dir / "annotation.json"
            if annotation_path.exists() and not args.overwrite:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                write_manifest_entry(manifest, raw_dir, index, annotation_path, annotation)
                downloaded += 1
                continue

            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                set_progress_status(progress, f"下载样本 {index:06d} 左页")
                left_path = download_image(
                    str(sample["meta"]["url1"]),
                    sample_dir,
                    "page_left",
                    timeout=args.download_timeout,
                    retries=args.download_retries,
                    retry_sleep=args.retry_sleep,
                )
                set_progress_status(progress, f"下载样本 {index:06d} 右页")
                right_path = download_image(
                    str(sample["meta"]["url2"]),
                    sample_dir,
                    "page_right",
                    timeout=args.download_timeout,
                    retries=args.download_retries,
                    retry_sleep=args.retry_sleep,
                )
            except NetworkDownloadError as exc:
                if not args.skip_network_errors:
                    raise
                skipped += 1
                print(f"Network error for sample {index:06d}; skip: {exc}", flush=True)
                continue

            annotation = to_jsonable(sample)
            annotation["_local_pages"] = {
                "url1_path": relative_to(raw_dir, left_path),
                "url2_path": relative_to(raw_dir, right_path),
                "url1": str(sample["meta"]["url1"]),
                "url2": str(sample["meta"]["url2"]),
            }
            with annotation_path.open("w", encoding="utf-8") as handle:
                json.dump(annotation, handle, ensure_ascii=False, indent=2)
            write_manifest_entry(manifest, raw_dir, index, annotation_path, annotation)
            manifest.flush()
            downloaded += 1
            set_progress_status(progress, f"已保存 {downloaded}")

    meta = {
        "dataset_name": args.dataset_name,
        "source_split": args.split,
        "raw_dir": str(raw_dir),
        "downloaded_count": downloaded,
        "skipped_count": skipped,
        "manifest_path": "manifest.jsonl",
    }
    with (raw_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    if args.download_timeout <= 0:
        raise ValueError("--download-timeout must be positive")
    if args.download_retries < 0:
        raise ValueError("--download-retries must be non-negative")
    if args.retry_sleep < 0:
        raise ValueError("--retry-sleep must be non-negative")


def load_mangazero_dataset(args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def iter_dataset(dataset: Any) -> Iterable[dict[str, Any]]:
    for sample in dataset:
        yield sample


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
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required to download MangaZero page images") from exc

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
                time.sleep(retry_sleep)
    raise NetworkDownloadError(f"failed to download {url}: {last_error}")


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
        "image_path": str(annotation.get("image_path", "")),
        "annotation_path": relative_to(raw_dir, annotation_path),
        "url1_path": local_pages.get("url1_path", ""),
        "url2_path": local_pages.get("url2_path", ""),
    }
    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


def relative_to(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def set_progress_status(progress: Any | None, status: str) -> None:
    if progress is None:
        return
    try:
        progress.set_postfix_str(f"step={status}", refresh=True)
    except Exception:
        return


if __name__ == "__main__":
    main()
