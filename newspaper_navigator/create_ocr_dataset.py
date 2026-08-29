"""Create an OCR image dataset from Chronicling America page references.

The main output layout is:

    output_dir/
      images/<id>.jpg
      manifest.csv

Inputs can be either:
  1. NDNP/Chronicling America manifest files with one .jp2 page path per line.
  2. Individual --page values with either .jp2 paths or Chronicling America URLs.
  3. A local directory that already contains .jpg/.jpeg image files.

Examples:
    python create_ocr_dataset.py --manifest manifests/processed/some_batch.txt --output-dir ocr_dataset --limit 100
    python create_ocr_dataset.py --page batch_x/data/sn00000000/00000000000/1900010101/0001.jp2 --output-dir ocr_dataset
    python create_ocr_dataset.py --input-dir chronam_files --output-dir ocr_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import socket
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


S3_JPEG_SURROGATES_URL = "https://s3.amazonaws.com/ndnp-jpeg-surrogates"
CHRONAM_IIIF_URL = "https://chroniclingamerica.loc.gov/iiif/2"
USER_AGENT = "newspaper-navigator-ocr-dataset/1.0"


@dataclass(frozen=True)
class DownloadPlan:
    source: str
    image_urls: tuple[str, ...]


@dataclass
class DatasetRow:
    item_id: str
    source: str
    image_path: str
    image_url: str = ""
    status: str = "ok"
    error_type: str = ""
    error: str = ""


class DownloadFailure(RuntimeError):
    """Download error with a machine-readable reason."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def read_refs(manifest_paths: Iterable[Path], pages: Iterable[str], limit: int | None) -> list[str]:
    refs: list[str] = []

    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ref = line.strip()
                if ref:
                    refs.append(ref)

    refs.extend(page.strip() for page in pages if page.strip())

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique_refs: list[str] = []
    for ref in refs:
        if ref not in seen:
            unique_refs.append(ref)
            seen.add(ref)
        if limit is not None and len(unique_refs) >= limit:
            break

    return unique_refs


def safe_item_id(source: str) -> str:
    parsed = urlparse(source)
    raw = parsed.path if parsed.scheme else source
    raw = raw.strip("/\\")

    for suffix in (".jp2", ".jpg", ".jpeg"):
        if raw.lower().endswith(suffix):
            raw = raw[: -len(suffix)]
            break

    cleaned = "".join(char if char.isalnum() else "_" for char in raw)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]

    if not cleaned:
        cleaned = "page"
    if len(cleaned) > 90:
        cleaned = cleaned[-90:]

    return f"{cleaned}_{digest}"


def make_download_plan(source: str) -> DownloadPlan:
    if looks_like_chronam_page_url(source):
        base_url = source.rstrip("/") + "/"
        seq_url = base_url.rstrip("/")
        return DownloadPlan(
            source=source,
            image_urls=(
                f"{seq_url}.jpg",
                f"{seq_url}.jp2/full/pct:25/0/default.jpg",
            ),
        )

    page_path = normalize_page_path(source)
    jpg_path = page_path.rsplit(".", 1)[0] + ".jpg"
    encoded_jp2 = quote(page_path, safe="")
    s3_safe_jpg = quote(jpg_path, safe="/")

    return DownloadPlan(
        source=source,
        image_urls=(
            f"{S3_JPEG_SURROGATES_URL}/{s3_safe_jpg}",
            f"{CHRONAM_IIIF_URL}/{encoded_jp2}/full/pct:25/0/default.jpg",
        ),
    )


def looks_like_chronam_page_url(source: str) -> bool:
    parsed = urlparse(source)
    return bool(parsed.scheme and "chroniclingamerica.loc.gov" in parsed.netloc and "/seq-" in parsed.path)


def normalize_page_path(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme:
        path = parsed.path.lstrip("/")
        if path.endswith(".jp2"):
            return path
        raise ValueError(f"URL input is not a Chronicling America page URL or .jp2 path: {source}")

    page_path = source.replace("\\", "/").strip("/")
    if not page_path.endswith(".jp2"):
        raise ValueError(f"manifest/page entries must be .jp2 paths: {source}")
    return page_path


def create_image_xml_pairs(dataset_dir: Path = Path("ocr_dataset")) -> list[tuple[Path, Path]]:
    """Return available newspaper image paths paired with their OCR XML paths.

    The downloaded image names come from manifest source paths such as:
    ``<batch>/data/<lccn>/<reel>/<yyyymmdd><edition>/<page>.jp2``.
    The OCR labels are stored as:
    ``labels/<batch>/<lccn>/<yyyy>/<mm>/<dd>/ed-<edition>/seq-*/ocr.xml``.

    XML ``seq-*`` numbers are not always the same as page file numbers, so this
    function indexes XML files by the ``<fileName>`` value inside each ALTO XML
    file and pairs only images that exist on disk and have a matching XML file.
    """

    manifest_path = dataset_dir / "manifest.csv"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    xml_index = index_ocr_xml_files(labels_dir)
    pairs: list[tuple[Path, Path]] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row.get("source", "")
            image_path_value = row.get("image_path", "")
            if not source or not image_path_value:
                continue

            image_path = Path(image_path_value)
            if not image_path.is_absolute():
                image_path = dataset_dir.parent / image_path
            if not image_path.exists():
                continue

            key = ocr_xml_index_key(source)
            if key is None:
                continue

            xml_path = xml_index.get(key)
            if xml_path is not None:
                pairs.append((image_path, xml_path))

    return pairs


def index_ocr_xml_files(labels_dir: Path) -> dict[tuple[str, str, str, str], Path]:
    xml_index: dict[tuple[str, str, str, str], Path] = {}
    xml_paths = sorted(labels_dir.rglob("ocr.xml"))

    for xml_path in progress(xml_paths, desc="Indexing OCR XML", unit="xml"):
        key = ocr_xml_index_key_from_path(xml_path, labels_dir)
        if key is not None:
            xml_index[key] = xml_path

    return xml_index


def progress(items: Iterable[Path], desc: str, unit: str) -> Iterable[Path]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit)


def ocr_xml_index_key(source: str) -> tuple[str, str, str, str] | None:
    try:
        page_path = normalize_page_path(source)
    except ValueError:
        return None

    parts = Path(page_path).parts
    if len(parts) < 6 or parts[1] != "data":
        return None

    issue = parts[-2]
    page_stem = Path(parts[-1]).stem
    if len(issue) < 10:
        return None


    batch = parts[0]
    lccn = parts[2]
    issue_date = issue[:8]
    edition = str(int(issue[8:])) if issue[8:].isdigit() else issue[8:].lstrip("0")
    file_name = f"{page_stem}.tif"

    return batch, lccn, f"{issue_date}_ed-{edition}", file_name


def ocr_xml_index_key_from_path(xml_path: Path, labels_dir: Path) -> tuple[str, str, str, str] | None:
    try:
        rel_parts = xml_path.relative_to(labels_dir).parts
    except ValueError:
        return None

    if len(rel_parts) < 8:
        return None

    batch, lccn, year, month, day, edition = rel_parts[:6]
    file_name = read_ocr_xml_source_file_name(xml_path)
    if not file_name:
        return None

    return batch, lccn, f"{year}{month}{day}_{edition}", file_name


def read_ocr_xml_source_file_name(xml_path: Path) -> str | None:
    try:
        for _, element in ElementTree.iterparse(xml_path, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] == "fileName":
                return element.text.strip() if element.text else None
            element.clear()
    except ElementTree.ParseError:
        return None
    return None


def download_image(
    source: str,
    images_dir: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> DatasetRow:
    item_id = safe_item_id(source)
    image_path = images_dir / f"{item_id}.jpg"
    row = DatasetRow(item_id=item_id, source=source, image_path=str(image_path))

    try:
        if image_path.exists() and not overwrite:
            try:
                validate_jpeg_file(image_path)
                return row
            except DownloadFailure:
                safe_unlink(image_path)

        plan = make_download_plan(source)
        image_stage = staging_path(image_path)

        image_url = download_first_available(plan.image_urls, image_stage, timeout, retries, overwrite=True)

        image_stage.replace(image_path)
        row.image_url = image_url
        return row
    except DownloadFailure as exc:
        cleanup_paths(staging_path(image_path))
        row.status = "failed"
        row.error_type = exc.reason
        row.error = exc.detail
        return row
    except Exception as exc:  # noqa: BLE001 - this is a CLI batch job; keep going per item.
        cleanup_paths(staging_path(image_path))
        row.status = "failed"
        row.error_type = exc.__class__.__name__
        row.error = str(exc)
        return row


def download_first_available(
    urls: Iterable[str],
    destination: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return ""

    errors: list[DownloadFailure] = []
    for url in urls:
        try:
            download_url(url, destination, timeout=timeout, retries=retries)
            return url
        except DownloadFailure as exc:
            errors.append(exc)

    reason = summarize_failure_reasons(errors)
    detail = " | ".join(f"{error.reason}: {error.detail}" for error in errors)
    raise DownloadFailure(reason, detail)


def download_url(url: str, destination: Path, timeout: int, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    last_error: DownloadFailure | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if destination.suffix in {".jpg", ".jpeg"} and any(
                    token in content_type.lower() for token in ("xml", "html", "text/")
                ):
                    raise DownloadFailure("invalid_content_type", f"{url} returned {content_type} for image")
                expected_size = parse_content_length(response.headers.get("Content-Length"))

                with tmp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)

            actual_size = tmp_path.stat().st_size
            if actual_size == 0:
                raise DownloadFailure("empty_file", f"{url} downloaded an empty file")
            if expected_size is not None and actual_size != expected_size:
                raise DownloadFailure(
                    "incomplete_download",
                    f"{url} expected {expected_size} bytes but downloaded {actual_size} bytes",
                )
            if destination.suffix in {".jpg", ".jpeg"}:
                validate_jpeg_file(tmp_path)

            tmp_path.replace(destination)
            return
        except DownloadFailure as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 5))
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
            last_error = classify_download_exception(exc, url)
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 5))

    if last_error is None:
        raise DownloadFailure("unknown", f"{url} failed for an unknown reason")
    raise last_error


def classify_download_exception(exc: Exception, url: str) -> DownloadFailure:
    if isinstance(exc, HTTPError):
        return DownloadFailure(
            f"http_{exc.code}",
            f"{url} returned HTTP {exc.code} {exc.reason}",
        )

    if isinstance(exc, URLError):
        reason = exc.reason
        reason_text = str(reason)
        if isinstance(reason, socket.gaierror) or "getaddrinfo failed" in reason_text:
            return DownloadFailure("dns_error", f"{url} DNS lookup failed: {reason_text}")
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in reason_text.lower():
            return DownloadFailure("timeout", f"{url} timed out: {reason_text}")
        if isinstance(reason, ConnectionRefusedError):
            return DownloadFailure("connection_refused", f"{url} connection refused: {reason_text}")
        if isinstance(reason, ConnectionResetError):
            return DownloadFailure("connection_reset", f"{url} connection reset: {reason_text}")
        return DownloadFailure("network_error", f"{url} network error: {reason_text}")

    if isinstance(exc, (TimeoutError, socket.timeout)):
        return DownloadFailure("timeout", f"{url} timed out: {exc}")

    if isinstance(exc, OSError):
        return DownloadFailure("io_error", f"{url} I/O error: {exc}")

    return DownloadFailure(exc.__class__.__name__, f"{url} failed: {exc}")


def summarize_failure_reasons(errors: list[DownloadFailure]) -> str:
    if not errors:
        return "download_failed"

    reasons = {error.reason for error in errors}
    if len(reasons) == 1:
        return errors[0].reason
    if any(reason.startswith("http_") for reason in reasons):
        return "http_error"
    if reasons & {"dns_error", "timeout", "connection_refused", "connection_reset", "network_error"}:
        return "network_error"
    return "download_failed"


def parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def validate_jpeg_file(path: Path) -> None:
    try:
        size = path.stat().st_size
        if size < 4:
            raise DownloadFailure("invalid_jpeg", f"{path} is too small to be a valid JPEG")

        with path.open("rb") as handle:
            start = handle.read(2)
            handle.seek(-2, 2)
            end = handle.read(2)

        if start != b"\xff\xd8":
            raise DownloadFailure("invalid_jpeg", f"{path} does not start with a JPEG SOI marker")
        if end != b"\xff\xd9":
            raise DownloadFailure("truncated_jpeg", f"{path} does not end with a JPEG EOI marker")
    except OSError as exc:
        raise DownloadFailure("io_error", f"{path} could not be validated: {exc}") from exc


def staging_path(destination: Path, suffix: str | None = None) -> Path:
    if suffix is None:
        return destination.with_name(f".{destination.name}")
    return destination.with_name(f".{destination.stem}{suffix}")


def cleanup_paths(*paths: Path) -> None:
    for path in paths:
        safe_unlink(path)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def create_from_downloads(args: argparse.Namespace) -> list[DatasetRow]:
    refs = read_refs(args.manifest, args.page, args.limit)
    if not refs:
        raise SystemExit("No input pages found. Pass --manifest, --page, or use --input-dir.")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[DatasetRow] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_image,
                ref,
                images_dir,
                args.timeout,
                args.retries,
                args.overwrite,
            )
            for ref in refs
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if row.status == "failed":
                print(
                    f"[{idx}/{len(futures)}] failed: {row.item_id} "
                    f"({row.error_type}) {row.error}",
                    file=sys.stderr,
                )
            else:
                print(f"[{idx}/{len(futures)}] {row.status}: {row.item_id}", file=sys.stderr)

    return rows


def create_from_local(input_dir: Path, output_dir: Path, overwrite: bool) -> list[DatasetRow]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[DatasetRow] = []
    image_paths = (
        sorted(input_dir.rglob("*.jpg"))
        + sorted(input_dir.rglob("*.jpeg"))
    )

    for image_src in image_paths:
        rel_source = str(image_src.relative_to(input_dir))
        item_id = safe_item_id(rel_source)
        image_dst = images_dir / f"{item_id}{image_src.suffix.lower()}"

        copy_file(image_src, image_dst, overwrite)

        rows.append(
            DatasetRow(
                item_id=item_id,
                source=rel_source,
                image_path=str(image_dst),
            )
        )
        print(f"[{len(rows)}] copied: {item_id}", file=sys.stderr)

    return rows


def copy_file(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_manifest(rows: list[DatasetRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "item_id",
        "source",
        "image_path",
        "image_url",
        "status",
        "error_type",
        "error",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row.item_id,
                    "source": row.source,
                    "image_path": row.image_path,
                    "image_url": row.image_url,
                    "status": row.status,
                    "error_type": row.error_type,
                    "error": row.error,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public JPG page images for an OCR dataset.")
    parser.add_argument("--manifest", type=Path, nargs="*", default=[], help="Text manifest(s), one .jp2 page path per line.")
    parser.add_argument("--page", action="append", default=[], help="Single .jp2 page path or Chronicling America page URL.")
    parser.add_argument("--input-dir", type=Path, help="Local directory containing already downloaded .jpg/.jpeg images.")
    parser.add_argument("--output-dir", type=Path, default=Path("ocr_dataset"), help="Dataset output directory.")
    parser.add_argument("--limit", type=int, help="Maximum number of manifest/page entries to process.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for each URL candidate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_dir:
        rows = create_from_local(args.input_dir, args.output_dir, overwrite=args.overwrite)
    else:
        rows = create_from_downloads(args)

    write_manifest(rows, args.output_dir)

    ok_count = sum(1 for row in rows if row.status == "ok")
    failed_count = sum(1 for row in rows if row.status != "ok")
    print(f"Done. Downloaded images: {ok_count}. Failed items: {failed_count}.")
    print(f"Dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
