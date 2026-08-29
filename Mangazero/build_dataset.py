"""Build MangaZero ordering puzzles from locally downloaded raw samples."""

from __future__ import annotations
import torch
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from queue import Empty, Full
from typing import Any, Iterable

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class PanelRecord:
    manga_id: str
    chapter_id: str
    source_image_path: str
    page_id: str
    page_index: int
    panel_index_in_page: int
    global_order: int
    bbox: list[int]
    page_size: list[int]
    padded_path: str
    raw_size: list[int]
    padded_size: list[int]
    pad: list[int]
    caption: str
    dialog_bboxes: list[list[int]]
    dialog_texts: list[str]
    dialog_text: str
    character_ids: list[str]
    character_bboxes: list[list[int]]
    character_types: list[int]

    @property
    def panel_id(self) -> str:
        return f"{self.page_id}_panel_{self.panel_index_in_page:03d}"

    def to_json(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["panel_id"] = self.panel_id
        return data


@dataclass
class PreparedPanel:
    record: PanelRecord
    padded_image: Image.Image


class NetworkImageError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MangaZero puzzles from raw local samples.")
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/Mangazero/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/Mangazero/ordering_dataset"))
    parser.add_argument("--panel-count", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--shuffle-per-window", type=int, default=1)
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--puzzle-num", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--skip-network-errors", action="store_true", default=True)
    parser.add_argument("--no-skip-network-errors", action="store_false", dest="skip_network_errors")
    parser.add_argument("--skip-ocr-errors", action="store_true", default=True)
    parser.add_argument("--no-skip-ocr-errors", action="store_false", dest="skip_ocr_errors")
    parser.add_argument("--ocr-lang", default="ch")
    parser.add_argument("--ocr-version", default="PP-OCRv6", choices=("PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"))
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ocr-mode", default="subprocess", choices=("subprocess", "inline"))
    parser.add_argument("--ocr-timeout", type=float, default=30.0)
    parser.add_argument("--ocr-max-restarts", type=int, default=3)
    parser.add_argument("--ocr-use-angle-cls", action="store_true")
    parser.add_argument("--ocr-enable-pir", action="store_true")
    parser.add_argument("--ocr-enable-mkldnn", action="store_true")
    parser.add_argument("--debug-ocr", action="store_true", help="Print OCR worker lifecycle logs.")
    parser.add_argument("--ocr-debug-probe", action="store_true", help="Only initialize OCR and run one small probe image.")
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    output_dir = args.output_dir
    raw_dir = args.raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = None if args.disable_ocr else build_ocr_engine(args)
    try:
        if args.ocr_debug_probe:
            run_ocr_debug_probe(ocr)
            return
        raw_samples = load_raw_samples(raw_dir)
        dataset_iterable = iter(raw_samples)
        if args.max_samples is not None:
            dataset_iterable = islice(dataset_iterable, args.max_samples)
        if tqdm is not None:
            dataset_iterable = tqdm(dataset_iterable, total=args.max_samples, desc="processing pages", dynamic_ncols=True)

        puzzle_count = 0
        page_group_count = 0
        panel_count_total = 0
        manifest_path = output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_handle:
            for sample in dataset_iterable:
                progress = dataset_iterable if tqdm is not None else None
                prepared_panels = process_sample(
                    sample=sample,
                    raw_dir=raw_dir,
                    target_size=(args.target_width, args.target_height),
                    ocr=ocr,
                    skip_ocr_errors=args.skip_ocr_errors,
                    skip_network_errors=args.skip_network_errors,
                    progress=progress,
                )
                if not prepared_panels:
                    continue
                group_key = f"{prepared_panels[0].record.manga_id}/{prepared_panels[0].record.chapter_id}"
                page_groups = group_by_page(prepared_panels)
                page_group_count += len(page_groups)
                panel_count_total += len(prepared_panels)
                for page_key, page_panels in page_groups.items():
                    page_panels.sort(key=lambda panel: panel.record.panel_index_in_page)
                    for start in range(0, len(page_panels), args.stride):
                        if puzzle_count >= args.puzzle_num:
                            break
                        if start + args.panel_count > len(page_panels):
                            continue
                        window = page_panels[start:start + args.panel_count]
                        for _ in range(args.shuffle_per_window):
                            if puzzle_count >= args.puzzle_num:
                                break
                            set_progress_status(progress, f"写入puzzle {puzzle_count + 1}/{args.puzzle_num}")
                            sample_json = write_puzzle_directory(
                                puzzle_index=puzzle_count,
                                group_key=group_key,
                                page_key=page_key,
                                window=window,
                                output_dir=output_dir,
                                target_size=(args.target_width, args.target_height),
                                image_format=args.image_format,
                                rng=rng,
                            )
                            manifest_handle.write(json.dumps({
                                "puzzle_index": puzzle_count,
                                "sequence_id": sample_json["sequence_id"],
                                "sample_path": f"{puzzle_count:06d}/sample.json",
                            }, ensure_ascii=False) + "\n")
                            manifest_handle.flush()
                            puzzle_count += 1
                    if puzzle_count >= args.puzzle_num:
                        break
                set_progress_status(progress, "等待下一页")
                if puzzle_count >= args.puzzle_num:
                    break

        meta = {
            "raw_dir": str(raw_dir),
            "source": "local raw MangaZero samples",
            "panel_count": args.panel_count,
            "stride": args.stride,
            "shuffle_per_window": args.shuffle_per_window,
            "puzzle_num": args.puzzle_num,
            "target_panel_size": [args.target_width, args.target_height],
            "ocr_enabled": not args.disable_ocr,
            "ocr_mode": None if args.disable_ocr else args.ocr_mode,
            "ocr_lang": args.ocr_lang if not args.disable_ocr else None,
            "ocr_version": args.ocr_version if not args.disable_ocr else None,
            "ocr_device": args.ocr_device if not args.disable_ocr else None,
            "group_count": page_group_count,
            "panel_count_total": panel_count_total,
            "puzzle_count": puzzle_count,
        }
        with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    finally:
        close = getattr(ocr, "close", None)
        if callable(close):
            close()


def validate_args(args: argparse.Namespace) -> None:
    if args.panel_count <= 1:
        raise ValueError("--panel-count must be greater than 1")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.shuffle_per_window <= 0:
        raise ValueError("--shuffle-per-window must be positive")
    if args.puzzle_num <= 0:
        raise ValueError("--puzzle-num must be positive")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")


def debug_log(enabled: bool, message: str) -> None:
    if not enabled:
        return
    pid = os.getpid()
    print(f"[ocr-debug pid={pid} {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def log_paddle_device_info(args: argparse.Namespace) -> None:
    if not getattr(args, "debug_ocr", False):
        return
    try:
        import paddle

        compiled_cuda = paddle.is_compiled_with_cuda()
        cuda_count = paddle.device.cuda.device_count() if compiled_cuda else 0
        debug_log(True, f"paddle cuda_compiled={compiled_cuda} cuda_count={cuda_count}")
    except Exception as exc:
        debug_log(True, f"paddle device info failed {type(exc).__name__}: {exc}")


def run_ocr_debug_probe(ocr: Any | None) -> None:
    if ocr is None:
        print("OCR is disabled; probe skipped.")
        return
    image = Image.new("RGB", (128, 48), "white")
    print("Running OCR debug probe on a small blank image...", flush=True)
    text = ocr.recognize(image, context="ocr_debug_probe")
    print(f"OCR debug probe finished. text={text!r}", flush=True)


class InlineOCREngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.debug = args.debug_ocr
        debug_log(self.debug, f"inline OCR init start device={args.ocr_device}")
        self.ocr = build_paddle_ocr(args)
        debug_log(self.debug, "inline OCR init done")

    def recognize(self, image: Image.Image, context: str = "") -> str:
        debug_log(self.debug, f"inline OCR predict start context={context} size={image.size}")
        return run_dialog_ocr_inline(image, self.ocr)


class SubprocessOCREngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timeout = args.ocr_timeout
        self.max_restarts = args.ocr_max_restarts
        self.debug = args.debug_ocr
        self.restart_count = 0
        self.ctx = mp.get_context("spawn")
        self.request_queue: mp.Queue = self.ctx.Queue(maxsize=1)
        self.response_queue: mp.Queue = self.ctx.Queue(maxsize=1)
        self.process: mp.Process | None = None
        self.start_worker()

    def start_worker(self) -> None:
        debug_log(self.debug, "OCR worker start")
        self.process = self.ctx.Process(
            target=ocr_worker_main,
            args=(ocr_worker_config(self.args), self.request_queue, self.response_queue),
            daemon=True,
        )
        self.process.start()
        debug_log(self.debug, f"OCR worker pid={self.process.pid}")

    def recognize(self, image: Image.Image, context: str = "") -> str:
        for _ in range(self.max_restarts + 1):
            if self.process is None or not self.process.is_alive():
                exitcode = None if self.process is None else self.process.exitcode
                debug_log(self.debug, f"OCR worker not alive before request context={context} exitcode={exitcode}")
                self.restart_worker(context)
            debug_log(self.debug, f"OCR request send context={context} size={image.size}")
            try:
                self.request_queue.put((context, image_to_png_bytes(image)), timeout=5)
            except Full:
                debug_log(self.debug, f"OCR request queue full context={context}; restart worker")
                self.restart_worker(context)
                continue
            try:
                debug_log(self.debug, f"OCR wait response context={context} timeout={self.timeout}")
                status, payload = self.response_queue.get(timeout=self.timeout)
            except Empty:
                exitcode = None if self.process is None else self.process.exitcode
                debug_log(self.debug, f"OCR timeout context={context} worker_exitcode={exitcode}")
                self.restart_worker(context)
                continue
            if status == "ok":
                self.restart_count = 0
                debug_log(self.debug, f"OCR ok context={context} text_len={len(str(payload))}")
                return str(payload)
            debug_log(self.debug, f"OCR error context={context} payload={payload}")
            raise RuntimeError(f"OCR worker failed for {context}: {payload}")
        raise RuntimeError(f"OCR worker repeatedly failed for {context}")

    def restart_worker(self, context: str = "") -> None:
        debug_log(self.debug, f"OCR worker restart context={context}")
        self.close()
        self.restart_count += 1
        if self.restart_count > self.max_restarts:
            raise RuntimeError(f"OCR worker exceeded restart limit near {context}")
        self.request_queue = self.ctx.Queue(maxsize=1)
        self.response_queue = self.ctx.Queue(maxsize=1)
        self.start_worker()

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            try:
                self.request_queue.put_nowait(None)
            except Exception:
                pass
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.process = None


def build_ocr_engine(args: argparse.Namespace) -> InlineOCREngine | SubprocessOCREngine:
    if args.ocr_mode == "inline":
        return InlineOCREngine(args)
    return SubprocessOCREngine(args)


def ocr_worker_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ocr_lang": args.ocr_lang,
        "ocr_version": args.ocr_version,
        "ocr_device": args.ocr_device,
        "ocr_use_angle_cls": args.ocr_use_angle_cls,
        "ocr_enable_pir": args.ocr_enable_pir,
        "ocr_enable_mkldnn": args.ocr_enable_mkldnn,
        "debug_ocr": args.debug_ocr,
    }


def ocr_worker_main(config: dict[str, Any], request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    args = argparse.Namespace(**config)
    debug = bool(config.get("debug_ocr", False))
    try:
        debug_log(debug, f"worker init start device={args.ocr_device}")
        ocr = build_paddle_ocr(args)
        debug_log(debug, "worker init done")
    except Exception as exc:
        debug_log(debug, f"worker init failed {type(exc).__name__}: {exc}")
        response_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return

    while True:
        debug_log(debug, "worker wait request")
        item = request_queue.get()
        if item is None:
            debug_log(debug, "worker shutdown")
            break
        context, image_bytes = item
        try:
            debug_log(debug, f"worker received context={context} bytes={len(image_bytes)}")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            debug_log(debug, f"worker predict start context={context} size={image.size}")
            text = run_dialog_ocr_inline(image, ocr)
            debug_log(debug, f"worker predict done context={context} text_len={len(text)}")
            response_queue.put(("ok", text))
        except Exception as exc:
            debug_log(debug, f"worker predict failed context={context} {type(exc).__name__}: {exc}")
            response_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def build_paddle_ocr(args: argparse.Namespace):
    debug_log(getattr(args, "debug_ocr", False), f"build PaddleOCR device={args.ocr_device}")
    if not args.ocr_enable_pir:
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
    if not args.ocr_enable_mkldnn:
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_onednn"] = "0"
        os.environ["FLAGS_tracer_onednn_ops_on"] = ""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is required unless --disable-ocr is set. "
            "Install paddleocr and paddlepaddle before generating dialog text."
        ) from exc
    patch_missing_paddle_optimization_level()
    log_paddle_device_info(args)
    return PaddleOCR(
        lang=args.ocr_lang,
        ocr_version=args.ocr_version,
        device=args.ocr_device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=args.ocr_use_angle_cls,
    )


def patch_missing_paddle_optimization_level() -> None:
    try:
        import paddle

        analysis_config = paddle.base.libpaddle.AnalysisConfig
    except Exception:
        return
    if hasattr(analysis_config, "set_optimization_level"):
        return

    def set_optimization_level(self, level: int) -> None:
        return None

    try:
        setattr(analysis_config, "set_optimization_level", set_optimization_level)
    except (AttributeError, TypeError):
        return


def load_raw_samples(raw_dir: Path) -> list[dict[str, Any]]:
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
            samples.append(json.loads(annotation_path.read_text(encoding="utf-8")))
    return samples


def process_sample(
    sample: dict[str, Any],
    raw_dir: Path,
    target_size: tuple[int, int],
    ocr: Any | None,
    skip_ocr_errors: bool,
    skip_network_errors: bool,
    progress: Any | None = None,
) -> list[PreparedPanel]:
    source_image_path = str(sample.get("image_path", ""))
    manga_id, chapter_id, page_index = parse_source_path(source_image_path)
    page_id = make_page_id(source_image_path)
    try:
        set_progress_status(progress, "读取本地页面")
        page_image = load_page_image_from_raw(sample, raw_dir)
    except NetworkImageError as exc:
        return handle_network_error(exc, skip_network_errors, source_image_path)
    page_width, page_height = page_image.size

    records = []
    frames = sample.get("frames", [])
    for panel_index, frame in enumerate(frames):
        set_progress_status(progress, f"裁剪panel {panel_index + 1}/{len(frames)}")
        bbox = clamp_bbox([int(value) for value in frame["bbox"]], page_width, page_height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        panel_id = f"{page_id}_panel_{panel_index:03d}"
        raw_crop = page_image.crop(bbox).convert("RGB")
        padded, pad = pad_image(raw_crop, target_size)

        dialog_bboxes = [
            clamp_bbox([int(value) for value in dialog["bbox"]], page_width, page_height)
            for dialog in frame.get("dialogs", [])
        ]
        dialog_texts = [
            recognize_dialog_text(
                page_image.crop(dialog_bbox).convert("RGB"),
                ocr,
                skip_errors=skip_ocr_errors,
                context=f"{panel_id} dialog {dialog_index}",
                progress=progress,
                progress_status=f"OCR panel {panel_index + 1}/{len(frames)} dialog {dialog_index + 1}/{len(dialog_bboxes)}",
            )
            for dialog_index, dialog_bbox in enumerate(dialog_bboxes)
        ]
        dialog_text = " ".join(text for text in dialog_texts if text).strip()

        characters = frame.get("characters", [])
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
                    caption=str(frame.get("caption", "")),
                    dialog_bboxes=dialog_bboxes,
                    dialog_texts=dialog_texts,
                    dialog_text=dialog_text,
                    character_ids=[str(character.get("id", "")) for character in characters],
                    character_bboxes=[
                        [int(value) for value in character.get("bbox", [])]
                        for character in characters
                    ],
                    character_types=[int(character.get("type", -1)) for character in characters],
                ),
                padded_image=padded,
            )
        )

    return records


def load_page_image_from_raw(
    sample: dict[str, Any],
    raw_dir: Path,
    progress: Any | None = None,
) -> Image.Image:
    local_pages = sample.get("_local_pages", {})
    left_path = raw_dir / str(local_pages["url1_path"])
    right_path = raw_dir / str(local_pages["url2_path"])
    if not left_path.exists() or not right_path.exists():
        raise NetworkImageError("missing local page image")
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    meta = sample["meta"]
    width1 = int(meta.get("width1", left.size[0]))
    width2 = int(meta.get("width2", right.size[0]))
    set_progress_status(progress, "拼接页面")
    page = Image.new("RGB", (width1 + width2, max(left.size[1], right.size[1])), "black")
    page.paste(left.resize((width1, left.size[1])), (0, 0))
    page.paste(right.resize((width2, right.size[1])), (width1, 0))
    return page


def group_by_page(prepared_panels: list[PreparedPanel]) -> dict[str, list[PreparedPanel]]:
    page_groups: dict[str, list[PreparedPanel]] = {}
    for prepared_panel in prepared_panels:
        record = prepared_panel.record
        page_key = f"{record.manga_id}/{record.chapter_id}/{record.page_id}"
        page_groups.setdefault(page_key, []).append(prepared_panel)
    return page_groups


def write_puzzle_directory(
    puzzle_index: int,
    group_key: str,
    page_key: str,
    window: list[PreparedPanel],
    output_dir: Path,
    target_size: tuple[int, int],
    image_format: str,
    rng: random.Random,
) -> dict[str, Any]:
    puzzle_dir = output_dir / f"{puzzle_index:06d}"
    padded_dir = puzzle_dir / "panels_padded"
    padded_dir.mkdir(parents=True, exist_ok=True)

    panel_count = len(window)
    input_order = list(range(panel_count))
    rng.shuffle(input_order)
    target_order = [input_order.index(index) for index in range(panel_count)]

    shuffled_panels = []
    for input_index, original_index in enumerate(input_order):
        prepared_panel = window[original_index]
        record = prepared_panel.record
        padded_name = f"input_{input_index:02d}.{image_format}"
        padded_path = padded_dir / padded_name
        prepared_panel.padded_image.save(padded_path)

        panel_json = record.to_json()
        panel_json["padded_path"] = f"{puzzle_index:06d}/panels_padded/{padded_name}"
        panel_json["input_index"] = input_index
        panel_json["original_window_index"] = original_index
        shuffled_panels.append(panel_json)

    sample = {
        "sequence_id": f"{page_key.replace('/', '_')}_puzzle_{puzzle_index:06d}",
        "puzzle_index": puzzle_index,
        "manga_id": window[0].record.manga_id,
        "chapter_id": window[0].record.chapter_id,
        "page_id": window[0].record.page_id,
        "page_index": window[0].record.page_index,
        "group_key": group_key,
        "page_key": page_key,
        "panel_count": panel_count,
        "target_panel_size": [target_size[0], target_size[1]],
        "panels": shuffled_panels,
        "input_order": input_order,
        "target_order": target_order,
        "text_source": "dialog_text",
    }
    with (puzzle_dir / "sample.json").open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)
    return sample


def parse_source_path(image_path: str) -> tuple[str, str, int]:
    parts = Path(image_path).parts
    manga_id = parts[0] if parts else "unknown_manga"
    page_stem = Path(image_path).stem
    page_index = int(page_stem) if page_stem.isdigit() else stable_int(page_stem) % 1_000_000
    chapter_id = "default"
    if len(parts) >= 3:
        chapter_id = parts[-2]
    return manga_id, chapter_id, page_index


def make_page_id(image_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path).strip("_").replace(".", "_")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got {bbox}")
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]


def pad_image(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, list[int]]:
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


def recognize_dialog_text(
    image: Image.Image,
    ocr: Any | None,
    skip_errors: bool = True,
    context: str = "",
    progress: Any | None = None,
    progress_status: str = "OCR",
) -> str:
    if ocr is None:
        return ""
    try:
        set_progress_status(progress, progress_status)
        return ocr.recognize(image, context=context)
    except Exception as exc:
        return handle_ocr_error(exc, skip_errors, context)


def handle_ocr_error(exc: Exception, skip_errors: bool, context: str) -> str:
    if not skip_errors:
        raise exc
    prefix = f"OCR failed for {context}: " if context else "OCR failed: "
    print(f"{prefix}{type(exc).__name__}: {exc}")
    return ""


def run_dialog_ocr_inline(image: Image.Image, ocr: Any | None) -> str:
    if ocr is None:
        return ""
    image_array = pil_to_rgb_array(image)
    try:
        predict = getattr(ocr, "predict", None)
        if callable(predict):
            result = predict(image_array)
            texts = extract_ocr_texts(result)
            return " ".join(text for text in texts if text)
        result = ocr.ocr(image_array, cls=True)
    except TypeError as exc:
        if "cls" not in str(exc):
            raise
        result = ocr.ocr(image_array)
    texts = extract_ocr_texts(result)
    return " ".join(text for text in texts if text)


def extract_ocr_texts(result: Any) -> list[str]:
    texts: list[str] = []
    if result is None:
        return texts
    if isinstance(result, dict):
        for key in ("rec_texts", "texts"):
            values = result.get(key)
            if isinstance(values, list):
                texts.extend(str(value).strip() for value in values if str(value).strip())
        for key in ("text", "transcription"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        for key in ("res", "result", "ocr_result"):
            if key in result:
                texts.extend(extract_ocr_texts(result[key]))
        return texts
    if isinstance(result, str):
        return [result.strip()] if result.strip() else texts
    if isinstance(result, (list, tuple)):
        if len(result) >= 2 and isinstance(result[1], (list, tuple)) and result[1]:
            candidate = result[1][0]
            if isinstance(candidate, str) and candidate.strip():
                texts.append(candidate.strip())
        for item in result:
            texts.extend(extract_ocr_texts(item))
        return texts

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return extract_ocr_texts(to_dict())
    json_obj = getattr(result, "json", None)
    if callable(json_obj):
        return extract_ocr_texts(json_obj())
    return texts


def pil_to_rgb_array(image: Image.Image):
    import numpy as np

    return np.asarray(image.convert("RGB"))


def handle_network_error(exc: NetworkImageError, skip_errors: bool, source_image_path: str) -> list[PreparedPanel]:
    if not skip_errors:
        raise exc
    print(f"Network error for {source_image_path}; skip current page sample: {exc}")
    return []


def set_progress_status(progress: Any | None, status: str) -> None:
    if progress is None:
        return
    try:
        progress.set_postfix_str(f"step={status}", refresh=True)
    except Exception:
        return


if __name__ == "__main__":
    main()
