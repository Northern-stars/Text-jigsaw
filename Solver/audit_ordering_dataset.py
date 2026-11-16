"""Export offline label reviews using the shared training dataset and shuffle."""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from config.cli import parse_args_with_config, save_startup_config
from Solver.data.panel_ordering_dataset import (
    MangaZeroPanelOrderingDataset,
    collate_panel_ordering_batch,
)
from Solver.puzzle import shuffle_middle_batch, validate_order


TEXT_FIELDS = (
    "dialog_text", "dialog_texts", "original_text", "source_text",
    "translated_text", "dialog_text_translated", "translated_dialog_text",
    "translation", "ocr_text", "text", "panel_description", "description",
    "image_description", "alt_text", "blip3_caption", "caption",
)


def image_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def tensor_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy()
    return Image.fromarray(array)


def make_record(dataset, index, *, seed, shuffle_mode, include_original_images):
    item = dataset[index]
    batch = collate_panel_ordering_batch([item])
    if shuffle_mode == "train":
        # Independent per-sample seed: selecting additional samples does not
        # change the display of a sample already in the review.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + index)
            batch = shuffle_middle_batch(batch)
    validate_order(batch["target_order"])
    sample = dataset.samples[index]
    count = len(batch["panels"][0])
    if count != sample.panel_count:
        raise ValueError(f"{sample.sequence_id}: panel_count differs from loaded panels")
    panels = []
    for slot, panel in enumerate(batch["panels"][0]):
        source_path = dataset._resolve_path(panel["padded_path"], sample.source_dir or dataset.dataset_dir)
        rendered = tensor_image(batch["panel_images"][0, slot])
        original_uri = None
        with Image.open(source_path) as source:
            original_size = list(source.size)
            if include_original_images:
                original_uri = image_uri(source.convert("RGB"))
        panels.append({
            "slot": slot,
            "source_index": int(batch["input_panel_indices"][0, slot]),
            "role": "START" if slot == 0 else "END" if slot == count - 1 else "MIDDLE",
            "image_path": str(source_path.resolve()),
            "original_size": original_size,
            "training_size": list(rendered.size),
            "model_dialog_text": batch["dialog_texts"][0][slot],
            "texts": {key: panel[key] for key in TEXT_FIELDS if panel.get(key)},
            "metadata": panel,
            "image_uri": image_uri(rendered),
            "original_uri": original_uri,
        })
    return {
        "review_id": f"{dataset.split}:{index}:{sample.sequence_id}",
        "dataset_index": index, "sequence_id": sample.sequence_id,
        "split": dataset.split, "work_id": sample.work_id,
        "manga_id": sample.manga_id, "chapter_id": sample.chapter_id,
        "page_key": sample.page_key, "window_id": sample.window_id,
        "source_dir": sample.source_dir, "seed": seed,
        "sample_shuffle_seed": seed + index, "shuffle_mode": shuffle_mode,
        "raw_target_order": sample.target_order,
        "endpoint_dataset_target_order": item["target_order"].tolist(),
        "endpoint_dataset_source_indices": item["input_panel_indices"].tolist(),
        "target_order": batch["target_order"][0].tolist(),
        "input_source_indices": batch["input_panel_indices"][0].tolist(),
        "panels": panels,
    }


def export_audit(args):
    dataset = MangaZeroPanelOrderingDataset(
        args.dataset_dir, split=args.split,
        image_size=(args.image_width, args.image_height), use_layout=False,
        manifest_path=args.manifest_path, given_endpoints=True,
    )
    if args.sample_indices is not None:
        indices = args.sample_indices
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
            raise ValueError("--sample-indices must be unique indices within the selected split")
    else:
        indices = random.Random(args.seed).sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    records = []
    for index in indices:
        records.append(make_record(dataset, index, seed=args.seed, shuffle_mode=args.shuffle_mode,
                                   include_original_images=not args.no_original_images))
        print(f"prepared {len(records)}/{len(indices)}: index={index} id={records[-1]['sequence_id']}")
    settings = {"dataset_dir": str(args.dataset_dir.resolve()), "split": dataset.split,
                "seed": args.seed, "image_size": [args.image_width, args.image_height],
                "shuffle_mode": args.shuffle_mode, "indices": indices}
    payload = {"schema_version": 1, "settings": settings, "samples": records}
    # HTML script contents must not interpret strings from OCR/metadata as tags.
    encoded = json.dumps(payload, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("templates") / "ordering_audit.html"
    document = template.read_text(encoding="utf-8").replace("__AUDIT_PAYLOAD__", encoded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "audit.html"
    html_path.write_text(document, encoding="utf-8")
    # A compact reproducibility manifest omits the embedded image bytes.
    manifest = dict(payload, samples=[dict(record, panels=[
        {key: value for key, value in panel.items() if key not in ("image_uri", "original_uri")}
        for panel in record["panels"]]) for record in records])
    (args.output_dir / "audit_samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_startup_config(args, args.output_dir)
    print(f"Open in a browser: {html_path.resolve()}")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", nargs="+", type=int, help="Zero-based indices in the selected split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--shuffle-mode", choices=("train", "dataset"), default="train",
                        help="train applies the shared middle-panel training shuffle; dataset preserves loader order")
    parser.add_argument("--no-original-images", action="store_true", help="Omit expandable original images to reduce HTML size")
    parser.add_argument("--output-dir", type=Path, default=Path("output/ordering_audit"))
    args = parse_args_with_config(parser, argv, script_file=__file__)
    if min(args.num_samples, args.image_width, args.image_height) < 1:
        parser.error("sample count and image dimensions must be positive")
    return args


if __name__ == "__main__":
    export_audit(parse_args())
