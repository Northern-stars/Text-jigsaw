"""Run an interactive human benchmark on panel-ordering puzzles."""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Sequence

_project_root = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "config" / "cli.py").is_file()),
    None,
)
if _project_root is not None and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.cli import parse_args_with_config

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Give the first comic panel and score human ordering of the remaining panels."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--panel-text-mode",
        choices=("auto", "translation", "description", "none"),
        default="auto",
        help="Show translated dialogue, a panel description, the first available text, or no text.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where displayed puzzle images are saved.",
    )
    parser.add_argument("--results-path", type=Path, default=None,
                        help="Optional JSONL file for reproducible question and answer records.")
    return parse_args_with_config(parser, argv, script_file=__file__)


def make_puzzle_image(
    panel_images: Sequence,
    displayed_indices: Sequence[int],
    *,
    columns: int = 4,
    padding: int = 12,
    panel_texts: Sequence[str] | None = None,
    given_first: bool = False,
) -> Image.Image:
    """Create a numbered sheet, optionally placing the given opening panel on its own row."""
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not displayed_indices:
        raise ValueError("at least one panel is needed to render a puzzle")

    first = panel_images[displayed_indices[0]]
    if hasattr(first, "detach"):
        first = first.detach().cpu().numpy()
    tile_height, tile_width = int(first.shape[-2]), int(first.shape[-1])
    remaining_count = len(displayed_indices) - int(given_first)
    rows = int(given_first) + (remaining_count + columns - 1) // columns
    text_height = 82 if panel_texts is not None else 0
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * padding,
         rows * (tile_height + 30 + text_height) + (rows + 1) * padding),
        "#e9e9e9",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_panel_font()

    for position, panel_index in enumerate(displayed_indices):
        panel = panel_images[panel_index]
        if hasattr(panel, "detach"):
            panel = panel.detach().cpu().numpy()
        panel = np.asarray(panel)
        if panel.ndim == 3 and panel.shape[0] in (1, 3, 4):
            panel = np.transpose(panel[:3], (1, 2, 0))
        if panel.dtype.kind == "f":
            panel = np.clip(panel * 255.0 if panel.max(initial=0) <= 1.0 else panel, 0, 255)
        panel_image = Image.fromarray(panel.astype(np.uint8)).convert("RGB")

        row, column = divmod(position, columns)
        if given_first and position > 0:
            row, column = divmod(position - 1, columns)
            row += 1
        left = padding + column * (tile_width + padding)
        top = padding + row * (tile_height + 30 + text_height + padding)
        label = "Panel 1 (given start)" if given_first and position == 0 else f"Panel {position + 1}"
        draw.text((left, top), label, fill="black", font=font)
        canvas.paste(panel_image, (left, top + 20))
        if panel_texts is not None:
            text = panel_texts[panel_index] if panel_index < len(panel_texts) else ""
            draw_panel_text(draw, left, top + 22 + tile_height, tile_width, text_height, text, font)
    return canvas


def panel_display_text(panel: dict, mode: str = "auto") -> str:
    """Select translated dialogue or descriptive text from a panel record."""
    if mode == "none":
        return ""
    translated = first_panel_text(panel, (
        "translated_text", "dialog_text_translated", "translated_dialog_text", "translation"
    ))
    description = first_panel_text(panel, (
        "panel_description", "description", "image_description", "alt_text", "blip3_caption"
    ))
    caption = first_panel_text(panel, ("caption",))
    dialogue = first_panel_text(panel, ("dialog_text", "dialog_texts"))

    if mode == "translation":
        value = translated or dialogue
        label = "Translation" if translated else "Dialogue"
    elif mode == "description":
        value = description or caption
        label = "Description"
    else:
        value = translated or description or caption or dialogue
        label = (
            "Translation" if translated else
            "Description" if description or caption else
            "Dialogue"
        )
    return f"{label}: {value}" if value else ""


def first_panel_text(panel: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = panel.get(key)
        if isinstance(value, (list, tuple)):
            value = " ".join(str(item) for item in value if item)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def load_panel_font():
    for font_path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, 13)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_panel_text(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    if not text:
        return
    available = max(1, width - 8)
    lines = wrap_to_pixel_width(text, font, available, draw)
    line_height = max(13, draw.textbbox((0, 0), "Ag", font=font)[3] + 2)
    max_lines = max(1, height // line_height)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = (lines[-1][:-1] + "…") if lines[-1] else "…"
    draw.multiline_text(
        (left + 4, top + 4), "\n".join(lines), fill="#202020", font=font, spacing=1
    )


def wrap_to_pixel_width(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = ""
        if draw.textbbox((0, 0), word, font=font)[2] <= max_width:
            current = f"{current} {word}".strip()
            continue
        for character in word:
            candidate = current + character
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
    if current:
        lines.append(current)
    return lines


def parse_answer(answer: str, panel_count: int, *, first_panel_id: int = 1) -> list[int] | None:
    """Parse IDs from first_panel_id through panel_count; None means quit/blank."""
    value = answer.strip().lower()
    if not value or value in {"q", "quit", "exit"}:
        return None
    tokens = value.replace(",", " ").split()
    try:
        order = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("Enter panel numbers separated by spaces or commas.") from error
    expected = set(range(first_panel_id, panel_count + 1))
    if len(order) != len(expected) or set(order) != expected:
        raise ValueError(f"Use every panel number from {first_panel_id} to {panel_count} exactly once.")
    return order


def correct_display_order(target_order: Sequence[int], displayed_indices: Sequence[int]) -> list[int]:
    """Return 1-based displayed panel IDs in canonical reading order."""
    if len(target_order) != len(displayed_indices):
        raise ValueError("target_order and displayed_indices must have equal length")
    expected = set(range(len(target_order)))
    if set(target_order) != expected or set(displayed_indices) != expected:
        raise ValueError("orders must be permutations of input panel indices")
    input_to_display = {index: position + 1 for position, index in enumerate(displayed_indices)}
    return [input_to_display[index] for index in target_order]


def run_benchmark(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_questions <= 0:
        raise ValueError("--num-questions must be positive")
    if args.image_width <= 0 or args.image_height <= 0 or args.columns <= 0:
        raise ValueError("image dimensions and columns must be positive")

    dataset = MangaZeroPanelOrderingDataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        image_size=(args.image_width, args.image_height),
        use_layout=False,
    )
    rng = random.Random(args.seed)
    sample_indices = list(range(len(dataset)))
    rng.shuffle(sample_indices)
    correct_count = answered_count = 0

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    results_handle = None
    benchmark_run_id = str(uuid.uuid4())
    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        results_handle = args.results_path.open("a", encoding="utf-8")

    print(f"Human benchmark: {len(dataset)} puzzles available in split '{args.split}'.")
    print("Panel 1 is the given start. Order only the remaining panels; use q to finish.\n")

    for question_number in range(1, args.num_questions + 1):
        if not sample_indices:
            sample_indices = list(range(len(dataset)))
            rng.shuffle(sample_indices)
        sample_index = sample_indices.pop()
        sample = dataset[sample_index]
        panel_count = int(sample["target_order"].numel())
        if panel_count < 2:
            raise ValueError("Each puzzle must contain a starting panel and at least one remaining panel.")
        target_order = sample["target_order"].tolist()
        displayed_indices = [index for index in range(panel_count) if index != target_order[0]]
        rng.shuffle(displayed_indices)
        displayed_indices.insert(0, target_order[0])
        truth = correct_display_order(target_order, displayed_indices)[1:]
        panel_texts = [
            panel_display_text(panel, args.panel_text_mode)
            for panel in sample["panels"]
        ]
        display_texts = panel_texts if args.panel_text_mode != "none" else None
        puzzle_image = make_puzzle_image(
            sample["panel_images"],
            displayed_indices,
            columns=args.columns,
            panel_texts=display_texts,
            given_first=True,
        )

        image_path = None
        if args.output_dir is not None:
            image_path = args.output_dir / f"question_{question_number:04d}.png"
            puzzle_image.save(image_path)

        print(f"Question {question_number}/{args.num_questions} — {panel_count} panels")
        if image_path is not None:
            print(f"Puzzle image: {image_path.resolve()}")
        puzzle_image.show(title=f"Human benchmark — question {question_number}")

        while True:
            try:
                answer = parse_answer(
                    input(f"After Panel 1, remaining order (2-{panel_count}, or q to quit): "),
                    panel_count,
                    first_panel_id=2,
                )
                break
            except ValueError as error:
                print(error)
        if answer is None:
            break

        answered_count += 1
        if answer == truth:
            correct_count += 1
            print("Correct.")
        else:
            print(f"Incorrect. Correct remaining order after Panel 1: {' '.join(map(str, truth))}")
        if results_handle is not None:
            results_handle.write(json.dumps({
                "benchmark_run_id": benchmark_run_id,
                "task": "order_remaining_given_first",
                "question_number": question_number,
                "dataset_dir": str(args.dataset_dir.resolve()),
                "split": args.split,
                "seed": args.seed,
                "sequence_id": sample["sequence_id"],
                "reading_order_version": sample["panels"][0].get("reading_order_version"),
                "reading_order_method": sample["panels"][0].get("reading_order_method"),
                "reading_direction": sample.get("reading_direction"),
                "displayed_input_indices": displayed_indices,
                "given_display_order": [1],
                "reference_display_order": truth,
                "answer": answer,
                "correct": answer == truth,
                "panel_text_mode": args.panel_text_mode,
                "image_size": [args.image_width, args.image_height],
                "puzzle_image_path": str(image_path.resolve()) if image_path is not None else None,
            }, ensure_ascii=False) + "\n")
            results_handle.flush()
        print(f"Score: {correct_count}/{answered_count} ({100 * correct_count / answered_count:.1f}%)\n")

    if results_handle is not None:
        results_handle.close()
    accuracy = correct_count / answered_count if answered_count else 0.0
    print(f"Final score: {correct_count}/{answered_count}; accuracy={accuracy:.1%}")
    return correct_count, answered_count


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
