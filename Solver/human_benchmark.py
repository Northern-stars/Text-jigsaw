"""Run an interactive human benchmark on panel-ordering puzzles."""

from __future__ import annotations

import argparse
import random
import sys
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
        description="Display random panel-ordering puzzles and score human answers."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where displayed puzzle images are saved.",
    )
    return parse_args_with_config(parser, argv, script_file=__file__)


def make_puzzle_image(
    panel_images: Sequence,
    displayed_indices: Sequence[int],
    *,
    columns: int = 4,
    padding: int = 12,
) -> Image.Image:
    """Create a numbered panel sheet in the order shown to the participant."""
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not displayed_indices:
        raise ValueError("at least one panel is needed to render a puzzle")

    first = panel_images[displayed_indices[0]]
    if hasattr(first, "detach"):
        first = first.detach().cpu().numpy()
    tile_height, tile_width = int(first.shape[-2]), int(first.shape[-1])
    rows = (len(displayed_indices) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * padding,
         rows * (tile_height + 30) + (rows + 1) * padding),
        "#e9e9e9",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

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
        left = padding + column * (tile_width + padding)
        top = padding + row * (tile_height + 30 + padding)
        draw.text((left, top), f"Panel {position + 1}", fill="black", font=font)
        canvas.paste(panel_image, (left, top + 20))
    return canvas


def parse_answer(answer: str, panel_count: int) -> list[int] | None:
    """Parse a 1-based panel sequence; return None for quit/blank input."""
    value = answer.strip().lower()
    if not value or value in {"q", "quit", "exit"}:
        return None
    tokens = value.replace(",", " ").split()
    try:
        order = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("Enter panel numbers separated by spaces or commas, for example: 2 1 3 4") from error
    expected = set(range(1, panel_count + 1))
    if len(order) != panel_count or set(order) != expected:
        raise ValueError(f"Use every panel number from 1 to {panel_count} exactly once.")
    return order


def correct_display_order(target_order: Sequence[int], displayed_indices: Sequence[int]) -> list[int]:
    """Return 1-based displayed panel IDs in canonical reading order."""
    if len(target_order) != len(displayed_indices):
        raise ValueError("target_order and displayed_indices must have equal length")
    input_positions_in_reading_order = sorted(range(len(target_order)), key=lambda index: target_order[index])
    return [displayed_indices.index(index) + 1 for index in input_positions_in_reading_order]


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

    print(f"Human benchmark: {len(dataset)} puzzles available in split '{args.split}'.")
    print("Enter panel IDs from earliest to latest reading order; use q to finish.\n")

    for question_number in range(1, args.num_questions + 1):
        if not sample_indices:
            sample_indices = list(range(len(dataset)))
            rng.shuffle(sample_indices)
        sample_index = sample_indices.pop()
        sample = dataset[sample_index]
        panel_count = int(sample["target_order"].numel())
        displayed_indices = list(range(panel_count))
        rng.shuffle(displayed_indices)
        truth = correct_display_order(sample["target_order"].tolist(), displayed_indices)
        puzzle_image = make_puzzle_image(
            sample["panel_images"], displayed_indices, columns=args.columns
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
                    input(f"Reading order (1-{panel_count}, or q to quit): "), panel_count
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
            print(f"Incorrect. Correct order: {' '.join(map(str, truth))}")
        print(f"Score: {correct_count}/{answered_count} ({100 * correct_count / answered_count:.1f}%)\n")

    accuracy = correct_count / answered_count if answered_count else 0.0
    print(f"Final score: {correct_count}/{answered_count}; accuracy={accuracy:.1%}")
    return correct_count, answered_count


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
