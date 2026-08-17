"""Visualize a random solved jigsaw sample from a dataset directory."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import ImageDraw

sys.path.append(".")

from Solver.env.jigsaw_env import GRID_SIZE, SOLVED_ORDER, TextJigsawEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly select a dataset sample and save its solved jigsaw image.",
    )
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory, label.json, or manifest.json.")
    parser.add_argument(
        "--output",
        default="Solver/visualizations/random_solved_sample.jpg",
        help="Output image path.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional deterministic sample index after environment path discovery.",
    )
    parser.add_argument(
        "--draw-grid",
        action="store_true",
        help="Draw thin grid lines between pieces.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the saved visualization with the default image viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = TextJigsawEnv(dataset_dir=args.dataset_dir, seed=args.seed)
    sample_path = select_sample_path(env.sample_paths, seed=args.seed, sample_index=args.sample_index)

    env.sample_path = sample_path
    env.pieces = env._load_sample(sample_path)
    env.current_order = SOLVED_ORDER.copy()
    env.step_count = 0

    image = env.render()
    if args.draw_grid:
        image = draw_grid(image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"sample_path={sample_path}")
    print(f"output={output_path}")
    print(f"image_size={image.size[0]}x{image.size[1]}")

    if args.show:
        image.show()


def select_sample_path(
    sample_paths: list[Path],
    seed: int | None,
    sample_index: int | None,
) -> Path:
    if not sample_paths:
        raise ValueError("No samples found.")

    if sample_index is not None:
        if not 0 <= sample_index < len(sample_paths):
            raise IndexError(
                f"sample_index={sample_index} out of range for {len(sample_paths)} samples",
            )
        return sample_paths[sample_index]

    rng = random.Random(seed)
    return rng.choice(sample_paths)


def draw_grid(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    piece_w = width // GRID_SIZE
    piece_h = height // GRID_SIZE

    for index in range(1, GRID_SIZE):
        x = index * piece_w
        y = index * piece_h
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=2)
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=2)

    return image


if __name__ == "__main__":
    main()
