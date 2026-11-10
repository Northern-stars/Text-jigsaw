"""Create a t-SNE visualization of newspaper page image embeddings."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_project_root = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "config" / "cli.py").is_file()),
    None,
)
if _project_root is not None and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.cli import parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="../chronam-get-images/predicted/**/*.jpg",
        help="Glob for page images; recursive patterns are supported.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tests"))
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=10)
    parser.add_argument("--figure-size", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=500)
    return parse_args_with_config(parser, script_file=__file__)


def main() -> None:
    args = parse_args()

    from img2vec_pytorch import Img2Vec
    import matplotlib.pyplot as plt
    from matplotlib.image import BboxImage
    from matplotlib.transforms import Bbox, TransformedBbox
    from PIL import Image
    from sklearn.manifold import TSNE

    image_paths = sorted(glob.glob(args.input_glob, recursive=True))
    if not image_paths:
        raise FileNotFoundError(f"No images matched --input-glob: {args.input_glob}")
    if args.perplexity <= 0 or args.perplexity >= len(image_paths):
        raise ValueError(f"--perplexity must be positive and smaller than image count ({len(image_paths)})")

    image_vectors = []
    image_model = Img2Vec(cuda=args.use_cuda, model="alexnet")
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_vectors.append(image_model.get_vec(image.convert("RGB"), tensor=False))

    embedded = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="random",
        random_state=args.random_state,
    ).fit_transform(image_vectors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.scatter(embedded[:, 0], embedded[:, 1], marker=".", color="blue")
    plt.savefig(args.output_dir / "tsne_test.png")
    plt.clf()

    figure = plt.figure(figsize=(args.figure_size, args.figure_size))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    axis = figure.add_subplot(111)
    plt.axis("off")
    for index, image_path in enumerate(image_paths):
        bbox = Bbox.from_bounds(embedded[index][0], embedded[index][1], 1.0, 1.0)
        transformed_bbox = TransformedBbox(bbox, axis.transData)
        bbox_image = BboxImage(transformed_bbox, norm=None, origin=None, clip_on=False, cmap="gray")
        bbox_image.set_data(plt.imread(image_path))
        axis.add_artist(bbox_image)

    plt.savefig(args.output_dir / "testfig.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
