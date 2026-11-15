"""Shared CLI and model options for the two ordering entry points."""

import argparse
import json
from pathlib import Path

import torch

from config.cli import parse_args_with_config
from Solver.solvers import SOLVERS


def parse_args(argv=None, *, testing=False):
    parser = argparse.ArgumentParser(description="Train/test endpoint-conditioned comic ordering solvers")
    parser.add_argument("--solver", choices=tuple(SOLVERS), required=not testing)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, required=not testing)
    parser.add_argument("--model-options", type=json.loads, default={}, help="JSON object of solver constructor options")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--decoder-dim", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=None)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora", default=None)
    parser.add_argument("--use-layout", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    if testing:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--split", choices=("train", "val", "test"), default="test")
        parser.add_argument("--output-json", type=Path, default=None)
    else:
        parser.add_argument("--epochs", "--epoch", type=int, default=3)
        parser.add_argument("--grad-accum-steps", type=int, default=8)
        parser.add_argument("--grad-clip-norm", type=float, default=1.0)
        parser.add_argument("--lr-backbone", type=float, default=1e-4)
        parser.add_argument("--lr-decoder", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--save-dir", type=Path, default=None)
        parser.add_argument("--resume", "--load", type=Path, default=None)
        parser.add_argument("--save-every", type=int, default=1)
    entrypoint = Path(__file__).with_name("test.py" if testing else "train.py")
    args = parse_args_with_config(parser, argv, script_file=entrypoint)
    if args.batch_size < 1 or args.image_width < 1 or args.image_height < 1:
        parser.error("batch size and image dimensions must be positive")
    if not testing and (args.panel_count < 3 or args.epochs < 1 or args.grad_accum_steps < 1):
        parser.error("panel-count >= 3, epochs >= 1 and grad-accum-steps >= 1 are required")
    if not isinstance(args.model_options, dict):
        parser.error("--model-options must be a JSON object")
    return args


def model_options(args):
    options = dict(args.model_options)
    options["panel_count"] = args.panel_count
    for key in ("model_name", "decoder_dim", "decoder_layers", "encoder_layers", "num_heads",
                "torch_dtype", "device_map", "load_in_4bit", "use_lora", "use_layout"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.solver == "made2order":
        options.setdefault("image_size", (args.image_height, args.image_width))
        options.setdefault("device", args.device)
    elif args.solver in ("set2seq", "permutation"):
        options.setdefault("image_size", (args.image_height, args.image_width))
        options.setdefault("use_layout", False)
    return options
