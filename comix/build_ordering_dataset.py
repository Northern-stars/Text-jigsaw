"""Build COMIX panel-ordering puzzles from an existing raw directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comix.comix_pipeline import build_ordering_from_raw, parse_args, validate_args


def main() -> None:
    args = parse_args("build")
    args.build_only = True
    validate_args(args)
    build_ordering_from_raw(args)


if __name__ == "__main__":
    main()
