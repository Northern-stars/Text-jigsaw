"""Download the COMIX pages dataset into a reusable local raw directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comix.comix_pipeline import parse_args, stream_raw_samples, validate_args


def main() -> None:
    args = parse_args("download")
    args.stream_only = True
    validate_args(args)
    stream_raw_samples(args)


if __name__ == "__main__":
    main()
