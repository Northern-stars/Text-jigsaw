"""Combined COMIX stream + ordering-build CLI (compatibility entry point).

The implementation now lives in :mod:`comix.comix_pipeline`, which is the single
source of truth for both phases of the COMIX (Hugging Face) pipeline:

1. ``stream_raw_samples`` streams raw page images and metadata from the official
   COMIX Hugging Face dataset (``emanuelevivoli/comix-v0_1-pages``).
2. ``build_ordering_from_raw`` converts those raw samples into fixed-count
   shuffled ordering puzzles (manifest.jsonl, sample.json, panels_padded/).

This module is kept so existing invocations keep working:

    # full pipeline: stream + build
    python comix/stream_comix.py --dataset-name emanuelevivoli/comix-v0_1-pages --max-pages 100 --puzzle-num 500

    # stream only (no ordering build)
    python comix/stream_comix.py --dataset-name emanuelevivoli/comix-v0_1-pages --stream-only

    # build only (skip download, use existing raw dir)
    python comix/stream_comix.py --build-only --raw-dir Data/comix/raw

Equivalent two-phase entry points are ``comix/download_raw.py`` (download only)
and ``comix/build_ordering_dataset.py`` (build only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comix.comix_pipeline import main

if __name__ == "__main__":
    main()
