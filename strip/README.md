# Comic strips ordering dataset

This directory supports `bghira/comicstrips-gpt4o-blip3` in two offline-friendly stages.

```bash
python strip/download_raw.py --max-strips 100
python strip/build_ordering_dataset.py --panel-count 4 --puzzle-num 500
```

The builder writes `sample.json`, padded panel images, `train.jsonl`,
`val.jsonl`, `test.jsonl`, `manifest.jsonl`, and `meta.json` under
`Data/strip/ordering`. The manifest format is the same one consumed by
`Solver/data/panel_ordering_dataset.py`, so the existing ordering solvers can
use this directory directly.

Each source strip is assigned to exactly one split before its windows are
generated. All windows and shuffle variants from that strip stay in the same
split, preventing source-strip leakage. The default split budget is 80/10/10;
`--train-puzzles`, `--val-puzzles`, and `--test-puzzles` enable explicit quotas.

If the downloaded records contain panel images or bounding boxes, those are
used. Otherwise the builder treats each strip image as a regular grid. For
datasets with a known layout, set both `--grid-rows` and `--grid-cols`, for
example `--grid-rows 1 --grid-cols 4`.
