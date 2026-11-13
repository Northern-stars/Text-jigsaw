# First-panel classifier

This feasibility experiment receives all panels in one randomized puzzle and
predicts the first input slot listed in the reading-order sequence. It uses the
last hidden state of Qwen2.5-VL as a puzzle representation followed by a
linear classification head with `panel_count` output classes. The layout
features in the shared ordering dataset are disabled, so the experiment is
layout-blind.

The target is derived from `target_order`: for an input puzzle with
`target_order = [2, 0, 1]`, the label is `2` because input slot 2 is the first
panel in the original sequence. The dataset must contain the standard
`train.jsonl`, `val.jsonl`, and `test.jsonl` (or a split-aware manifest) read by
`Solver.data.panel_ordering_dataset.MangaZeroPanelOrderingDataset`.

Checkpoints trained with the old `argmin(target_order)` label implement a
different task and must be retrained; the corrected label definition is logged
as `target_order_first_input_panel_index_v2`.

Run the default frozen-backbone feasibility test:

```powershell
python -m experiment.first_panel_classifier.train `
  --dataset-dir path\to\ordering_dataset `
  --panel-count 6 `
  --model-name Qwen/Qwen2.5-VL-3B-Instruct
```

The default configuration is
`config/experiment/first_panel_classifier/train.json`; command-line values
override it. A run creates `config.json`, `run.log`, `metrics.jsonl`, and
atomic checkpoints under `experiments/first_panel_classifier/<run-id>/`.
Use `--resume path\to\checkpoints\epoch_0001.pt` to restore optimizer state,
random states, and trainable weights. `--grad-accum-steps` controls the number
of micro-batches per optimizer update.

By default only the classifier is trainable, which is the recommended first
smoke test. Add `--train-backbone` to insert LoRA adapters; `--load-in-4bit`
can then be used for QLoRA on a compatible bitsandbytes installation. The
classifier head is placed on the backbone output device when a Transformers
`device_map` is used.
