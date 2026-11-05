# Experiment runners

Shared logging and checkpoint utilities live in `run_logger.py`. Each research question has its own folder:

- `RQ_2/`: work-level manifest construction and budget-matched style balance training.
- `RQ_3/`: paired visual-only, text-only, and visual-plus-text evaluation.
- `RQ_4/`: direct, CoT, and structured evidence prompt evaluation.
- `RQ_1/`: shared VLM model adapters and model-comparison/scaling experiment entry points.

Read the README in the relevant folder for the command lines.

# RQ2 Style-Balance Experiment

The runner compares `manga_only`, `comics_only`, and `mixed` Qwen2.5-VL LoRA runs. All conditions use the same total number of unique training puzzles, epochs, batch size, gradient accumulation, optimizer configuration, model revision, and seed. The mixed condition alternates styles and rotates across works. Validation selects the best epoch; test is evaluated once afterward.

## 1. Build a work-level manifest

Use existing generated ordering dataset directories as sources. Each directory must contain `manifest.jsonl` and the referenced sample JSON files.

```powershell
python -m experiment.RQ_2.build_rq2_manifest `
  --source manga:manga109:Data/Manga109/ordering_output `
  --source manga:mangazero:Mangazero/ordering_dataset `
  --source comics:comix:comix/ordering_dataset `
  --output experiments/manifests/rq2_seed42.jsonl `
  --seed 42
```

The command splits by canonical source/work ID, not by puzzle. It writes a JSONL manifest with source paths, style, work, page/window IDs, and split, plus a `.meta.json` summary. Check that every style has enough distinct works for train, validation, and test before training. The current builder namespaces work IDs by source; provide one canonical source per work or manually merge known cross-dataset aliases before the run.

## 2. Run the three conditions

Use the same arguments for all conditions, changing only `--condition` and `--run-id`. `--train-puzzles` is the common total budget per condition; when omitted it uses the smaller style pool size. The mixed group divides that budget as evenly as possible between styles. Run at least three seeds for the study.

```powershell
python -m experiment.RQ_2.rq2_style_balance --manifest experiments/manifests/rq2_seed42.jsonl --condition manga_only --run-id manga_seed42 --seed 42 --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT --train-puzzles 20000 --epochs 3 --load-in-4bit --device-map auto
python -m experiment.RQ_2.rq2_style_balance --manifest experiments/manifests/rq2_seed42.jsonl --condition comics_only --run-id comics_seed42 --seed 42 --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT --train-puzzles 20000 --epochs 3 --load-in-4bit --device-map auto
python -m experiment.RQ_2.rq2_style_balance --manifest experiments/manifests/rq2_seed42.jsonl --condition mixed --run-id mixed_seed42 --seed 42 --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT --train-puzzles 20000 --epochs 3 --load-in-4bit --device-map auto
```

The example budget is illustrative; choose a value supported by all three pools. Model downloads and GPU training are not run by tests. If training without quantization, omit `--load-in-4bit --device-map auto`. Pin the same model revision and use the same hardware/software environment across conditions.

## 3. Reproduction artifacts

Each run directory contains immutable `config.json`, append-only `run.log` and `metrics.jsonl`, and `checkpoints/` with `epoch_*.pt`, `latest.pt`, `best.pt`, and `final.pt`. Checkpoints contain adapter weights, optimizer state, epoch/global step, Python/NumPy/PyTorch RNG states, and validation/test metadata. Resume from an epoch-boundary checkpoint with the identical CLI configuration and `--resume <checkpoint>`; checkpoints are not saved mid-epoch.

Prediction JSONL rows preserve sequence/work/style IDs, target, parsed prediction, raw output, validity, and metrics. The aggregate records exact sequence accuracy, position accuracy, pairwise accuracy, and per-style results. Test is run with the checkpoint selected by validation.
