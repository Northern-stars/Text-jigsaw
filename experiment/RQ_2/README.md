# RQ2: Style-Balanced Training

Compare `manga_only`, `comics_only`, and `mixed` Qwen2.5-VL LoRA runs under the same number of unique puzzles, epochs, optimizer settings, model revision, and seed. Validation selects the checkpoint; test is evaluated once after training.

Build a work-level split manifest:

```powershell
python -m experiment.RQ_2.build_rq2_manifest `
  --source manga:manga109:Data/Manga109/ordering_output `
  --source comics:comix:comix/ordering_dataset `
  --output experiments/manifests/rq2_seed42.jsonl --seed 42
```

Run one condition (repeat with other conditions and at least three seeds):

```powershell
python -m experiment.RQ_2.rq2_style_balance --manifest experiments/manifests/rq2_seed42.jsonl `
  --condition mixed --run-id mixed_seed42 --seed 42 --train-puzzles 20000 --epochs 3 `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT --load-in-4bit --device-map auto
```

Each run stores immutable config, append-only log and JSONL metrics, and latest/best/epoch/final checkpoints under `experiments/rq2/<run-id>`. `--resume` resumes at an epoch boundary using the exact saved CLI configuration.
