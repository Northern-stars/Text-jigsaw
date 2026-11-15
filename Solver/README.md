# Comic panel ordering solvers

Comic ordering now has two entry points: `Solver/train.py` and `Solver/test.py`.
Select the model with `--solver`; the puzzle and epoch runner do not select
model-specific losses or decoding algorithms.

## Structure

- `puzzle.py`: shared endpoint task, synchronized shuffling, dispatch and metrics.
- `data/panel_ordering_dataset.py`: shared dataset reader, split manifests and collate.
- `solvers/`: registry and solver-owned `training_loss`, `predict`, training
  augmentation and checkpoint policies; `agent/` contains the model architectures.
- `runtime.py`: one gradient-accumulation loop, evaluation and checkpoint IO.
- `cli.py`, `train.py`, `test.py`: shared command-line configuration and entry points.
- `utils.py`: compatibility helpers used by research experiments.

The separate `train_rl.py` implements the spatial text-jigsaw DQN task.
Files named `test_*` are regression tests, not model evaluation entry points.
Old per-model comic ordering train/test launchers and their configs have been
removed; their model architectures remain available.

## Puzzle contract

The GT opening panel is placed in input slot **0**, and the GT ending panel in
slot **K-1**. Only the **K-2 middle panels** are shuffled and predicted. All
modalities move together, including dialogue, layout features and provenance.
`target_order` is a sequence of input slots in reading order, not a vector of
per-slot ranks. Full target/prediction permutations retain both endpoints.

`panel_count` includes both given panels and must be at least 3. Images for all
panels remain available as context. Pointer and rank losses omit both endpoint
rows/columns; classifiers enumerate `(K-2)!` orders. Text generation emits only
middle panel IDs. Exact match, position and pairwise accuracy score only middle
panels; a single middle panel is a trivial puzzle with accuracy 1.

No dataset regeneration is needed: `load_panel_ordering_splits` enables endpoint
conditioning by default. The raw dataset reader still defaults to the original
indices, for first-panel studies. Pass `given_endpoints=True` when constructing
it directly for ordering. `data/given_start.py` retains compatibility imports for
first-only studies; the implementations live in `puzzle.py`. The human benchmark
and the RQ ordering code use the same endpoint contract.

## Available solvers

| `--solver` | Architecture |
| --- | --- |
| `set2seq` | visual/text/multimodal Transformer pointer |
| `permutation` | visual/text/multimodal permutation classifier |
| `jigsaw` | CNN and bounded permutation classifier |
| `made2order` | upstream Made2Order rank prediction |
| `qwen_vl_autoregressive` | Qwen2.5-VL + LoRA and autoregressive pointer |
| `qwen_vl_nonautoregressive` | Qwen2.5-VL + LoRA, set encoder, parallel pointer and row/column loss |
| `qwen_vl_classifier` | Qwen2.5-VL + LoRA permutation classifier |
| `qwen_vl` | Qwen2.5-VL generates middle ordering as text |
| `qwen_vl_vector_expert` | vector-conditioned experts on an endpoint-trained Qwen pointer checkpoint |

## Train and test

The entry points automatically load
[`config/Solver/train.json`](../config/Solver/train.json) and
[`config/Solver/test.json`](../config/Solver/test.json). Paths in the configs
are resolved from the working directory; run the commands from the repository
root. CLI arguments override JSON defaults. Use `--config path/to/config.json`
to select a different config file.

| Setting | Training default | Test default |
| --- | --- | --- |
| `solver` | `qwen_vl_nonautoregressive` | restored from checkpoint |
| `panel_count` | 6, including both given endpoints | restored from checkpoint |
| `dataset_dir` | required via CLI or config | required via CLI or config |
| `checkpoint` | use `resume` to continue training | required via CLI or config |
| `epochs` / `grad_accum_steps` | 3 / 8 | not applicable |
| `batch_size` / `num_workers` | 1 / 0 | 1 / 0 |
| `image_width` / `image_height` | 224 / 224 | restored from checkpoint |
| `lr_backbone` / `lr_decoder` | 0.0001 / 0.001 | not applicable |
| `grad_clip_norm` / `weight_decay` | 1.0 / 0.0 | not applicable |
| `device` / `seed` | auto CUDA/CPU / 42 | auto CUDA/CPU / 42 |
| `save_dir` / `save_every` | `Solver/checkpoints/<solver>` / each epoch | not applicable |
| `split` / `output_json` | final test after validation selection | `test` / no output file |

`null` keeps the CLI's own default. Required paths such as `dataset_dir` and
`checkpoint` must still be supplied. Model-related fields are deliberately
`null`, and `model_options` defaults to `{}`, so selecting a different solver
does not inherit another architecture's options. For the default NAR solver,
constructor defaults are Qwen2.5-VL-3B-Instruct with LoRA, bfloat16, decoder
dimension 256, 2 set-encoder layers, 1 parallel decoder layer, 8 attention
heads, dropout 0.1 and column loss weight 1.0. Quantization is disabled.

Start with the default non-autoregressive solver using only the dataset path:

```sh
python Solver/train.py --dataset-dir Data/comix/6panel-ordering
```

Train the non-autoregressive solver from the repository root:

```sh
python Solver/train.py --solver qwen_vl_nonautoregressive --dataset-dir Data/comix/6panel-ordering --panel-count 6 --epochs 3 --batch-size 1 --grad-accum-steps 8 --save-dir Solver/checkpoints/qwen_nar
```

Use `--no-lora` for a frozen backbone feasibility run. Use
`--load-in-4bit --device-map auto` for QLoRA on a compatible CUDA/bitsandbytes installation.
Model constructor settings beyond the common CLI flags go into `--model-options`
as a JSON object (or a JSON object under `model_options` in `--config`). For
example, a lightweight text-only baseline:

```sh
python -m Solver.train --solver set2seq --dataset-dir Data/comix/6panel-ordering --panel-count 6 --model-options '{"solver_type":"text","use_layout":false,"d_model":64,"text_feature_dim":64,"encoder_layers":1,"decoder_layers":1,"num_heads":4}'
```

The same JSON mechanism accepts each architecture's constructor options: e.g.
`max_permutations`/`permutation_seed` for Jigsaw, `patch_size`/`dim`/`heads` for
Made2Order, `column_loss_weight` for NAR, or `base_checkpoint` for vector experts.
Unsupported constructor keys fail explicitly. CLI values override JSON defaults.
Common flags include `--model-name`, `--decoder-dim`, `--decoder-layers`,
`--encoder-layers`, `--num-heads`, `--torch-dtype`, `--use-layout`,
`--lr-backbone` and `--lr-decoder` (`--lr` alias).

For NAR-specific settings, copy `config/Solver/train.json` to a custom file
and edit its defaults, for example:

```json
{
  "defaults": {
    "solver": "qwen_vl_nonautoregressive",
    "dataset_dir": "Data/comix/6panel-ordering",
    "panel_count": 6,
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "epochs": 5,
    "model_options": {
      "encoder_layers": 2,
      "decoder_layers": 1,
      "decoder_dim": 256,
      "num_heads": 8,
      "column_loss_weight": 1.0,
      "max_pixels": 401408
    }
  }
}
```

```sh
python Solver/train.py --config config/Solver/train_nar_local.json --epochs 10
```

This example assumes the custom JSON was saved as `train_nar_local.json`.
Top-level model flags take precedence over matching keys in `model_options`.
For testing, leave architecture fields such as `model_name`, `decoder_dim`,
`encoder_layers`, `use_lora` and `use_layout` as `null`, and `model_options` as
`{}`; they are restored from the checkpoint. `torch_dtype`, `device_map` and
`load_in_4bit` may be overridden for evaluation. The image dimensions shown in
the shared test config are replaced by the checkpoint's training dimensions.

Evaluate a checkpoint; architecture and image dimensions are restored from it:

```sh
python Solver/test.py --solver qwen_vl_nonautoregressive --dataset-dir Data/comix/6panel-ordering --checkpoint Solver/checkpoints/qwen_nar/best.pt --split test --output-json Solver/checkpoints/qwen_nar/eval.json
```

Resume with the same settings and an increased total epoch count:

```sh
python Solver/train.py --solver qwen_vl_nonautoregressive --dataset-dir Data/comix/6panel-ordering --panel-count 6 --epochs 10 --resume Solver/checkpoints/qwen_nar/last.pt
```

Each run stores immutable `config.json`, `startup_config.json`, `run.log`,
`metrics.jsonl`, `best.pt`, `last.pt`, periodic epoch checkpoints and `test.json`.
Validation selects `best.pt`; test runs once after training using that selected
checkpoint. Without validation, the last checkpoint is evaluated. Checkpoints
store constructor options, task/schema, optimizer, best validation score and
Python/NumPy/Torch/CUDA random states. Gradient accumulation normalizes by the
actual number of samples, including partial final windows.

Earlier full-order or first-only checkpoints have different task semantics and
are rejected by the new entry points; retrain them for comparable endpoint
results. Vector experts require a new endpoint-trained stage-one checkpoint.

## Regression checks

```sh
python -m unittest discover -s Solver -p "test*.py"
python -m unittest discover -s experiment -p "test*.py"
```

Tests include full train/test/resume round trips on small models and the real
NAR projection/decoder with a substituted Qwen backbone. They do not download
or validate full GPU VLM weights.
