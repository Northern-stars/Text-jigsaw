# Qwen-VL non-autoregressive panel ordering solver

`Solver/agent/mangazero_qwen_vl_nonautoregressive.py` adds a second Qwen-VL
ordering head based on the non-autoregressive sentence-ordering implementation
under `third_party/nonautoregressive-sentence-ordering-main`.

The Qwen2.5-VL backbone receives all panels in one multimodal message. The task
description and all available OCR dialogue appear before the images, with
explicit input-slot labels before each image. This lets the causal language
layers' image-token states attend to the task and every panel's supplied text.
OCR IDs refer to shuffled input slots; they do not expose GT middle positions.
The `<|image_pad|>` token runs are mean-pooled into one vector per input panel,
projected into decoder space, and contextualised by a Transformer set encoder.
The decoder then receives K position queries at once and cross-attends to the
set memory. Its output is a `[batch, reading_position, input_panel]` pointer
matrix; there is no teacher-forced prefix and no autoregressive loop.

```python
from Solver.agent.mangazero_qwen_vl_nonautoregressive import (
    MangaZeroQwenVLNonAutoregressiveSolver,
    nonautoregressive_ordering_loss,
)

model = MangaZeroQwenVLNonAutoregressiveSolver(panel_count=6)
# Input slots 0 and K-1 hold the GT opening/ending panels.
logits = model(panel_images, target_order=target_order, dialog_texts=dialog_texts)
loss = nonautoregressive_ordering_loss(logits, target_order)
order = model.predict_order(panel_images, dialog_texts=dialog_texts)
```

The loss contains the usual position-to-panel cross entropy and an inverse
panel-to-position cross entropy, following the reference model's row/column
objective. `predict_order` performs row-wise greedy bipartite matching and
always returns a permutation. `trainable_state_dict` stores the projection,
set encoder, parallel decoder, and (when enabled) Qwen LoRA weights, so the
existing solver training/checkpoint conventions can be reused.

## Unified training

The ordering task now fixes both GT endpoints and predicts only the middle.
Both directions of the loss operate on `logits[:, 1:-1, 1:-1]`. All panels
remain visible to Qwen and the set encoder as context. `predict_order` keeps
0 first and K-1 last and matches only the middle candidates.

```sh
python Solver/train.py --solver qwen_vl_nonautoregressive --dataset-dir Data/comix/6panel-ordering --panel-count 6 --epochs 3 --grad-accum-steps 8
python Solver/test.py --solver qwen_vl_nonautoregressive --dataset-dir Data/comix/6panel-ordering --checkpoint Solver/checkpoints/qwen_vl_nonautoregressive/best.pt --split test
```

See `Solver/README.md` for checkpoint layout, model options and migration.
