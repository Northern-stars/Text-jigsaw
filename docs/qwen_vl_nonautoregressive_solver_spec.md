# Qwen-VL non-autoregressive panel ordering solver

`Solver/agent/mangazero_qwen_vl_nonautoregressive.py` adds a second Qwen-VL
ordering head based on the non-autoregressive sentence-ordering implementation
under `third_party/nonautoregressive-sentence-ordering-main`.

The Qwen2.5-VL backbone receives all panels in one multimodal message. The
`<|image_pad|>` token runs are mean-pooled into one vector per input panel,
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
