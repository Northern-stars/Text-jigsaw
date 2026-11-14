# Comic ordering puzzles

All comic ordering training and evaluation entry points use
`order_remaining_given_first`, matching the human benchmark: provide the correct
opening panel and order only the remaining panels.

- The known opening panel occupies input slot **0** (Panel 1 in the human UI).
  Images, dialogue, panel records, and layout features are moved together.
- `panel_count` still counts **all** panels, including the given opening.
  Internal `target_order` and `predict_order` results remain full permutations
  beginning with 0. Online augmentation shuffles only the continuation.
- Set-to-sequence, Qwen pointer, and vector-expert decoders begin with slot 0
  selected and mask it from subsequent decisions. Pointer loss excludes the
  first position. Made2Order excludes the first panel and rank from its loss.
- Permutation classifiers and Jigsaw candidate banks enumerate only orders
  beginning with 0: a six-panel puzzle has at most **5! = 120** classes.
- Text-generating VLMs receive an explicit given-opening instruction and must
  output only the remaining IDs (normally 1 through K-1). The parser prepends 0
  for internal APIs. Training answer strings also omit the given panel.
- Exact match, position accuracy, pairwise accuracy, and the sequential
  environment reward evaluate only the continuation. The known start earns no
  free credit. For two-panel puzzles the sole remaining panel scores 1.0.

Existing dataset files need no regeneration. `load_panel_ordering_splits()`
enables this task by default. Direct `MangaZeroPanelOrderingDataset(...)` calls
must pass `given_first=True` when feeding a comic ordering model; the raw reader
defaults to False so first-panel classification and human benchmark preparation
can still use the original input indices. Direct model callers must provide the
opening panel in slot 0 and keep other modalities aligned. RQ1-RQ4 ordering
entry points use the same convention.

Training checkpoints and evaluation JSON identify the new task. Old full-order
classifier checkpoints have incompatible class definitions and require
retraining. Pointer/ranking backbone weights can be reused for initialization,
but old full-order scores are not comparable with continuation-only scores;
retrain or fine-tune and evaluate under this task. A capped Jigsaw bank with the
same shape but different permutations is explicitly rejected when loading.

Run the focused regression suite without downloading any VLM weights:

```sh
python -m unittest Solver.test_given_start Solver.test_human_benchmark Solver.test_panel_ordering_dataset_splits experiment.RQ_1.test_model_adapters
```
