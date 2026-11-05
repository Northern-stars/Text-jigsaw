# RQ4: Prompting Strategies

This runner keeps the trained Qwen2.5-VL adapter, puzzle order, image inputs, maximum output tokens, decoding parameters, and parser budget fixed. It compares `direct`, `cot`, and `structured` prompts. The CoT condition requests private reasoning because hidden reasoning is not scored as an observable explanation; the structured condition requires observable evidence and an `ORDER:` line so evidence text cannot corrupt permutation parsing.

```powershell
python -m experiment.RQ_4.rq4_prompting `
  --manifest experiments/manifests/rq2_seed42.jsonl --split test `
  --checkpoint experiments/rq2/mixed_seed42/checkpoints/best.pt `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT `
  --run-root experiments/rq4 --run-id mixed_seed42 --load-in-4bit --device-map auto
```

`metrics.jsonl` stores raw responses, parsed permutations, validity, and error categories (`local_adjacent_swap`, `boundary_or_transition`, `other_permutation_error`, and `invalid_output`). Aggregate results are in `rq4_results.json`.
