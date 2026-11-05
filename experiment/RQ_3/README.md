# RQ3: Evidence Modalities

`rq3_evidence_modalities.py` evaluates the same held-out puzzles under `visual_only`, `text_only`, and `visual_text`. All inputs are independent panels with randomized display order and no layout metadata. Generate the visual-only copy from the ordinary manifest with the included geometry-aware masking utility:

```powershell
python -m experiment.RQ_3.make_masked_manifest `
  --manifest experiments/manifests/rq2_seed42.jsonl `
  --output-dir experiments/manifests/rq3_visual_masked_data `
  --output-manifest experiments/manifests/rq3_visual_masked.jsonl
```

The runner records this protocol in `config.json` and does not perform an unverified mask in memory.

Use the same trained Qwen2.5-VL checkpoint and decoding configuration for all three conditions:

```powershell
python -m experiment.RQ_3.rq3_evidence_modalities `
  --visual-manifest experiments/manifests/rq3_visual_masked.jsonl `
  --text-manifest experiments/manifests/rq3_text.jsonl `
  --multimodal-manifest experiments/manifests/rq3_multimodal.jsonl `
  --checkpoint experiments/rq2/mixed_seed42/checkpoints/best.pt `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct --model-revision MODEL_COMMIT `
  --run-root experiments/rq3 --run-id mixed_seed42 --load-in-4bit --device-map auto
```

Every prediction is written to `metrics.jsonl`; aggregate metrics and parse failures are written to `rq3_results.json`. Use paired sequence IDs when comparing conditions.
