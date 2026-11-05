# RQ1: Model Interfaces and Scaling

`model_adapters.py` maps every model family in the proposal to the shared `Solver.agent.vlm_ordering_solver.VLMOrderingSolver` protocol:

| Family | Adapter | Training status |
|---|---|---|
| `qwen2.5-vl` | Existing `QwenVLOrderingSolver` with `use_lora=False` | zero-shot and LoRA scaling |
| `internvl2.5` | Native `AutoModel.chat` adapter | zero-shot first; QLoRA only after a separate memory smoke test |
| `llama3.2-vision` | Transformers `MllamaForConditionalGeneration` adapter | inference first; training depends on 32GB validation |
| `openai` | OpenAI-compatible multimodal API | closed-source zero-shot reference |
| `gemini` | `google-genai` multimodal API | closed-source zero-shot reference |
| `claude` | Anthropic multimodal API | closed-source zero-shot reference |

The proposal's Made2Order, Qwen permutation-classifier, and Qwen autoregressive-pointer rows remain available through their existing task-specific solver classes. They are deliberately not registered as prompt adapters: they emit a tensor/class or pointer prediction and cannot be placed in the same direct/CoT/structured prompt protocol. Report them as architecture baselines with their existing train/eval entry points.

The adapter layer does not claim that every open model has passed the project's 32GB training smoke test. InternVL2.5 and Llama 3.2 Vision are inference adapters first; only Qwen2.5-VL currently has the repository's validated ordering LoRA training path. LoRA/QLoRA training on another family requires model-specific training, optimizer, and checkpoint work after the target hardware test.

Run a zero-shot model comparison with identical layout-blind inputs:

```powershell
python -m experiment.RQ_1.rq1_model_comparison `
  --manifest experiments/manifests/rq2_seed42.jsonl --split test `
  --model-family qwen2.5-vl --model-name Qwen/Qwen2.5-VL-7B-Instruct `
  --run-root experiments/rq1 --run-id qwen25vl_7b_seed42 --device-map auto
```

Replace `--model-family` and `--model-name` for other adapters. API adapters read `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`; do not place keys in `config.json`. Each run stores raw responses, parsed order, latency/provider metadata supplied by the SDK, metrics, and parse failures under `run.log`, `metrics.jsonl`, and `rq1_results.json`.

RQ1 scaling uses nested subsets from one fixed training pool (for example 5k/10k/20k/40k), identical model revision, style composition, optimizer budget definition, validation/test works, and multiple seeds. Use `experiment.RQ_2.rq2_style_balance` as the LoRA training implementation after adding the desired nested-budget schedule; the adapter protocol is shared by training-time Qwen evaluation and zero-shot model references.
