"""Japanese-to-English translation helpers for the Manga109 pipeline.

The Manga109 XML already provides the Japanese text, so no OCR is needed.
This module only translates that XML text into English, with an on-disk cache
to avoid repeated model calls for the same sentence.

Supported translation formats:

1. CTranslate2 (recommended)
   ``entai2965/sugoi-v4-ja-en-ctranslate2``
   Fast on CPU and GPU, low memory footprint, ideal for many short manga lines.

2. Hugging Face CausalLM
   ``sugoitoolkit/Sugoi-14B-Ultra-HF`` style chat models.
   Higher quality but needs a large GPU plus ``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_SUGOI_CT2_MODEL = "entai2965/sugoi-v4-ja-en-ctranslate2"
DEFAULT_SUGOI_HF_MODEL = "sugoitoolkit/Sugoi-14B-Ultra-HF"


class TranslationCache:
    """Persist translations so repeated sentences are translated only once."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path
        self.entries: dict[str, str] = {}
        if cache_path is not None and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = str(record.get("key", ""))
                    value = str(record.get("value", ""))
                    if key:
                        self.entries[key] = value
        self.handle = None

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def put(self, key: str, value: str) -> None:
        self.entries[key] = value
        if self.handle is not None:
            self.handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            self.handle.flush()

    def open(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.cache_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def resolve_model_source(
    model_name: str,
    local_files_only: bool = False,
    model_dir: Path | None = None,
) -> Path:
    """Resolve a HF repo id or local path into a local directory."""
    candidate = Path(model_name)
    if candidate.exists() and candidate.is_dir():
        return candidate

    target = Path(model_dir) if model_dir is not None else None
    if target is not None and target.exists() and any(target.iterdir()):
        return target

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=str(target) if target is not None else None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain translation model {model_name!r}.\n"
            "Check that the repository id exists on https://huggingface.co/models, "
            "or pass a local directory via --translation-model /path/to/dir.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(path)


def detect_format(path: Path) -> str:
    """Detect whether a local model dir is CTranslate2 or a HF causal LM."""
    if not path.is_dir():
        raise RuntimeError(f"Model path is not a directory: {path}")

    names = {p.name for p in path.iterdir()}
    if "model.bin" in names:
        return "ctranslate2"
    if any(name.endswith(".safetensors") for name in names):
        return "hf"
    if "model.safetensors.index.json" in names:
        return "hf"
    if "pytorch_model.bin" in names:
        return "hf"
    if "config.json" in names:
        return "hf"

    raise RuntimeError(
        f"Cannot detect the model format in {path}. "
        "Expected CTranslate2 (model.bin) or a HF causal LM (safetensors/config.json)."
    )


class CTranslate2SugoiTranslator:
    """Translate Japanese to English with a CTranslate2 Sugoi model."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUGOI_CT2_MODEL,
        device: str = "cpu",
        compute_type: str = "default",
        cache: TranslationCache | None = None,
        batch_size: int = 32,
        local_files_only: bool = False,
        model_dir: Path | None = None,
    ) -> None:
        try:
            import ctranslate2
        except ImportError as exc:
            raise ImportError(
                "ctranslate2 is required for the ct2 Sugoi translator. "
                "Install it with: pip install ctranslate2 sentencepiece"
            ) from exc

        self.ctranslate2 = ctranslate2
        self.cache = cache
        self.batch_size = max(1, int(batch_size))
        self.source = resolve_model_source(
            model_name,
            local_files_only=local_files_only,
            model_dir=model_dir,
        )

        fmt = detect_format(self.source)
        if fmt != "ctranslate2":
            raise RuntimeError(
                f"Model {model_name!r} is not a CTranslate2 model (detected {fmt}). "
                "Use --translation-engine sugoi-hf for Hugging Face causal LM models."
            )

        self.device = device
        if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            print("WARN: no CUDA device visible to ctranslate2; falling back to CPU")
            self.device = "cpu"

        if compute_type == "default":
            compute_type = "int8_float16" if self.device == "cuda" else "int8"

        self.translator = ctranslate2.Translator(
            str(self.source),
            device=self.device,
            compute_type=compute_type,
        )

    def translate_batch(self, texts: list[str]) -> list[str]:
        results = self.translator.translate_batch(
            texts,
            beam_size=4,
            max_batch_size=self.batch_size,
            max_decoding_length=256,
        )
        outputs: list[str] = []
        for result in results:
            outputs.append(result.hypotheses[0].strip() if result.hypotheses else "")
        return outputs


class HFCausalSugoiTranslator:
    """Translate Japanese to English with a HF causal LM Sugoi model."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUGOI_HF_MODEL,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache: TranslationCache | None = None,
        batch_size: int = 8,
        local_files_only: bool = False,
        model_dir: Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for the hf Sugoi translator") from exc

        self.torch = torch
        self.cache = cache
        self.batch_size = max(1, int(batch_size))

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("WARN: CUDA unavailable; falling back to CPU for translation")
            device = "cpu"

        source = resolve_model_source(
            model_name,
            local_files_only=local_files_only,
            model_dir=model_dir,
        )

        torch_dtype = getattr(torch, dtype) if dtype in ("bfloat16", "float16", "float32") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(source),
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
            local_files_only=True,
        )
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()

    def translate_batch(self, texts: list[str]) -> list[str]:
        torch = self.torch
        prompts = []
        for text in texts:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional Japanese-to-English manga translator. "
                        "Translate the given Japanese text into natural English. "
                        "Output only the translation, with no notes or explanations."
                    ),
                },
                {"role": "user", "content": text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results: list[str] = []
        for prompt_len, output in zip(
            [len(self.tokenizer(prompt)["input_ids"]) for prompt in prompts],
            outputs,
        ):
            generated = output[prompt_len:]
            results.append(
                self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            )
        return results


def build_translator(
    translation_engine: str,
    model_name: str,
    device: str,
    dtype: str,
    cache_path: Path | None,
    batch_size: int,
    local_files_only: bool = False,
    model_dir: Path | None = None,
    compute_type: str = "default",
) -> Any | None:
    """Build the requested translator, or return None when translation is off."""
    if translation_engine == "none":
        return None

    cache = TranslationCache(cache_path) if cache_path is not None else None
    if cache is not None:
        cache.open()

    try:
        if translation_engine == "sugoi-ct2":
            return CTranslate2SugoiTranslator(
                model_name=model_name or DEFAULT_SUGOI_CT2_MODEL,
                device=device,
                compute_type=compute_type,
                cache=cache,
                batch_size=batch_size,
                local_files_only=local_files_only,
                model_dir=model_dir,
            )
        if translation_engine in ("sugoi", "sugoi-hf"):
            return HFCausalSugoiTranslator(
                model_name=model_name or DEFAULT_SUGOI_HF_MODEL,
                device=device,
                dtype=dtype,
                cache=cache,
                batch_size=batch_size,
                local_files_only=local_files_only,
                model_dir=model_dir,
            )
    except Exception:
        if cache is not None:
            cache.close()
        raise

    if cache is not None:
        cache.close()
    raise ValueError(f"Unsupported translation engine: {translation_engine}")


def translate_text_map(
    text_map: dict[str, str],
    translator: Any | None,
) -> dict[str, str]:
    """Translate a frame_id -> Japanese text mapping into English."""
    if translator is None:
        return text_map

    result: dict[str, str] = {}
    untranslated: list[tuple[str, str]] = []
    for frame_id, text in text_map.items():
        source = (text or "").strip()
        if not source:
            result[frame_id] = ""
            continue
        if translator.cache is not None:
            cached = translator.cache.get(source)
            if cached is not None:
                result[frame_id] = cached
                continue
        untranslated.append((frame_id, source))

    if not untranslated:
        return result

    outputs: list[str] = []
    iterator = untranslated
    if tqdm is not None:
        iterator = tqdm(untranslated, desc="translating", unit="text", dynamic_ncols=True)

    batch: list[str] = []
    for _, source in iterator:
        batch.append(source)
        if len(batch) >= translator.batch_size:
            outputs.extend(translator.translate_batch(batch))
            batch = []
    if batch:
        outputs.extend(translator.translate_batch(batch))

    for (frame_id, source), translated in zip(untranslated, outputs):
        translated_text = translated.strip()
        result[frame_id] = translated_text
        if translator.cache is not None:
            translator.cache.put(source, translated_text)

    return result


def close_translator(translator: Any | None) -> None:
    """Close translator resources (cache handles)."""
    if translator is None:
        return
    cache = getattr(translator, "cache", None)
    if cache is not None:
        cache.close()