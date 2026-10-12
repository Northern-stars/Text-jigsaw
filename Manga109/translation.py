"""Japanese-to-English translation helpers for the Manga109 pipeline.

The Manga109 XML already provides the Japanese text, so no OCR is needed.
This module only translates that XML text into English, with an on-disk cache
to avoid repeated model calls for the same sentence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_SUGOI_MODEL = "mHagos/sugoi-4-japanese-to-english-translator"


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


class SugoiTranslator:
    """Translate Japanese text to English with a local Sugoi v4 model."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUGOI_MODEL,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache: TranslationCache | None = None,
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for Sugoi translation") from exc

        self.torch = torch
        self.cache = cache
        self.batch_size = max(1, int(batch_size))

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("WARN: CUDA unavailable; falling back to CPU for translation")
            device = "cpu"

        torch_dtype = getattr(torch, dtype) if dtype in ("bfloat16", "float16", "float32") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        torch = self.torch
        prompts = []
        for text in texts:
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional Japanese-to-English manga translator. "
                    "Translate the given Japanese text into natural English. "
                    "Output only the translation, with no notes or explanations.",
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
) -> SugoiTranslator | None:
    """Build the requested translator, or return None when translation is off."""
    if translation_engine == "none":
        return None
    if translation_engine == "sugoi":
        cache = TranslationCache(cache_path) if cache_path is not None else None
        return SugoiTranslator(
            model_name=model_name,
            device=device,
            dtype=dtype,
            cache=cache,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported translation engine: {translation_engine}")


def translate_text_map(
    text_map: dict[str, str],
    translator: SugoiTranslator | None,
    progress: Any | None = None,
) -> dict[str, str]:
    """Translate a frame_id -> Japanese text mapping into English."""
    if translator is None:
        return text_map

    untranslated: list[tuple[str, str]] = []
    result: dict[str, str] = {}
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

    texts = [text for _, text in untranslated]
    iterator = untranslated
    if tqdm is not None:
        iterator = tqdm(untranslated, desc="translating", unit="text", dynamic_ncols=True)

    pending: list[tuple[str, str]] = []
    outputs: list[str] = []
    for frame_id, text in iterator:
        pending.append((frame_id, text))
        if len(pending) >= translator.batch_size:
            batch_outputs = translator._translate_batch([t for _, t in pending])
            outputs.extend(batch_outputs)
            pending = []
    if pending:
        outputs.extend(translator._translate_batch([t for _, t in pending]))

    for (frame_id, source), translated in zip(untranslated, outputs):
        translated_text = translated.strip()
        result[frame_id] = translated_text
        if translator.cache is not None:
            translator.cache.put(source, translated_text)

    return result