"""Model adapters exposing the shared :class:`VLMOrderingSolver` protocol.

Adapters load optional local-model/API dependencies only when instantiated. All
backends return the same ``VLMOrderingResult`` and use the shared panel-order
parser, so RQ1 metrics do not depend on a provider-specific response format.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Sequence

from PIL import Image

from Solver.agent.vlm_ordering_solver import (
    VLMOrderingSolver,
    parse_torch_dtype,
    to_pil,
)


class Qwen25VLAdapter:
    """Qwen2.5-VL adapter using the existing Qwen solver implementation."""

    def __new__(cls, model_name: str, **kwargs: Any) -> VLMOrderingSolver:
        from Solver.agent.qwen_vl_ordering_solver import QwenVLOrderingSolver

        kwargs.setdefault("use_lora", False)
        return QwenVLOrderingSolver(model_name=model_name, **kwargs)


class HuggingFaceVisionAdapter(VLMOrderingSolver):
    """Generic Transformers image-to-text adapter for Llama 3.2 Vision."""

    def __init__(self, model_name: str, *, panel_count: int | None = None, max_new_tokens: int = 64, **kwargs: Any) -> None:
        try:
            import torch
            from transformers import AutoProcessor, MllamaForConditionalGeneration
        except ImportError as exc:
            raise ImportError("Transformers with Mllama support is required for the Llama Vision adapter") from exc
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, **{k: v for k, v in kwargs.items() if k in {"revision", "trust_remote_code"}})
        model_kwargs = {"torch_dtype": parse_torch_dtype(kwargs.get("torch_dtype", "auto"))}
        if kwargs.get("revision"):
            model_kwargs["revision"] = kwargs["revision"]
        if kwargs.get("device_map"):
            model_kwargs["device_map"] = kwargs["device_map"]
        self.model = MllamaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.max_new_tokens = int(max_new_tokens)
        self.device = next(self.model.parameters()).device
        super().__init__(vlm_infer=self._infer, panel_count=panel_count)

    def _infer(self, messages: Any) -> str:
        images, text = [], ""
        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "image":
                    images.append(to_pil(content["image"]))
                elif content.get("type") == "text":
                    text += content["text"]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=images or None, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


class InternVL25Adapter(VLMOrderingSolver):
    """InternVL2.5 adapter using its native ``chat`` interface."""

    def __init__(self, model_name: str, *, panel_count: int | None = None, max_new_tokens: int = 64, **kwargs: Any) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Transformers is required for the InternVL2.5 adapter") from exc
        self._torch = torch
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if kwargs.get("revision"):
            model_kwargs["revision"] = kwargs["revision"]
        if kwargs.get("device_map"):
            model_kwargs["device_map"] = kwargs["device_map"]
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        if kwargs.get("revision"):
            tokenizer_kwargs["revision"] = kwargs["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = int(max_new_tokens)
        super().__init__(vlm_infer=self._infer, panel_count=panel_count)

    def _infer(self, messages: Any) -> str:
        images = [to_pil(item["image"]) for message in messages for item in message.get("content", []) if item.get("type") == "image"]
        text_parts = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image":
                    text_parts.append("<image>")
                elif item.get("type") == "text":
                    text_parts.append(item["text"])
        text = "\n".join(text_parts)
        if not images:
            response = self.model.chat(self.tokenizer, None, text, dict(max_new_tokens=self.max_new_tokens, do_sample=False))
        else:
            pixel_values, num_patches_list = self._preprocess_images(images)
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                text,
                dict(max_new_tokens=self.max_new_tokens, do_sample=False),
                num_patches_list=num_patches_list,
            )
        return response[0] if isinstance(response, tuple) else str(response)

    def _preprocess_images(self, images: Sequence[Image.Image]) -> tuple[Any, list[int]]:
        try:
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:
            raise ImportError("torchvision is required for InternVL image preprocessing") from exc
        from PIL import Image as PILImage

        def dynamic_preprocess(image: Image.Image, *, image_size: int = 448, max_num: int = 6) -> list[Image.Image]:
            width, height = image.size
            aspect_ratio = width / height
            candidates = sorted({(cols, rows) for n in range(1, max_num + 1) for cols in range(1, n + 1) for rows in range(1, n + 1) if 1 <= cols * rows <= max_num}, key=lambda x: x[0] * x[1])
            target = min(candidates, key=lambda grid: abs(aspect_ratio - grid[0] / grid[1]))
            resized = image.resize((image_size * target[0], image_size * target[1]), PILImage.Resampling.BICUBIC)
            tiles = [resized.crop((col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)) for row in range(target[1]) for col in range(target[0])]
            return tiles

        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image_tiles = [dynamic_preprocess(image) for image in images]
        pixel_values = [transform(tile) for tiles in image_tiles for tile in tiles]
        dtype = next(self.model.parameters()).dtype
        return self._torch.stack(pixel_values).to(device=self.device, dtype=dtype), [len(tiles) for tiles in image_tiles]


def _data_url(image: Any) -> str:
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAICompatibleAdapter(VLMOrderingSolver):
    """OpenAI-compatible multimodal API adapter (OpenAI GPT-4.1/GPT-4o)."""

    def __init__(self, model_name: str, *, panel_count: int | None = None, max_new_tokens: int = 64, api_key: str | None = None, base_url: str | None = None, **_: Any) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package for the OpenAI adapter") from exc
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.last_metadata: dict[str, Any] = {"provider": "openai", "model": model_name}
        super().__init__(vlm_infer=self._infer, panel_count=panel_count)

    def _infer(self, messages: Any) -> str:
        converted = []
        for message in messages:
            content = []
            for item in message.get("content", []):
                content.append({"type": "text", "text": item["text"]} if item.get("type") == "text" else {"type": "image_url", "image_url": {"url": _data_url(item["image"])}})
            converted.append({"role": message["role"], "content": content})
        response = self.client.chat.completions.create(model=self.model_name, messages=converted, temperature=0, max_tokens=self.max_new_tokens)
        self.last_metadata = {
            "provider": "openai",
            "model": self.model_name,
            "request_id": getattr(response, "id", None),
            "usage": getattr(getattr(response, "usage", None), "model_dump", lambda: None)(),
        }
        return response.choices[0].message.content or ""


class GeminiAdapter(VLMOrderingSolver):
    """Google Gemini multimodal API adapter."""

    def __init__(self, model_name: str, *, panel_count: int | None = None, max_new_tokens: int = 64, api_key: str | None = None, **_: Any) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install google-genai for the Gemini adapter") from exc
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.last_metadata: dict[str, Any] = {"provider": "google", "model": model_name}
        super().__init__(vlm_infer=self._infer, panel_count=panel_count)

    def _infer(self, messages: Any) -> str:
        parts = []
        for message in messages:
            for item in message.get("content", []):
                parts.append(to_pil(item["image"]) if item.get("type") == "image" else item["text"])
        response = self.client.models.generate_content(model=self.model_name, contents=parts, config={"temperature": 0, "max_output_tokens": self.max_new_tokens})
        usage = getattr(response, "usage_metadata", None)
        self.last_metadata = {"provider": "google", "model": self.model_name, "usage": getattr(usage, "__dict__", None)}
        return response.text or ""


class ClaudeAdapter(VLMOrderingSolver):
    """Anthropic Claude multimodal API adapter."""

    def __init__(self, model_name: str, *, panel_count: int | None = None, max_new_tokens: int = 64, api_key: str | None = None, **_: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic for the Claude adapter") from exc
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.last_metadata: dict[str, Any] = {"provider": "anthropic", "model": model_name}
        super().__init__(vlm_infer=self._infer, panel_count=panel_count)

    def _infer(self, messages: Any) -> str:
        content = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "text":
                    content.append({"type": "text", "text": item["text"]})
                else:
                    encoded = _data_url(item["image"]).split(",", 1)[1]
                    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}})
        response = self.client.messages.create(model=self.model_name, max_tokens=self.max_new_tokens, temperature=0, messages=[{"role": "user", "content": content}])
        usage = getattr(response, "usage", None)
        self.last_metadata = {
            "provider": "anthropic",
            "model": self.model_name,
            "request_id": getattr(response, "id", None),
            "usage": getattr(usage, "model_dump", lambda: None)(),
        }
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


MODEL_ADAPTERS = {
    "qwen2.5-vl": Qwen25VLAdapter,
    "internvl2.5": InternVL25Adapter,
    "llama3.2-vision": HuggingFaceVisionAdapter,
    "openai": OpenAICompatibleAdapter,
    "gemini": GeminiAdapter,
    "claude": ClaudeAdapter,
}


def create_adapter(model_family: str, model_name: str, **kwargs: Any) -> VLMOrderingSolver:
    key = model_family.strip().lower()
    if key not in MODEL_ADAPTERS:
        raise ValueError(f"Unknown model family {model_family!r}; choose from {sorted(MODEL_ADAPTERS)}")
    if key == "llama3.2-vision":
        kwargs.setdefault("model_name", model_name)
        return MODEL_ADAPTERS[key](**kwargs)
    return MODEL_ADAPTERS[key](model_name=model_name, **kwargs)
