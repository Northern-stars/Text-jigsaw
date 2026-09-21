"""Qwen2.5-VL + LoRA permutation classifier for MangaZero ordering."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


class MangaZeroQwenVLLoRAClassifier(nn.Module):
    """Use Qwen2.5-VL as a LoRA-tuned backbone and a full classification head."""

    def __init__(
        self,
        panel_count: int,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_permutation_classes: int = 50000,
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        classifier_hidden_dim: int = 0,
        classifier_dropout: float = 0.1,
        load_in_4bit: bool = False,
        device_map: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if panel_count <= 1:
            raise ValueError("panel_count must be greater than 1")
        self.panel_count = panel_count
        self.model_name = model_name
        self.num_classes = self._num_permutations(panel_count, max_permutation_classes)

        class_orders = torch.tensor(
            list(itertools.permutations(range(panel_count))),
            dtype=torch.long,
        )
        self.register_buffer("class_orders", class_orders, persistent=True)
        self._class_index = {
            tuple(order.tolist()): index
            for index, order in enumerate(class_orders)
        }

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "transformers with Qwen2.5-VL support is required. "
                "Install a recent transformers release before using this solver."
            ) from exc

        processor_kwargs = {"trust_remote_code": trust_remote_code}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": parse_torch_dtype(torch_dtype),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("bitsandbytes quantization requires transformers BitsAndBytesConfig") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if device_map:
            model_kwargs["device_map"] = device_map

        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.backbone.config.use_cache = False

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            except ImportError as exc:
                raise ImportError("peft is required for LoRA fine-tuning this solver.") from exc
            if load_in_4bit:
                self.backbone = prepare_model_for_kbit_training(self.backbone)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = resolve_hidden_size(self.backbone)
        if classifier_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_size, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(classifier_hidden_dim, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_size, self.num_classes),
            )

    def forward(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        inputs = self.prepare_inputs(panel_images, dialog_texts=dialog_texts)
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            pooled = hidden[:, -1]
        else:
            indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        classifier_dtype = next(self.classifier.parameters()).dtype
        return self.classifier(pooled.to(dtype=classifier_dtype))

    def _call_processor(self, **forward_kwargs):
        """Call this processor across transformers versions.

        Newer Hugging Face processors merge top-level processing options such
        as ``return_tensors`` and ``padding`` through ``ProcessingKwargs``.
        Older releases accept them as plain call kwargs, so pass them at the
        top level and only drop them if the installed version rejects them.
        """
        processing_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        try:
            return self.processor(
                **processing_kwargs,
                **forward_kwargs,
            )
        except TypeError:
            return self.processor(
                **forward_kwargs,
            )

    def prepare_inputs(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> dict[str, Tensor]:
        if panel_images.ndim != 5:
            raise ValueError(f"panel_images must have shape [B,K,3,H,W], got {tuple(panel_images.shape)}")
        batch_size, panel_count = panel_images.shape[:2]
        if panel_count != self.panel_count:
            raise ValueError(f"expected panel_count={self.panel_count}, got {panel_count}")
        conversations = []
        for batch_index in range(batch_size):
            content = []
            for panel_index in range(panel_count):
                content.append(
                    {
                        "type": "image",
                        "image": tensor_to_pil(panel_images[batch_index, panel_index]),
                    }
                )
            prompt = build_ordering_prompt(panel_count, dialog_texts[batch_index] if dialog_texts else None)
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])

        try:
            from qwen_vl_utils import process_vision_info

            texts = [
                self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for conversation in conversations
            ]
            images, videos = process_vision_info(conversations)
            inputs = self._call_processor(
                text=texts,
                images=images,
                videos=videos,
            )
        except ImportError:
            try:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": True},
                )
            except TypeError:
                try:
                    inputs = self.processor.apply_chat_template(
                        conversations,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    )
                except TypeError:
                    texts = [
                    self.processor.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for conversation in conversations
                ]
                images = [
                    [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
                    for conversation in conversations
                ]
                inputs = self._call_processor(
                    text=texts,
                    images=images,
                )
        return {
            key: value.to(self._input_device())
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

    def target_to_class_indices(self, target_order: Tensor) -> Tensor:
        if target_order.ndim != 2 or target_order.size(1) != self.panel_count:
            raise ValueError(f"target_order must have shape [B,{self.panel_count}], got {tuple(target_order.shape)}")
        indices = []
        for order in target_order.detach().cpu().tolist():
            key = tuple(int(value) for value in order)
            try:
                indices.append(self._class_index[key])
            except KeyError as exc:
                raise ValueError(f"target_order is not a valid permutation: {order}") from exc
        return torch.tensor(indices, dtype=torch.long, device=target_order.device)

    @torch.no_grad()
    def predict_order(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        logits = self(panel_images, dialog_texts=dialog_texts)
        class_ids = logits.argmax(dim=-1)
        return self.class_orders.to(device=class_ids.device)[class_ids]

    def trainable_state_dict(self) -> dict[str, Tensor | dict[str, Tensor]]:
        state: dict[str, Tensor | dict[str, Tensor]] = {
            "classifier": {
                name: tensor.detach().cpu()
                for name, tensor in self.classifier.state_dict().items()
            },
            "class_orders": self.class_orders.detach().cpu(),
        }
        try:
            from peft import get_peft_model_state_dict
        except ImportError:
            state["backbone"] = {
                name: tensor.detach().cpu()
                for name, tensor in self.backbone.state_dict().items()
                if "lora_" in name
            }
        else:
            state["backbone"] = {
                name: tensor.detach().cpu()
                for name, tensor in get_peft_model_state_dict(self.backbone).items()
            }
        return state

    def load_trainable_state_dict(self, state_dict: dict[str, Tensor | dict[str, Tensor]]) -> None:
        if "classifier" in state_dict or "backbone" in state_dict:
            classifier_state = state_dict.get("classifier", {})
            if isinstance(classifier_state, dict):
                self.classifier.load_state_dict(classifier_state, strict=True)
            backbone_state = state_dict.get("backbone", {})
            if isinstance(backbone_state, dict) and backbone_state:
                try:
                    from peft import set_peft_model_state_dict
                except ImportError:
                    self.backbone.load_state_dict(backbone_state, strict=False)
                else:
                    set_peft_model_state_dict(self.backbone, backbone_state)
            return
        self.load_state_dict(state_dict, strict=False)

    def _input_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @staticmethod
    def _num_permutations(panel_count: int, max_permutation_classes: int) -> int:
        total = 1
        for value in range(2, panel_count + 1):
            total *= value
        if total > max_permutation_classes:
            raise ValueError(
                f"panel_count={panel_count} creates {total} classes, exceeding "
                f"max_permutation_classes={max_permutation_classes}"
            )
        return total


def build_ordering_prompt(panel_count: int, dialog_texts: list[str] | None = None) -> str:
    panel_ids = ", ".join(str(index) for index in range(panel_count))
    prompt = (
        "You are solving a manga panel ordering puzzle. "
        f"There are {panel_count} input panels numbered {panel_ids}. "
        "Use visual continuity, reading flow, characters, dialog bubbles, and scene transitions "
        "to infer the correct chronological reading order. "
        "Do not generate the order; provide an internal representation for classification."
    )
    if dialog_texts:
        text_lines = [
            f"panel {index}: {str(text).strip()}"
            for index, text in enumerate(dialog_texts)
            if str(text).strip()
        ]
        if text_lines:
            prompt += "\nOCR text:\n" + "\n".join(text_lines)
    return prompt


def parse_torch_dtype(raw_dtype: str) -> torch.dtype | str:
    if raw_dtype == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[raw_dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {raw_dtype}") from exc


def resolve_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size)
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        return resolve_hidden_size(base_model)
    raise ValueError("Unable to infer Qwen hidden size from model config")


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def qwen_vl_cross_entropy(
    logits: Tensor,
    target_order: Tensor,
    model: MangaZeroQwenVLLoRAClassifier,
) -> Tensor:
    class_targets = model.target_to_class_indices(target_order)
    return F.cross_entropy(logits, class_targets)
