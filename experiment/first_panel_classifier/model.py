"""Qwen2.5-VL hidden-state classifier for the first panel in reading order."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from Solver.agent.mangazero_qwen_vl_lora_classifier import (
    resolve_hidden_size,
    tensor_to_pil,
)
from Solver.agent.vlm_ordering_solver import parse_torch_dtype


def first_panel_targets(target_order: Tensor) -> Tensor:
    """Return the input-panel index having reading rank zero for each puzzle."""
    if target_order.ndim != 2:
        raise ValueError(f"target_order must be [batch, panel_count], got {tuple(target_order.shape)}")
    if target_order.size(1) < 2:
        raise ValueError("target_order must contain at least two panels")
    sorted_values = torch.sort(target_order, dim=1).values
    expected = torch.arange(target_order.size(1), device=target_order.device).expand_as(target_order)
    if not torch.equal(sorted_values, expected):
        raise ValueError("each target_order row must be a permutation of panel indices")
    return target_order.argmin(dim=1)


class QwenVLFirstPanelClassifier(nn.Module):
    """Pool a Qwen-VL puzzle hidden state and classify its first input panel."""

    def __init__(
        self,
        panel_count: int,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        revision: str | None = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        min_pixels: int | None = None,
        max_pixels: int | None = 512 * 28 * 28,
        train_backbone: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ),
        load_in_4bit: bool = False,
        device_map: str | None = None,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if panel_count <= 1:
            raise ValueError("panel_count must be greater than one")
        if load_in_4bit and not train_backbone:
            raise ValueError("--load-in-4bit is only supported with --train-backbone LoRA")
        self.panel_count = int(panel_count)
        self.model_name = model_name
        self.train_backbone = bool(train_backbone)

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("A recent transformers release with Qwen2.5-VL support is required") from exc

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": parse_torch_dtype(torch_dtype),
        }
        if revision is not None:
            processor_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if device_map:
            model_kwargs["device_map"] = device_map

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.backbone.config.use_cache = False
        if train_backbone:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            if load_in_4bit:
                self.backbone = prepare_model_for_kbit_training(self.backbone)
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, config)
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = resolve_hidden_size(self.backbone)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, self.panel_count),
        )
        output_embeddings = self.backbone.get_output_embeddings()
        output_device = (
            output_embeddings.weight.device
            if output_embeddings is not None
            else self.backbone.get_input_embeddings().weight.device
        )
        self.classifier.to(output_device)

    def forward(self, panel_images: Tensor, dialog_texts: list[list[str]] | None = None) -> Tensor:
        if panel_images.ndim != 5:
            raise ValueError(f"panel_images must have shape [B,K,3,H,W], got {tuple(panel_images.shape)}")
        batch_size, panel_count = panel_images.shape[:2]
        if panel_count != self.panel_count:
            raise ValueError(f"expected {self.panel_count} panels, received {panel_count}")
        conversations = []
        for batch_index in range(batch_size):
            content = [
                {"type": "image", "image": tensor_to_pil(panel_images[batch_index, panel_index])}
                for panel_index in range(panel_count)
            ]
            text_lines = []
            if dialog_texts is not None:
                for panel_index, text in enumerate(dialog_texts[batch_index]):
                    if text and str(text).strip():
                        text_lines.append(f"Panel {panel_index + 1} text: {str(text).strip()}")
            prompt = (
                f"This puzzle contains {panel_count} comic panels in a randomized input order. "
                "Identify which input panel is the first panel in the original reading sequence. "
                "Use all images and optional panel text as evidence."
            )
            if text_lines:
                prompt += "\n" + "\n".join(text_lines)
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])

        texts = [
            self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in conversations
        ]
        try:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(conversations)
            inputs = self._call_processor(text=texts, images=images, videos=videos)
        except ImportError:
            image_batches = [
                [part["image"] for part in conversation[0]["content"] if part["type"] == "image"]
                for conversation in conversations
            ]
            inputs = self._call_processor(text=texts, images=image_batches)
        inputs = {key: value.to(self._input_device()) for key, value in inputs.items() if torch.is_tensor(value)}

        context = torch.enable_grad() if self.train_backbone else torch.no_grad()
        with context:
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
        classifier_parameter = next(self.classifier.parameters())
        return self.classifier(pooled.to(device=classifier_parameter.device, dtype=classifier_parameter.dtype))

    def _call_processor(self, **kwargs: Any) -> Any:
        try:
            return self.processor(return_tensors="pt", padding=True, **kwargs)
        except TypeError:
            return self.processor(**kwargs)

    def _input_device(self) -> torch.device:
        embedding_device = self.backbone.get_input_embeddings().weight.device
        return embedding_device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "classifier": {key: value.detach().cpu() for key, value in self.classifier.state_dict().items()}
        }
        if self.train_backbone:
            from peft import get_peft_model_state_dict

            state["backbone_lora"] = {
                key: value.detach().cpu() for key, value in get_peft_model_state_dict(self.backbone).items()
            }
        return state

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        self.classifier.load_state_dict(state["classifier"])
        if self.train_backbone and "backbone_lora" in state:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.backbone, state["backbone_lora"])
