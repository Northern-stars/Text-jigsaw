"""Qwen2.5-VL LoRA solver that generates the panel ordering as text.

Unlike the local-decoder solvers (``mangazero_qwen_vl_autoregressive.py``,
``mangazero_qwen_vl_lora_classifier.py``) this module adds no head: the VLM
itself answers the puzzle. ``prompt_generator`` builds an instruction prompt
that asks for a comma-separated panel order, ``vlm_infer`` decodes it, and
``answer_generator`` parses the text back into a permutation.

Training fine-tunes the backbone with LoRA on the instruction followed by the
ground-truth order rendered as a string, e.g. ``2, 0, 1, 3``. The loss is the
standard causal-LM cross entropy over the answer tokens.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from Solver.agent.vlm_ordering_solver import (
    QwenVLInfer,
    VLMOrderingSolver,
    build_ordering_messages,
    parse_torch_dtype,
    to_pil,
)


def format_order_label(order: Sequence[int], *, start_index: int = 0) -> str:
    """Render a panel order as the training label string, e.g. ``2, 0, 1, 3``."""
    return ", ".join(str(start_index + int(index)) for index in order)


def common_prefix_length(prefix: Tensor, full: Tensor) -> int:
    """Length of the shared token prefix between the prompt and the full chat.

    Labels are masked with this length so only the assistant answer contributes
    to the loss. Comparing tokens (rather than trusting the shorter length)
    keeps the mask correct even if the chat template encodes the trailing
    assistant turn differently between the two encodings.
    """
    limit = min(int(prefix.numel()), int(full.numel()))
    matches = 0
    while matches < limit and int(prefix[matches]) == int(full[matches]):
        matches += 1
    return matches


class QwenVLOrderingSolver(VLMOrderingSolver):
    """Qwen2.5-VL + LoRA that directly generates the ordering text.

    Inherits the prompt/infer/answer pipeline from :class:`VLMOrderingSolver`
    and implements :meth:`train` by LoRA fine-tuning the backbone with a causal
    LM objective. The single trainable component is the language model; no
    ordering head is added.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        panel_count: Optional[int] = None,
        *,
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        min_pixels: int | None = None,
        max_pixels: int | None = 512 * 28 * 28,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        load_in_4bit: bool = False,
        device_map: str | None = None,
        trust_remote_code: bool = True,
        start_index: int = 0,
        max_new_tokens: int = 64,
        generation_kwargs: Optional[dict[str, Any]] = None,
        strict_answer: bool = True,
        prompt_generator: Any = None,
        answer_generator: Any = None,
    ) -> None:
        self.model_name = model_name
        self.start_index = int(start_index)
        self.max_new_tokens = int(max_new_tokens)
        self.strict_answer = bool(strict_answer)

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "transformers with Qwen2.5-VL support is required. "
                "Install a recent transformers release before using this solver."
            ) from exc

        processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        model_kwargs: dict[str, Any] = {
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

        super().__init__(
            vlm_infer=QwenVLInfer(
                self.backbone,
                self.processor,
                max_new_tokens=self.max_new_tokens,
                generation_kwargs=generation_kwargs,
            ),
            prompt_generator=prompt_generator or self._default_prompt_generator,
            answer_generator=answer_generator,
            panel_count=panel_count,
            one_indexed=bool(self.start_index),
            strict_answer=self.strict_answer,
        )

    # ------------------------------------------------------------------
    # Prompt / message construction
    # ------------------------------------------------------------------

    def _default_prompt_generator(self, panels: Sequence[Any], texts: Any) -> list[dict[str, Any]]:
        return build_ordering_messages(panels, texts, start_index=self.start_index)

    def _chat_messages(
        self,
        panels: Sequence[Any],
        texts: Any,
        order: Optional[Sequence[int]] = None,
    ) -> list[dict[str, Any]]:
        """Build chat messages; append the assistant answer when ``order`` is given."""
        messages = self.prompt_generator(panels, texts)
        if order is not None:
            messages = messages + [
                {"role": "assistant", "content": format_order_label(order, start_index=self.start_index)}
            ]
        return messages

    def _encode_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert chat messages into processor inputs (full conversation)."""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        try:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        except ImportError:
            inputs = self.processor(text=[text], return_tensors="pt")
        return inputs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
        target_order: Optional[Tensor] = None,
    ) -> Tensor:
        """Causal-LM loss on the answer tokens after the ordering instruction."""
        if target_order is None:
            raise ValueError("target_order is required for training")
        batch_size = panel_images.shape[0]
        losses = []
        for batch_index in range(batch_size):
            panel_count = panel_images.shape[1]
            panels = [to_pil(panel_images[batch_index, index]) for index in range(panel_count)]
            texts = dialog_texts[batch_index] if dialog_texts is not None else None
            order = target_order[batch_index].detach().cpu().tolist()

            prompt_inputs = self._encode_messages(self.prompt_generator(panels, texts))
            full_inputs = self._encode_messages(self._chat_messages(panels, texts, order))

            prompt_len = common_prefix_length(
                prompt_inputs["input_ids"][0],
                full_inputs["input_ids"][0],
            )
            labels = full_inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            outputs = self.backbone(**full_inputs, labels=labels)
            losses.append(outputs.loss)
        return torch.stack(losses).mean()

    def train(
        self,
        batches: Any,
        *,
        epochs: int = 1,
        optimizer: Any = None,
        device: Any = None,
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 0.0,
        log_every: int = 0,
    ) -> dict[str, float]:
        """LoRA fine-tune the VLM to emit the ordering string.

        ``batches`` is an iterable of batches (or a callable returning such an
        iterable per epoch). Each batch must provide ``panel_images`` and
        ``target_order``, and optionally ``dialog_texts``. Returns
        ``{"train_loss": mean_loss, "steps": n}``.
        """
        if device is None:
            device = next(self.backbone.parameters()).device
        if optimizer is None:
            params = [p for p in self.backbone.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=1e-4)

        was_training = self.backbone.training
        self.backbone.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for epoch in range(epochs):
            epoch_batches = batches(epoch) if callable(batches) else batches
            for batch in epoch_batches:
                panel_images = batch["panel_images"].to(device)
                target_order = batch["target_order"].to(device)
                dialog_texts = batch.get("dialog_texts")
                loss = self.compute_loss(panel_images, dialog_texts, target_order) / grad_accum_steps
                loss.backward()
                step += 1
                if step % grad_accum_steps == 0:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.item() * grad_accum_steps))
                if log_every and step % log_every == 0:
                    print(f"epoch {epoch + 1} step {step} loss {losses[-1]:.4f}", flush=True)
        if step % grad_accum_steps != 0:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not was_training:
            self.backbone.eval()
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        return {"train_loss": mean_loss, "steps": float(step)}

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    # ------------------------------------------------------------------
    # Inference override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, panels: Sequence[Any], texts: Any = None):  # type: ignore[override]
        """Generate the ordering text and parse it into a permutation."""
        self.backbone.eval()
        return super().solve(panels, texts)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def trainable_state_dict(self) -> dict[str, Any]:
        """Return the LoRA adapter weights (what training actually updates)."""
        try:
            from peft import get_peft_model_state_dict

            backbone_state = {
                name: tensor.detach().cpu()
                for name, tensor in get_peft_model_state_dict(self.backbone).items()
            }
        except ImportError:
            backbone_state = {
                name: tensor.detach().cpu()
                for name, tensor in self.backbone.state_dict().items()
                if "lora_" in name
            }
        return {"backbone": backbone_state, "model_name": self.model_name}

    def load_trainable_state_dict(self, state_dict: dict[str, Any]) -> None:
        backbone_state = state_dict.get("backbone", state_dict)
        if isinstance(backbone_state, dict) and backbone_state:
            try:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(self.backbone, backbone_state)
            except ImportError:
                self.backbone.load_state_dict(backbone_state, strict=False)
