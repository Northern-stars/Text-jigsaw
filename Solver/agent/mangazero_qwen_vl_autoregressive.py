"""Qwen2.5-VL + LoRA autoregressive pointer solver for MangaZero ordering.

The model embeds each input panel with a VLM backbone, then repeatedly selects
the most likely next panel from the remaining candidates with a pointer decoder.
This follows docs/qwen_vl_autoregressive_solver_spec.md.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


class MangaZeroQwenVLAutoregressiveSolver(nn.Module):
    """Use Qwen2.5-VL as a LoRA-tuned encoder and pointer-decode panel order."""

    def __init__(
        self,
        panel_count: int,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
        decoder_layers: int = 2,
        num_heads: int = 8,
        decoder_dim: int = 256,
        decoder_dropout: float = 0.1,
        load_in_4bit: bool = False,
        device_map: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if panel_count <= 1:
            raise ValueError("panel_count must be greater than 1")
        self.panel_count = panel_count
        self.model_name = model_name
        self.decoder_dim = decoder_dim

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

        # Map the VLM's pooled per-panel representation into decoder space.
        self.panel_projection = nn.Linear(hidden_size, decoder_dim)
        self.panel_norm = nn.LayerNorm(decoder_dim)

        # Per-panel type token and decoder BOS token, like the pointer baseline.
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pointer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.output_norm = nn.LayerNorm(decoder_dim)
        self.query_projection = nn.Linear(decoder_dim, decoder_dim)
        self.key_projection = nn.Linear(decoder_dim, decoder_dim)
        self._image_pad_token_id: int | None = None
        self._reset_parameters()

    # ------------------------------------------------------------------
    # Input encoding
    # ------------------------------------------------------------------

    def _call_processor(self, **forward_kwargs):
        """Call this processor across transformers versions.

        Newer Hugging Face processors require generation/image options like
        ``return_tensors`` to live inside ``processor_kwargs``; older releases
        accept them as top-level call kwargs. Try the new form first, then
        fall back to the legacy form.
        """
        processing_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        try:
            return self.processor(
                processor_kwargs=processing_kwargs,
                **forward_kwargs,
            )
        except TypeError:
            return self.processor(
                **processing_kwargs,
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

    def encode_panels(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
    ) -> Tensor:
        """Return a per-panel memory tensor [B, K, decoder_dim]."""
        inputs = self.prepare_inputs(panel_images, dialog_texts=dialog_texts)
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        memory = self._pool_per_panel(hidden, inputs)
        memory = memory.to(dtype=next(self.panel_projection.parameters()).dtype)
        memory = self.panel_projection(memory)
        memory = self.panel_norm(memory)
        return memory

    def _pool_per_panel(self, hidden: Tensor, inputs: dict[str, Tensor]) -> Tensor:
        """Aggregate VLM hidden states into one vector per input panel.

        Qwen2.5-VL inserts ``<|image_pad|>`` tokens for every image in message
        order. We locate those tokens in ``input_ids``, split their contiguous
        runs into per-panel segments, and mean-pool each segment.
        """
        input_ids = inputs["input_ids"]
        batch_size = hidden.size(0)
        image_token_id = self._get_image_pad_token_id()
        image_mask = input_ids == image_token_id
        if image_mask.sum().item() == 0:
            raise RuntimeError(
                "no <|image_pad|> tokens found; cannot pool per-panel "
                "representations"
            )

        dtype = hidden.dtype
        device = hidden.device
        pooled_rows = []
        for batch_index in range(batch_size):
            positions = image_mask[batch_index].nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                raise RuntimeError(f"sample {batch_index} has no image tokens")
            runs: list[list[int]] = []
            current_run = [int(positions[0])]
            for position in positions[1:].tolist():
                if position == current_run[-1] + 1:
                    current_run.append(position)
                else:
                    runs.append(current_run)
                    current_run = [position]
            runs.append(current_run)
            if len(runs) != self.panel_count:
                raise RuntimeError(
                    f"sample {batch_index} has {len(runs)} image token runs, "
                    f"expected {self.panel_count}"
                )
            panel_vectors = []
            for run in runs:
                panel_vectors.append(hidden[batch_index, run, :].mean(dim=0))
            pooled_rows.append(torch.stack(panel_vectors, dim=0))
        return torch.stack(pooled_rows, dim=0).to(dtype=dtype, device=device)

    def _get_image_pad_token_id(self) -> int:
        if self._image_pad_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("processor has no tokenizer")
            token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise RuntimeError("tokenizer does not map <|image_pad|> to a token id")
            self._image_pad_token_id = int(token_id)
        return self._image_pad_token_id

    # ------------------------------------------------------------------
    # Autoregressive pointer decoding
    # ------------------------------------------------------------------

    def forward(
        self,
        panel_images: Tensor,
        target_order: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None,
        memory: Optional[Tensor] = None,
    ) -> Tensor:
        """Return pointer logits [B, K, K] when training with teacher forcing,
        or the decoded order [B, K] when ``target_order`` is None."""
        if memory is None:
            memory = self.encode_panels(panel_images, dialog_texts=dialog_texts)
        if target_order is None:
            return self.greedy_decode(memory=memory)
        decoder_inputs = self._teacher_forcing_inputs(memory, target_order)
        causal_mask = torch.triu(
            torch.ones(decoder_inputs.size(1), decoder_inputs.size(1), device=memory.device, dtype=torch.bool),
            diagonal=1,
        )
        decoder_states = self.pointer_decoder(decoder_inputs, memory, tgt_mask=causal_mask)
        return self._pointer_logits(decoder_states, memory, target_order=target_order)

    @torch.no_grad()
    def predict_order(
        self,
        panel_images: Tensor,
        dialog_texts: Optional[list[list[str]]] = None,
        memory: Optional[Tensor] = None,
    ) -> Tensor:
        if memory is None:
            memory = self.encode_panels(panel_images, dialog_texts=dialog_texts)
        return self.greedy_decode(memory=memory)

    def greedy_decode(self, memory: Tensor) -> Tensor:
        """Iteratively select the next panel from the remaining candidates."""
        batch_size = memory.size(0)
        selected = torch.empty(batch_size, 0, dtype=torch.long, device=memory.device)

        for _ in range(self.panel_count):
            decoder_inputs = self._decode_inputs_from_selected(memory, selected)
            causal_mask = torch.triu(
                torch.ones(decoder_inputs.size(1), decoder_inputs.size(1), device=memory.device, dtype=torch.bool),
                diagonal=1,
            )
            decoder_states = self.pointer_decoder(decoder_inputs, memory, tgt_mask=causal_mask)
            logits = self._pointer_logits(decoder_states[:, -1:], memory).squeeze(1)
            if selected.numel() > 0:
                logits.scatter_(1, selected, -torch.inf)
            next_index = logits.argmax(dim=-1, keepdim=True)
            selected = torch.cat([selected, next_index], dim=1)

        return selected

    def _teacher_forcing_inputs(self, memory: Tensor, target_order: Tensor) -> Tensor:
        batch_size = memory.size(0)
        bos = self.bos_token.expand(batch_size, 1, -1)
        prefix = target_order[:, :-1]
        if prefix.numel() == 0:
            return bos
        selected_tokens = memory.gather(
            dim=1,
            index=prefix.unsqueeze(-1).expand(-1, -1, memory.size(-1)),
        )
        return torch.cat([bos, selected_tokens], dim=1)

    def _decode_inputs_from_selected(self, memory: Tensor, selected: Tensor) -> Tensor:
        batch_size = memory.size(0)
        bos = self.bos_token.expand(batch_size, 1, -1)
        if selected.numel() == 0:
            return bos
        selected_tokens = memory.gather(
            dim=1,
            index=selected.unsqueeze(-1).expand(-1, -1, memory.size(-1)),
        )
        return torch.cat([bos, selected_tokens], dim=1)

    def _pointer_logits(
        self,
        decoder_states: Tensor,
        memory: Tensor,
        target_order: Optional[Tensor] = None,
    ) -> Tensor:
        queries = self.query_projection(self.output_norm(decoder_states))
        keys = self.key_projection(memory)
        logits = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        if target_order is None:
            return logits

        selected_mask = torch.zeros_like(logits, dtype=torch.bool)
        for step in range(self.panel_count):
            if step > 0:
                previous = target_order[:, :step]
                selected_mask[:, step, :].scatter_(1, previous, True)
        return logits.masked_fill(selected_mask, -torch.inf)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def trainable_state_dict(self) -> dict[str, Tensor | dict[str, Tensor]]:
        state: dict[str, Tensor | dict[str, Tensor]] = {
            "decoder": {
                name: tensor.detach().cpu()
                for name, tensor in self.state_dict().items()
                if not name.startswith("backbone.")
            },
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
        if "decoder" in state_dict or "backbone" in state_dict:
            decoder_state = state_dict.get("decoder", {})
            if isinstance(decoder_state, dict) and decoder_state:
                decoder_keys = {
                    name: tensor for name, tensor in self.state_dict().items()
                    if not name.startswith("backbone.")
                }
                missing, unexpected = self._load_matching(decoder_keys, decoder_state)
                if missing:
                    raise RuntimeError(f"missing decoder keys: {sorted(missing)[:20]}")
                if unexpected:
                    raise RuntimeError(f"unexpected decoder keys: {sorted(unexpected)[:20]}")
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

    @staticmethod
    def _load_matching(
        target: dict[str, Tensor],
        source: dict[str, Tensor],
    ) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        unexpected: list[str] = []
        for name, tensor in target.items():
            if name in source:
                target[name].copy_(tensor)
            else:
                missing.append(name)
        for name in source:
            if name not in target:
                unexpected.append(name)
        return missing, unexpected

    def _input_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.bos_token, mean=0.0, std=0.02)


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


def pointer_cross_entropy(pointer_logits: Tensor, target_order: Tensor) -> Tensor:
    panel_count = pointer_logits.size(1)
    return F.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.size(-1)),
        target_order[:, :panel_count].reshape(-1),
    )
