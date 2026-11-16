"""Qwen2.5-VL set encoder with a non-autoregressive ordering decoder.

The decoder follows the design in ``third_party/nonautoregressive-sentence-
ordering-main``: all reading positions are represented by learned queries and
decoded in parallel.  A row pointer score predicts the panel at each position;
an auxiliary column loss makes the scores closer to a one-to-one assignment.
The Qwen backbone and per-panel image pooling are shared with the existing
Qwen-VL pointer solver, but no causal decoding loop is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from Solver.agent.mangazero_qwen_vl_autoregressive import (
    parse_torch_dtype,
    resolve_hidden_size,
    tensor_to_pil,
)


def validate_order(target_order: Tensor, panel_count: int | None = None) -> None:
    """Validate endpoint-conditioned input-slot permutations via the puzzle."""
    from Solver.puzzle import validate_order as validate_endpoints
    validate_endpoints(target_order)
    if panel_count is not None and target_order.size(1) != panel_count:
        raise ValueError(f"expected panel_count={panel_count}, got {target_order.size(1)}")


class NonAutoregressivePointerDecoder(nn.Module):
    """Parallel position decoder and bipartite panel pointer head.

    ``memory`` is an unordered set of panel vectors ``[B,K,D]``.  Every output
    position receives a learned query, so all K positions are decoded in one
    Transformer call.  ``forward`` returns scores of shape ``[B,K,K]`` where
    the second axis is reading position and the last axis is input panel.
    """

    def __init__(
        self,
        panel_count: int,
        d_model: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if panel_count < 3:
            raise ValueError("panel_count must be at least 3")
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if encoder_layers < 1 or decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        self.panel_count = int(panel_count)
        self.d_model = int(d_model)
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.position_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.position_queries = nn.Parameter(torch.empty(1, panel_count, d_model))
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.score_projection = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def forward(self, memory: Tensor) -> Tensor:
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [B,K,D], got {tuple(memory.shape)}")
        if memory.size(1) != self.panel_count or memory.size(2) != self.d_model:
            raise ValueError(
                f"expected memory [B,{self.panel_count},{self.d_model}], got {tuple(memory.shape)}"
            )
        encoded = self.set_encoder(memory)
        queries = self.position_queries.expand(memory.size(0), -1, -1)
        decoded = self.position_decoder(queries, encoded)
        query = self.query_projection(self.output_norm(decoded)).unsqueeze(2)
        # Keep panel identity in the pointer keys, while the decoder's
        # cross-attention sees the contextualised set representation.
        key = self.key_projection(memory).unsqueeze(1)
        scores = self.score_projection(torch.tanh(query + key)).squeeze(-1)
        return scores

    def predict_order(self, memory: Tensor) -> Tensor:
        """Decode a permutation with row-wise greedy bipartite matching."""
        scores = self(memory)
        selected = torch.zeros(
            scores.size(0), self.panel_count, dtype=torch.long, device=scores.device
        )
        available = torch.ones(
            scores.size(0), self.panel_count, dtype=torch.bool, device=scores.device
        )
        selected[:, -1] = self.panel_count - 1
        available[:, 0] = False
        available[:, -1] = False
        for position in range(1, self.panel_count - 1):
            masked = scores[:, position].masked_fill(~available, -torch.inf)
            selected[:, position] = masked.argmax(dim=-1)
            available.scatter_(1, selected[:, position : position + 1], False)
        return selected

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_queries, mean=0.0, std=0.02)


def nonautoregressive_ordering_loss(
    pointer_logits: Tensor,
    target_order: Tensor,
    column_loss_weight: float = 1.0,
) -> Tensor:
    """Train parallel row pointers plus the inverse (column) assignment.

    The row term is the usual position-to-panel cross entropy.  The column
    term transposes the score matrix and uses the inverse target permutation,
    matching the two-direction objective from the reference implementation.
    """
    if pointer_logits.ndim != 3:
        raise ValueError(f"pointer_logits must have shape [B,K,K], got {tuple(pointer_logits.shape)}")
    from Solver.puzzle import validate_order as validate_endpoints
    validate_endpoints(target_order)
    if target_order.shape != pointer_logits.shape[:2]:
        raise ValueError("Target and logits shapes differ")
    if pointer_logits.size(2) != pointer_logits.size(1):
        raise ValueError("pointer_logits must be square in its two pointer dimensions")
    pointer_logits = pointer_logits[:, 1:-1, 1:-1]
    target_order = target_order[:, 1:-1] - 1
    row_loss = F.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.size(-1)), target_order.reshape(-1)
    )
    inverse = torch.empty_like(target_order)
    ranks = torch.arange(target_order.size(1), device=target_order.device).expand_as(target_order)
    inverse.scatter_(1, target_order, ranks)
    column_loss = F.cross_entropy(
        pointer_logits.transpose(1, 2).reshape(-1, pointer_logits.size(1)), inverse.reshape(-1)
    )
    return row_loss + float(column_loss_weight) * column_loss


class MangaZeroQwenVLNonAutoregressiveSolver(nn.Module):
    """Qwen2.5-VL backbone followed by a parallel ordering decoder."""

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
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ),
        decoder_dim: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        num_heads: int = 8,
        decoder_dropout: float = 0.1,
        column_loss_weight: float = 1.0,
        load_in_4bit: bool = False,
        device_map: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if panel_count < 3:
            raise ValueError("panel_count must be at least 3")
        self.panel_count = int(panel_count)
        self.model_name = model_name
        self.decoder_dim = int(decoder_dim)
        self.column_loss_weight = float(column_loss_weight)
        if self.column_loss_weight < 0:
            raise ValueError("column_loss_weight must be non-negative")
        self.use_lora = bool(use_lora)
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("transformers with Qwen2.5-VL support is required") from exc

        processor_kwargs = {"trust_remote_code": trust_remote_code}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        model_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": parse_torch_dtype(torch_dtype)}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        if device_map:
            model_kwargs["device_map"] = device_map
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.backbone.config.use_cache = False
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            if load_in_4bit:
                self.backbone = prepare_model_for_kbit_training(self.backbone)
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, target_modules=list(lora_target_modules), bias="none",
            ))
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        hidden_size = resolve_hidden_size(self.backbone)
        self.panel_projection = nn.Linear(hidden_size, decoder_dim)
        self.panel_norm = nn.LayerNorm(decoder_dim)
        self.decoder = NonAutoregressivePointerDecoder(
            panel_count, decoder_dim, encoder_layers, decoder_layers, num_heads, decoder_dropout
        )
        self._image_pad_token_id: int | None = None
        # With ``device_map=auto`` the VLM is already sharded. Keep the new
        # trainable projection and decoder beside its input embedding device.
        if device_map:
            head_device = self._input_device()
            self.panel_projection.to(head_device)
            self.panel_norm.to(head_device)
            self.decoder.to(head_device)

    def forward(
        self, panel_images: Tensor, target_order: Optional[Tensor] = None,
        dialog_texts: Optional[list[list[str]]] = None, memory: Optional[Tensor] = None,
    ) -> Tensor:
        if target_order is not None:
            validate_order(target_order, self.panel_count)
        if memory is None:
            memory = self.encode_panels(panel_images, dialog_texts=dialog_texts)
        return self.decoder(memory)

    @torch.no_grad()
    def predict_order(
        self, panel_images: Tensor, dialog_texts: Optional[list[list[str]]] = None,
        memory: Optional[Tensor] = None,
    ) -> Tensor:
        if memory is None:
            memory = self.encode_panels(panel_images, dialog_texts=dialog_texts)
        return self.decoder.predict_order(memory)

    @torch.no_grad()
    def greedy_decode(self, memory: Tensor) -> Tensor:
        """Decode an already-encoded puzzle without running the Qwen backbone."""
        return self.decoder.predict_order(memory)

    def encode_panels(self, panel_images: Tensor, dialog_texts: Optional[list[list[str]]] = None) -> Tensor:
        inputs = self.prepare_inputs(panel_images, dialog_texts=dialog_texts)
        outputs = self.backbone(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = outputs.hidden_states[-1]
        memory = self._pool_per_panel(hidden, inputs)
        parameter = next(self.panel_projection.parameters())
        return self.panel_norm(self.panel_projection(memory.to(dtype=parameter.dtype, device=parameter.device)))

    def prepare_inputs(self, panel_images: Tensor, dialog_texts: Optional[list[list[str]]] = None) -> dict[str, Tensor]:
        if panel_images.ndim != 5 or panel_images.size(1) != self.panel_count:
            raise ValueError(f"panel_images must have shape [B,{self.panel_count},3,H,W], got {tuple(panel_images.shape)}")
        if dialog_texts and (
            len(dialog_texts) != panel_images.size(0)
            or any(len(row) != self.panel_count for row in dialog_texts)
        ):
            raise ValueError("dialog_texts must contain one text per input panel in each batch item")
        conversations = []
        for batch_index in range(panel_images.size(0)):
            prompt = (
                f"There are {self.panel_count} comic panels in input-slot order. "
                f"Panel 0 is the known opening and panel {self.panel_count - 1} is the known ending. "
                "The middle panels are shuffled. Panel IDs refer to input slots, not reading positions. "
                "Represent only the middle reading order using visual continuity and panel content."
            )
            if dialog_texts:
                text_lines = [f"panel {i}: {str(text).strip()}" for i, text in enumerate(dialog_texts[batch_index]) if str(text).strip()]
                if text_lines:
                    prompt += "\nOCR text:\n" + "\n".join(text_lines)
            # The readout pools image-token hidden states from a causal LM.
            # Put *all* task/OCR text first so even the opening image can see
            # it; appending text after the images makes it invisible to them.
            content = [{"type": "text", "text": prompt}]
            for index in range(self.panel_count):
                content.append({"type": "text", "text": f"\nPanel {index}:\n"})
                content.append({"type": "image", "image": tensor_to_pil(panel_images[batch_index, index])})
            conversations.append([{"role": "user", "content": content}])
        texts = [self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in conversations]
        try:
            from qwen_vl_utils import process_vision_info
            images, videos = process_vision_info(conversations)
            return self._call_processor(text=texts, images=images, videos=videos)
        except ImportError:
            images = [[item["image"] for item in c[0]["content"] if item["type"] == "image"] for c in conversations]
            return self._call_processor(text=texts, images=images)

    def _call_processor(self, **kwargs):
        try:
            return {key: value.to(self._input_device()) for key, value in self.processor(return_tensors="pt", padding=True, **kwargs).items() if torch.is_tensor(value)}
        except TypeError:
            return {key: value.to(self._input_device()) for key, value in self.processor(**kwargs).items() if torch.is_tensor(value)}

    def _pool_per_panel(self, hidden: Tensor, inputs: dict[str, Tensor]) -> Tensor:
        image_token_id = self._get_image_pad_token_id()
        image_mask = inputs["input_ids"] == image_token_id
        pooled_rows = []
        for batch_index in range(hidden.size(0)):
            positions = image_mask[batch_index].nonzero(as_tuple=False).flatten().tolist()
            runs: list[list[int]] = []
            for position in positions:
                if not runs or position != runs[-1][-1] + 1:
                    runs.append([position])
                else:
                    runs[-1].append(position)
            if len(runs) != self.panel_count:
                raise RuntimeError(f"sample {batch_index} has {len(runs)} image token runs, expected {self.panel_count}")
            pooled_rows.append(torch.stack([hidden[batch_index, run].mean(dim=0) for run in runs]))
        return torch.stack(pooled_rows)

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

    def _input_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    def trainable_state_dict(self) -> dict[str, dict[str, Tensor]]:
        state = {"decoder": {name: value.detach().cpu() for name, value in self.state_dict().items() if not name.startswith("backbone.")}}
        if self.use_lora:
            try:
                from peft import get_peft_model_state_dict
                state["backbone"] = {name: value.detach().cpu() for name, value in get_peft_model_state_dict(self.backbone).items()}
            except ImportError:
                state["backbone"] = {name: value.detach().cpu() for name, value in self.backbone.state_dict().items() if "lora_" in name}
        else:
            state["backbone"] = {}
        return state

    def load_trainable_state_dict(self, state_dict: dict[str, dict[str, Tensor]]) -> None:
        decoder_state = state_dict.get("decoder", state_dict)
        live = {name: value for name, value in self.state_dict().items() if not name.startswith("backbone.")}
        missing = sorted(set(live) - set(decoder_state))
        unexpected = sorted(set(decoder_state) - set(live))
        if missing or unexpected:
            raise RuntimeError(f"decoder checkpoint mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}")
        live.update(decoder_state)
        self.load_state_dict(live, strict=False)
        if self.use_lora and state_dict.get("backbone"):
            try:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(self.backbone, state_dict["backbone"])
            except ImportError:
                self.backbone.load_state_dict(state_dict["backbone"], strict=False)


def pointer_cross_entropy(pointer_logits: Tensor, target_order: Tensor, column_loss_weight: float = 1.0) -> Tensor:
    """Compatibility name for training scripts that use pointer losses."""
    return nonautoregressive_ordering_loss(pointer_logits, target_order, column_loss_weight)

