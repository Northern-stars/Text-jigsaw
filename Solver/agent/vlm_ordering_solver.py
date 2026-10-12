"""Generic VLM interface for panel-ordering puzzles.

One ``solve`` call wires three collaborating pieces together:

    panels (+ optional OCR texts)
        -> prompt_generator  -> multimodal prompt (vision + task text)
        -> vlm_infer         -> raw decoded model text
        -> answer_generator  -> final ordering

The default ``prompt_generator`` follows the task guidance used by the existing
Qwen2.5-VL solvers (``Solver/agent/mangazero_qwen_vl_*.py``): it numbers the
input panels, states the reading-order goal, and appends per-panel OCR text when
available. The default ``answer_generator`` parses the model's textual ordering
into a permutation of input panel indices. Each stage, together with the model
call itself, can be replaced so the same wrapper targets any other VLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image


PanelImage = Any
TextList = Optional[Sequence[Optional[str]]]

PromptGenerator = Callable[[Sequence[PanelImage], TextList], Any]
VLMInfer = Callable[[Any], Any]
AnswerGenerator = Callable[[Any, int], list[int]]


@dataclass
class VLMOrderingResult:
    """Final output of :meth:`VLMOrderingSolver.solve`."""

    order: list[int]
    raw_output: str
    prompt: Any = None
    panel_count: int = 0
    texts: list[Optional[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Panel normalization
# ---------------------------------------------------------------------------

def to_pil(panel: PanelImage) -> Image.Image:
    """Convert a panel item into a PIL RGB image.

    Accepts ``PIL.Image``, ``torch``/``numpy`` tensors in ``CHW`` or ``HWC``
    layout, a path, or a dataset panel dict exposing ``padded_path``.
    """
    if isinstance(panel, Image.Image):
        return panel.convert("RGB")

    if isinstance(panel, dict):
        for key in ("padded_path", "image_path", "path"):
            if panel.get(key):
                return to_pil(panel[key])
        for key in ("image", "padded_image", "panel_image"):
            if panel.get(key) is not None:
                return to_pil(panel[key])
        raise ValueError("panel dict has no usable image field (padded_path/image)")

    if isinstance(panel, (str, Path)):
        return Image.open(panel).convert("RGB")

    array = _to_numpy(panel)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"cannot interpret panel of shape {array.shape}")
    if array.dtype != np.uint8:
        array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


def _to_numpy(panel: Any) -> np.ndarray:
    if hasattr(panel, "detach"):
        panel = panel.detach().cpu().numpy()
    return np.asarray(panel)


def parse_torch_dtype(raw_dtype: str) -> Any:
    """Map a dtype string such as ``"bfloat16"`` onto a ``torch`` dtype."""
    import torch

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


# ---------------------------------------------------------------------------
# Default prompt generator
# ---------------------------------------------------------------------------

def build_ordering_text_prompt(
    panel_count: int,
    texts: TextList = None,
    *,
    start_index: int = 0,
    ocr_header: str = "OCR text",
) -> str:
    """Task-guidance text prompt for a panel-ordering puzzle.

    Mirrors the wording used by the existing Qwen-VL solvers. ``texts`` is the
    per-panel OCR/dialog list; pass ``None`` to omit the OCR block.
    """
    panel_ids = ", ".join(str(start_index + index) for index in range(panel_count))
    prompt = (
        "You are solving a manga panel ordering puzzle. "
        f"There are {panel_count} input panels numbered {panel_ids}. "
        "Use visual continuity, reading flow, characters, dialog bubbles, and scene transitions "
        "to infer the correct chronological reading order. "
        f"Output the order as a comma-separated list of the {panel_count} panel numbers, "
        "from first to last, and nothing else."
    )
    if texts:
        lines = []
        for index in range(panel_count):
            text = texts[index] if index < len(texts) else None
            text = str(text).strip() if text is not None else ""
            if text:
                lines.append(f"panel {start_index + index}: {text}")
        if lines:
            prompt += f"\n{ocr_header}:\n" + "\n".join(lines)
    return prompt


def build_ordering_messages(
    panels: Sequence[PanelImage],
    texts: TextList = None,
    *,
    start_index: int = 0,
    system_prompt: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Default prompt generator: interleave panel images with the task prompt.

    Returns chat-style messages ready for ``processor.apply_chat_template``.
    """
    content: list[dict[str, Any]] = []
    for panel in panels:
        content.append({"type": "image", "image": to_pil(panel)})
    content.append(
        {
            "type": "text",
            "text": build_ordering_text_prompt(len(panels), texts, start_index=start_index),
        }
    )
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


# ---------------------------------------------------------------------------
# Default answer generator
# ---------------------------------------------------------------------------

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]")
_NUMBER_RE = re.compile(r"-?\d+")


def parse_ordering_answer(
    raw_text: Any,
    panel_count: int,
    *,
    one_indexed: bool = False,
    strict: bool = True,
) -> list[int]:
    """Parse model text into a permutation of ``0..panel_count-1``.

    Accepts JSON arrays (``[2, 0, 1]``) or free-form number lists
    (``2 -> 0 -> 1``, ``2, 0, 1``, newline separated). Numbers are deduplicated
    in first-seen order and out-of-range values dropped. When ``strict`` and the
    result is not a full permutation, a ``ValueError`` is raised; otherwise the
    missing indices are appended in ascending order.
    """
    text = "" if raw_text is None else str(raw_text).strip()

    numbers: list[int] = []
    array_match = _ARRAY_RE.search(text)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                numbers = [int(value) for value in parsed]
        except (ValueError, TypeError):
            numbers = []
    if not numbers:
        numbers = [int(value) for value in _NUMBER_RE.findall(text)]

    if one_indexed:
        numbers = [value - 1 for value in numbers]

    order: list[int] = []
    seen: set[int] = set()
    for value in numbers:
        if 0 <= value < panel_count and value not in seen:
            order.append(value)
            seen.add(value)
        if len(order) == panel_count:
            break

    if len(order) < panel_count:
        if strict:
            raise ValueError(
                f"could not parse a full permutation of {panel_count} panels "
                f"from model output: {text!r}"
            )
        order.extend(index for index in range(panel_count) if index not in seen)
    return order[:panel_count]


# ---------------------------------------------------------------------------
# Default VLM backend (Qwen2.5-VL)
# ---------------------------------------------------------------------------

class QwenVLInfer:
    """Default ``vlm_infer`` callable backed by a Qwen2.5-VL model+processor.

    ``__call__`` accepts the messages produced by ``build_ordering_messages``
    and returns the decoded generation as plain text.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.setdefault("do_sample", do_sample)
        self.generation_kwargs.setdefault("max_new_tokens", max_new_tokens)

    def __call__(self, prompt: Any) -> str:
        import torch

        conversations = self._as_conversations(prompt)
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        inputs = self._prepare_inputs(texts, conversations)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(**inputs, **self.generation_kwargs)

        generated = generated[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    @staticmethod
    def _as_conversations(prompt: Any) -> list[Any]:
        """Normalize a prompt into a list of chat conversations.

        ``build_ordering_messages`` returns one conversation (a list of message
        dicts), but a batched caller may pass several conversations, so both
        ``[{role, content}, ...]`` and ``[[{...}], [{...}]]`` are accepted.
        """
        if (
            isinstance(prompt, list)
            and prompt
            and isinstance(prompt[0], dict)
            and isinstance(prompt[0].get("content"), (list, str))
        ):
            return [prompt]
        return list(prompt)

    def _prepare_inputs(self, texts: list[str], messages: list[Any]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            return self.processor(text=texts, return_tensors="pt", padding=True)
        images, videos = process_vision_info(messages)
        return self.processor(text=texts, images=images, videos=videos, return_tensors="pt", padding=True)


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class VLMOrderingSolver:
    """Universal interface that turns a panel list (+ OCR texts) into an ordering.

    Parameters
    ----------
    vlm_infer:
        Callable taking the prompt object produced by ``prompt_generator`` and
        returning raw model text (or anything the ``answer_generator`` accepts).
    prompt_generator:
        Callable ``(panels, texts) -> prompt``. Defaults to
        :func:`build_ordering_messages`.
    answer_generator:
        Callable ``(raw_output, panel_count) -> list[int]``. Defaults to
        :func:`parse_ordering_answer`.
    panel_count:
        Optional expected panel count; when set, inputs of a different size raise.
    one_indexed:
        Passed to the default answer generator to convert 1-based numbering.
    strict_answer:
        Passed to the default answer generator; raise on unparsable output.
    """

    def __init__(
        self,
        vlm_infer: VLMInfer,
        prompt_generator: Optional[PromptGenerator] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        *,
        panel_count: Optional[int] = None,
        one_indexed: bool = False,
        strict_answer: bool = True,
    ) -> None:
        self.vlm_infer = vlm_infer
        self.prompt_generator = prompt_generator or build_ordering_messages
        self.panel_count = panel_count
        if answer_generator is not None:
            self.answer_generator = answer_generator
        else:
            self.answer_generator = lambda raw, count: parse_ordering_answer(
                raw, count, one_indexed=one_indexed, strict=strict_answer
            )

    def solve(
        self,
        panels: Sequence[PanelImage],
        texts: TextList = None,
    ) -> VLMOrderingResult:
        """Run the full prompt -> infer -> answer pipeline for one puzzle."""
        panels = list(panels)
        if not panels:
            raise ValueError("panels must not be empty")
        panel_count = len(panels)
        if self.panel_count is not None and panel_count != self.panel_count:
            raise ValueError(f"expected panel_count={self.panel_count}, got {panel_count}")

        prompt = self.prompt_generator(panels, texts)
        raw_output = self.vlm_infer(prompt)
        order = self.answer_generator(raw_output, panel_count)
        self._validate(order, panel_count)
        return VLMOrderingResult(
            order=list(order),
            raw_output="" if raw_output is None else str(raw_output),
            prompt=prompt,
            panel_count=panel_count,
            texts=list(texts) if texts is not None else [],
        )

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
        """Abstract training hook.

        Concrete solvers adapt the VLM to the ordering task here. ``batches`` is
        an iterable of training batches (typically the dicts produced by
        ``collate_panel_ordering_batch``); each batch carries panel images, the
        optional OCR texts, and the ground-truth ``target_order``. Implementations
        own the loss, optimizer wiring, and device placement, then return a
        metrics dict (at least ``{"loss": ...}``).

        The base wrapper exposes this as an abstract method: the generic
        prompt/infer/answer path is inference-only, so subclasses must override
        it to become trainable.
        """
        raise NotImplementedError(
            "VLMOrderingSolver.train() is abstract; instantiate a trainable "
            "subclass (e.g. QwenVLOrderingSolver) that implements it."
        )

    @staticmethod
    def _validate(order: Sequence[int], panel_count: int) -> None:
        if len(order) != panel_count or sorted(order) != list(range(panel_count)):
            raise ValueError(f"answer is not a permutation of {panel_count} panels: {list(order)}")
