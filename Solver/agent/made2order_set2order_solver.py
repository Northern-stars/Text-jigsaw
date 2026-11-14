"""Independent panel-ordering solver backed by third_party/made2order.

This is a thin project-side adapter around the upstream Made2Order model
(ECCV 2024, https://github.com/charigyang/made2order). The upstream model
is kept in third_party/made2order and is imported at runtime so the
third-party code stays untouched.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

SEQUENCE_LEN_LIMIT = 16

# Updated before each model build so the upstream positional embedding is
# created on the same device as this wrapper.
_CURRENT_MODEL_DEVICE = torch.device("cpu")


def _install_posemb_device_patch(model_mod: Any) -> None:
    """Route upstream posemb_sincos_3d to the wrapper's target device.

    The vendor function defaults to `device='cuda'`; on CPU-only machines the
    construction would still try to allocate CUDA tensors even though the
    learned query .cuda() call is neutralized. The patch keeps the upstream
    function untouched in the repository but runs it on the active device.
    """
    original = getattr(model_mod, "_made2order_original_posemb_sincos_3d", None)
    if original is None:
        original = model_mod.posemb_sincos_3d
        model_mod._made2order_original_posemb_sincos_3d = original

    def posemb_sincos_3d(
        f: int,
        h: int,
        w: int,
        dim: int,
        temperature: float = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return original(
            f,
            h,
            w,
            dim,
            device=_CURRENT_MODEL_DEVICE.type,
            temperature=temperature,
            dtype=dtype,
        )

    model_mod.posemb_sincos_3d = posemb_sincos_3d


def _install_utils_stub(vendor_dir: Path) -> None:
    """Make model.py importable without the vendor's optional utils deps.

    The upstream model.py does `from utils import pair`, which triggers the
    vendor utils.py top-level imports (cv2, natsort, ...). The project-side
    wrapper only needs `pair`, so preload a tiny utils stub before importing
    the vendor model.
    """
    import types

    if "utils" in sys.modules:
        return
    stub = types.ModuleType("utils")
    stub.pair = lambda t: t if isinstance(t, tuple) else (t, t)
    sys.modules["utils"] = stub


def _load_upstream_module(module_name: str) -> Any:
    vendor_dir = Path(__file__).resolve().parents[2] / "third_party" / "made2order"
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    _install_utils_stub(vendor_dir)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module


def _upstream_model_module() -> Any:
    return _load_upstream_module("model")


class Made2OrderPanelOrderingSolver(nn.Module):
    """Wrap upstream Made2Order for fixed-size comic panel ordering.

    Args:
        panel_count: Number of panels in each puzzle (must be <= 16).
        image_size: (height, width) of the resized input images.
        patch_size: Side length of each square image patch used by the
            upstream transformer; image_size must be divisible by it.
        device: Target device. The upstream model hard-codes its learned
            query parameter on CUDA, so the wrapper moves that parameter to
            the requested device after construction.
    """

    def __init__(
        self,
        panel_count: int,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        dim: int = 256,
        heads: int = 4,
        decoder_dim: int = 64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if panel_count < 2:
            raise ValueError("panel_count must be at least 2")
        if panel_count > SEQUENCE_LEN_LIMIT:
            raise ValueError(f"made2order supports at most {SEQUENCE_LEN_LIMIT} panels")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.panel_count = panel_count
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        h, w = self.image_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {patch_size}")

        device_for_params = (
            torch.device(device) if device is not None else self._infer_device()
        )

        # The upstream model hard-codes .cuda() on its learned query parameter
        # and its positional embedding, which breaks CPU-only construction.
        # Temporarily neutralize .cuda() and pin posemb to the target device so
        # the vendor module can build, then move every parameter (including the
        # query) through the wrapper's .to().
        original_cuda = torch.Tensor.cuda
        if device_for_params.type != "cuda":
            torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
        global _CURRENT_MODEL_DEVICE
        _CURRENT_MODEL_DEVICE = device_for_params
        model_mod = _upstream_model_module()
        _install_posemb_device_patch(model_mod)
        try:
            baseline = getattr(model_mod, "Made2Order")(
                image_size=self.image_size,
                image_patch_size=patch_size,
                frames=panel_count,
                channels=3,
                dim=dim,
                heads=heads,
                decoder_dim=decoder_dim,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate upstream Made2Order from "
                f"{Path(__file__).resolve().parents[2] / 'third_party' / 'made2order' / 'model.py'} "
                f"(make sure the vendor tree is present): {exc}"
            ) from exc
        finally:
            torch.Tensor.cuda = original_cuda

        # Keep the upstream module as a child so its query parameter stays a
        # proper registered parameter; the wrapper's .to() then moves it. The
        # vendor stores pos_embedding as a plain tensor attribute, which .to()
        # ignores; replace it with a non-persistent buffer so device moves work.
        self.upstream = baseline
        pos_embedding = baseline.pos_embedding.detach().to(device_for_params)
        del baseline.pos_embedding
        baseline.register_buffer(
            "pos_embedding",
            pos_embedding,
            persistent=False,
        )
        self.embed_dim = dim
        self.to(device_for_params)

    @staticmethod
    def _infer_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _target_device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Rank the input panel sequence.

        Args:
            video: (batch, panel_count, channels, height, width) RGB float
                images normalized to [0, 1].

        Returns:
            (scores, attention): scores is (batch, panel_count, panel_count)
            and attention is (batch, panel_count, num_patches, panel_count).
        """
        if video.ndim != 5:
            raise ValueError(f"Expected (batch, frames, channels, H, W) input, got {tuple(video.shape)}")
        b, f, c, h, w = video.shape
        if f != self.panel_count:
            raise ValueError(
                f"Expected {self.panel_count} panels per sample, got {f}. "
                "Reconstruct the model with the dataset panel_count."
            )
        target_device = self._target_device()
        video = video.to(target_device)

        # The upstream forward rearranges to (b, c, f, h, w) internally.
        scores, attn = self.upstream(video)
        return scores, attn

    @torch.no_grad()
    def predict_order(self, video: Tensor) -> Tensor:
        """Return the predicted reading-order sequence of input slots.

        The upstream output predicts a monotonic rank for each input slot;
        sorting those ranks yields the sequence of input slots in predicted
        reading order, matching the dataset's target_order label format.
        """
        scores, _ = self.forward(video)
        pred_ranks = scores[:, 1:, 1:].argmax(dim=-1)
        tail = torch.argsort(pred_ranks, dim=-1) + 1
        return torch.cat([torch.zeros_like(tail[:, :1]), tail], dim=1)

    @staticmethod
    def ranking_loss(scores: Tensor, target_order: Tensor) -> Tensor:
        """Softmax cross-entropy between predicted ranks and target order.

        The dataset stores target_order as the reading-order sequence of input
        slot indices. The upstream model instead predicts a rank for each
        input slot, so invert the label before computing cross-entropy.
        """
        if scores.ndim != 3:
            raise ValueError(f"Expected (batch, panels, panels) scores, got {tuple(scores.shape)}")
        if target_order.ndim != 2:
            raise ValueError(f"Expected (batch, panels) target_order, got {tuple(target_order.shape)}")
        from Solver.data.given_start import validate_given_start
        validate_given_start(target_order)
        ranks = torch.argsort(target_order.long(), dim=-1)[:, 1:] - 1
        return torch.nn.functional.cross_entropy(
            scores[:, 1:, 1:].reshape(-1, scores.size(-1) - 1),
            ranks.reshape(-1),
        )

    @staticmethod
    def order_metrics(pred_order: Tensor, target_order: Tensor) -> dict[str, float]:
        from Solver.data.given_start import remaining_order_metrics
        return remaining_order_metrics(pred_order.long(), target_order.long())

    def train_metrics(self, pred_order: Tensor, target_order: Tensor) -> dict[str, float]:
        return self.order_metrics(pred_order, target_order)


def main() -> None:
    torch.manual_seed(0)
    solver = Made2OrderPanelOrderingSolver(
        panel_count=3,
        image_size=(56, 56),
        patch_size=7,
        device="cpu",
    )
    fake = torch.rand(2, 3, 3, 56, 56)
    scores, attn = solver(fake)
    order = solver.predict_order(fake)
    print("scores", tuple(scores.shape), "attn", tuple(attn.shape), "order", tuple(order.shape))
    print("loss", float(solver.ranking_loss(scores, order).detach()))


if __name__ == "__main__":
    main()
