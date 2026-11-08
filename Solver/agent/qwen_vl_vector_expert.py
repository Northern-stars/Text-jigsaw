"""Vector-conditioned LoRA experts for the Qwen-VL panel pointer solver."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from Solver.agent.mangazero_qwen_vl_autoregressive import (
    MangaZeroQwenVLAutoregressiveSolver,
    pointer_cross_entropy,
)


class GeneratedLoRALinear(nn.Module):
    """Frozen linear module with a differentiable LoRA update generated from z."""

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        latent_dim: int,
        *,
        seed: int,
        lora_alpha: float | None = None,
        a_scale: float = 0.01,
        b_scale: float = 0.001,
    ) -> None:
        super().__init__()
        weight = getattr(base, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("generated LoRA targets must expose a 2D weight")
        self.base = base
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.rank = int(rank)
        self.a_scale = float(a_scale)
        self.lora_scale = (float(lora_alpha) if lora_alpha is not None else float(rank)) / rank
        self.lora_scale *= float(b_scale)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.register_buffer(
            "a0",
            torch.empty(rank, self.in_features, dtype=torch.float32),
            persistent=True,
        )
        nn.init.kaiming_uniform_(self.a0, a=5**0.5, generator=generator)
        self.register_buffer(
            "map_a",
            torch.randn(rank * self.in_features, latent_dim, generator=generator)
            / latent_dim**0.5,
            persistent=True,
        )
        self.register_buffer(
            "map_b",
            torch.randn(self.out_features * rank, latent_dim, generator=generator)
            / latent_dim**0.5,
            persistent=True,
        )
        self._active_a: Tensor | None = None
        self._active_b: Tensor | None = None

    def parameters_from_latent(self, z: Tensor) -> tuple[Tensor, Tensor]:
        a_delta = F.linear(z.float(), self.map_a).view(self.rank, self.in_features)
        b = F.linear(z.float(), self.map_b).view(self.out_features, self.rank)
        a = self.a0.float() + self.a_scale * a_delta
        return a, self.lora_scale * b

    def activate(self, z: Tensor) -> None:
        self._active_a, self._active_b = self.parameters_from_latent(z)

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base(inputs)
        if self._active_a is None or self._active_b is None:
            return result
        a = self._active_a.to(device=inputs.device, dtype=inputs.dtype)
        b = self._active_b.to(device=inputs.device, dtype=inputs.dtype)
        delta = F.linear(F.linear(inputs, a), b)
        return result + delta


class FeatureHypernetwork(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.layers(z)


class QwenVLVectorExpertSolver(MangaZeroQwenVLAutoregressiveSolver):
    """Frozen shared Qwen-VL pointer model adapted by expert vectors."""

    def __init__(
        self,
        *,
        base_checkpoint: str | Path,
        panel_count: int,
        latent_dim: int = 64,
        num_experts: int = 4,
        lora_rank: int = 4,
        lora_last_n_layers: int = 2,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        feature_mode: str = "independent",
        feature_hidden_dim: int = 128,
        mapping_seed: int = 0,
        **model_kwargs: Any,
    ) -> None:
        base_path = Path(base_checkpoint)
        checkpoint = _torch_load(base_path, "cpu")
        family = checkpoint.get("solver_family")
        if family not in {None, "qwen_vl_autoregressive"}:
            raise ValueError(f"unsupported shared checkpoint family: {family}")
        config = checkpoint.get("config", {})
        model_kwargs = _model_kwargs_from_checkpoint(config, model_kwargs)
        model_kwargs["use_lora"] = True
        super().__init__(panel_count=panel_count, **model_kwargs)

        state = checkpoint.get("model", checkpoint)
        self.load_trainable_state_dict(state)
        self.base_checkpoint = str(base_path.resolve())
        self.feature_mode = feature_mode
        self.num_experts = int(num_experts)
        self.latent_dim = int(latent_dim)
        self.lora_rank = int(lora_rank)
        self.lora_last_n_layers = int(lora_last_n_layers)
        if feature_mode not in {"independent", "generated"}:
            raise ValueError("feature_mode must be 'independent' or 'generated'")
        if self.num_experts <= 0 or self.latent_dim <= 0 or self.lora_rank <= 0:
            raise ValueError("num_experts, latent_dim, and lora_rank must be positive")

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.expert_vectors = nn.Parameter(torch.randn(num_experts, latent_dim))
        if feature_mode == "independent":
            self.expert_features = nn.Parameter(torch.randn(num_experts, self.decoder_dim) * 0.02)
            self.feature_hypernetwork = None
        else:
            self.register_parameter("expert_features", None)
            self.feature_hypernetwork = FeatureHypernetwork(
                latent_dim,
                self.decoder_dim,
                feature_hidden_dim,
            )

        self._active_feature: Tensor | None = None
        self.generated_module_names: list[str] = []
        self.generated_modules: list[GeneratedLoRALinear] = []
        self._install_generated_lora(
            rank=lora_rank,
            last_n_layers=lora_last_n_layers,
            target_modules=set(lora_target_modules),
            seed=mapping_seed,
        )

    def _install_generated_lora(
        self,
        *,
        rank: int,
        last_n_layers: int,
        target_modules: set[str],
        seed: int,
    ) -> None:
        candidates: list[tuple[str, nn.Module, int, str]] = []
        layer_pattern = re.compile(r"language_model\.layers\.(\d+)\.self_attn\.([^.]+)$")
        layer_ids: set[int] = set()
        for name, module in self.backbone.named_modules():
            match = layer_pattern.search(name)
            if match and match.group(2) in target_modules:
                layer_id = int(match.group(1))
                layer_ids.add(layer_id)
                candidates.append((name, module, layer_id, match.group(2)))
        if not layer_ids:
            raise RuntimeError(
                "no Qwen language self-attention targets found; inspect named_modules() "
                "and pass matching q_proj/v_proj names"
            )
        if last_n_layers <= 0:
            selected_layers = layer_ids
        else:
            selected_layers = set(sorted(layer_ids)[-last_n_layers:])
        candidates = [item for item in candidates if item[2] in selected_layers]
        for index, (name, module, _, _) in enumerate(candidates):
            parent_name, _, child_name = name.rpartition(".")
            parent = self.backbone.get_submodule(parent_name) if parent_name else self.backbone
            wrapped = GeneratedLoRALinear(
                module,
                rank,
                self.latent_dim,
                seed=seed + index,
            )
            setattr(parent, child_name, wrapped)
            self.generated_module_names.append(name)
            self.generated_modules.append(wrapped)
        if not self.generated_modules:
            raise RuntimeError("selected layers contain no target modules")

    def activate_expert(self, expert_id: int) -> None:
        z = self.expert_vectors[expert_id]
        for module in self.generated_modules:
            module.activate(z)
        if self.feature_mode == "independent":
            self._active_feature = self.expert_features[expert_id]
        else:
            assert self.feature_hypernetwork is not None
            self._active_feature = self.feature_hypernetwork(z)

    def clear_expert(self) -> None:
        for module in self.generated_modules:
            module._active_a = None
            module._active_b = None
        self._active_feature = None

    def _condition_inputs(self, inputs: Tensor) -> Tensor:
        if self._active_feature is None:
            return inputs
        return inputs + self._active_feature.to(inputs).view(1, 1, -1)

    def _teacher_forcing_inputs(self, memory: Tensor, target_order: Tensor) -> Tensor:
        return self._condition_inputs(super()._teacher_forcing_inputs(memory, target_order))

    def _decode_inputs_from_selected(self, memory: Tensor, selected: Tensor) -> Tensor:
        return self._condition_inputs(super()._decode_inputs_from_selected(memory, selected))

    def forward_expert(
        self,
        expert_id: int,
        panel_images: Tensor,
        target_order: Tensor,
        dialog_texts: list[list[str]] | None = None,
    ) -> Tensor:
        self.activate_expert(expert_id)
        try:
            return super().forward(panel_images, target_order=target_order, dialog_texts=dialog_texts)
        finally:
            self.clear_expert()

    def ensemble_predict(
        self,
        panel_images: Tensor,
        dialog_texts: list[list[str]] | None = None,
    ) -> Tensor:
        memories = []
        for expert_id in range(self.num_experts):
            self.activate_expert(expert_id)
            memories.append(self.encode_panels(panel_images, dialog_texts=dialog_texts))
            self.clear_expert()
        selected = torch.empty(
            panel_images.size(0), 0, dtype=torch.long, device=panel_images.device
        )
        for _ in range(self.panel_count):
            expert_probabilities = []
            for expert_id, memory in enumerate(memories):
                self.activate_expert(expert_id)
                decoder_inputs = self._decode_inputs_from_selected(memory, selected)
                causal_mask = torch.triu(
                    torch.ones(
                        decoder_inputs.size(1),
                        decoder_inputs.size(1),
                        device=memory.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                states = self.pointer_decoder(
                    decoder_inputs,
                    memory,
                    tgt_mask=causal_mask,
                )
                logits = self._pointer_logits(states[:, -1:], memory).squeeze(1)
                self.clear_expert()
                if selected.numel():
                    logits = logits.scatter(1, selected, -torch.inf)
                expert_probabilities.append(logits.softmax(dim=-1))
            next_index = torch.stack(expert_probabilities).mean(0).argmax(-1, keepdim=True)
            selected = torch.cat([selected, next_index], dim=1)
        return selected

    def expert_trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [self.expert_vectors]
        if self.expert_features is not None:
            parameters.append(self.expert_features)
        if self.feature_hypernetwork is not None:
            parameters.extend(self.feature_hypernetwork.parameters())
        return parameters

    def expert_checkpoint_state(self) -> dict[str, Any]:
        return {
            "expert_vectors": self.expert_vectors.detach().cpu(),
            "expert_features": (
                None if self.expert_features is None else self.expert_features.detach().cpu()
            ),
            "feature_hypernetwork": (
                None
                if self.feature_hypernetwork is None
                else {k: v.detach().cpu() for k, v in self.feature_hypernetwork.state_dict().items()}
            ),
            "generated_lora_buffers": [
                {k: v.detach().cpu() for k, v in module.named_buffers(recurse=False)}
                for module in self.generated_modules
            ],
            "generated_module_names": list(self.generated_module_names),
        }

    def load_expert_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("generated_module_names") != self.generated_module_names:
            raise RuntimeError("generated LoRA target layout does not match checkpoint")
        if len(state["generated_lora_buffers"]) != len(self.generated_modules):
            raise RuntimeError("generated LoRA buffer count does not match checkpoint")
        self.expert_vectors.data.copy_(state["expert_vectors"].to(self.expert_vectors))
        if self.expert_features is not None:
            self.expert_features.data.copy_(state["expert_features"].to(self.expert_features))
        if self.feature_hypernetwork is not None:
            self.feature_hypernetwork.load_state_dict(state["feature_hypernetwork"])
        for module, buffers in zip(self.generated_modules, state["generated_lora_buffers"]):
            module.load_state_dict(buffers, strict=False)

    def initialize_generated_features(
        self,
        state: dict[str, Any],
        *,
        fit_steps: int = 500,
        fit_lr: float = 1e-3,
    ) -> None:
        if self.feature_mode != "generated" or self.feature_hypernetwork is None:
            raise ValueError("feature initialization requires feature_mode='generated'")
        if state.get("generated_module_names") != self.generated_module_names:
            raise RuntimeError("stage-one and stage-two LoRA target layouts differ")
        if len(state["generated_lora_buffers"]) != len(self.generated_modules):
            raise RuntimeError("stage-one generated LoRA buffer count differs")
        source_vectors = state["expert_vectors"].to(self.expert_vectors)
        source_features = state.get("expert_features")
        if source_features is None:
            raise ValueError("stage-one checkpoint must contain independent expert features")
        source_features = source_features.to(self.expert_vectors.device, dtype=torch.float32)
        if source_vectors.shape != self.expert_vectors.shape:
            raise ValueError("stage-one expert count or latent dimension does not match")
        if source_features.shape != (self.num_experts, self.decoder_dim):
            raise ValueError("stage-one feature dimension does not match this pointer decoder")
        if fit_steps <= 0:
            raise ValueError("fit_steps must be positive")

        self.expert_vectors.data.copy_(source_vectors)
        for module, buffers in zip(self.generated_modules, state["generated_lora_buffers"]):
            module.load_state_dict(buffers, strict=False)
        mapper_optimizer = torch.optim.Adam(self.feature_hypernetwork.parameters(), lr=fit_lr)
        for _ in range(fit_steps):
            prediction = self.feature_hypernetwork(self.expert_vectors.detach())
            loss = F.mse_loss(prediction, source_features)
            mapper_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            mapper_optimizer.step()


def expert_loss(model: QwenVLVectorExpertSolver, batch: dict[str, Any]) -> Tensor:
    losses = []
    for expert_id in range(model.num_experts):
        logits = model.forward_expert(
            expert_id,
            batch["panel_images"],
            batch["target_order"],
            dialog_texts=batch.get("dialog_texts"),
        )
        losses.append(pointer_cross_entropy(logits, batch["target_order"]))
    return torch.stack(losses).mean()


def iter_expert_losses(model: QwenVLVectorExpertSolver, batch: dict[str, Any]):
    for expert_id in range(model.num_experts):
        logits = model.forward_expert(
            expert_id,
            batch["panel_images"],
            batch["target_order"],
            dialog_texts=batch.get("dialog_texts"),
        )
        yield pointer_cross_entropy(logits, batch["target_order"])


def _model_kwargs_from_checkpoint(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_name",
        "torch_dtype",
        "attn_implementation",
        "min_pixels",
        "max_pixels",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "decoder_layers",
        "num_heads",
        "decoder_dim",
        "decoder_dropout",
        "load_in_4bit",
        "device_map",
    )
    values = {key: config[key] for key in keys if key in config}
    values["lora_target_modules"] = tuple(
        item.strip()
        for item in str(values.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")).split(",")
        if item.strip()
    )
    values.update(overrides)
    return values


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

