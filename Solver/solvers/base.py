"""Uniform training/inference contract around the existing model backbones."""

from __future__ import annotations

import torch
from torch import nn


class OrderingSolver(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Generative solvers wrap nn.Module backbones rather than inherit it.
        self.model = model
        if not isinstance(model, nn.Module):
            self.backbone = model.backbone

    def training_loss(self, batch):
        raise NotImplementedError

    def prepare_training_batch(self, batch):
        from Solver.puzzle import shuffle_middle_batch
        return shuffle_middle_batch(batch)

    def predict(self, batch):
        raise NotImplementedError

    def checkpoint_state(self):
        if hasattr(self.model, "trainable_state_dict"):
            return {"format": "trainable", "weights": self.model.trainable_state_dict()}
        return {"format": "full", "weights": self.model.state_dict()}

    def train(self, mode=True):
        super().train(mode)
        backbone = getattr(self.model, "backbone", None)
        if mode and backbone is not None and not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()
        return self

    def load_checkpoint_state(self, state):
        if state["format"] == "trainable":
            self.model.load_trainable_state_dict(state["weights"])
        else:
            self.model.load_state_dict(state["weights"], strict=True)

    def optimizer_groups(self, args):
        backbone, decoder = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (backbone if "backbone." in name else decoder).append(parameter)
        return [{"params": values, "lr": lr} for values, lr in
                ((backbone, args.lr_backbone), (decoder, args.lr_decoder)) if values]

    def place(self, device, *, dispatched=False):
        if dispatched:
            # Avoid .to() on quantized/sharded Qwen weights; move every head.
            if isinstance(self.model, nn.Module):
                for name, module in self.model.named_children():
                    if name != "backbone":
                        module.to(device)
                for parameter in self.model.parameters(recurse=False):
                    parameter.data = parameter.data.to(device)
        else:
            self.to(device)
        return self
