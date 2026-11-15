"""Qwen-VL text generation training and ordering inference."""

import torch

from Solver.solvers.base import OrderingSolver


class GenerativeSolver(OrderingSolver):
    def train(self, mode=True):
        super().train(mode)
        self.model.backbone.train(mode)
        return self

    def training_loss(self, batch):
        return self.model.compute_loss(batch["panel_images"], batch.get("dialog_texts"), batch["target_order"])

    def predict(self, batch):
        orders = []
        for index, images in enumerate(batch["panel_images"]):
            texts = batch["dialog_texts"][index] if batch.get("dialog_texts") is not None else None
            orders.append(self.model.solve(list(images), texts).order)
        return torch.tensor(orders, device=batch["panel_images"].device, dtype=torch.long)
