"""Permutation classifiers, including bounded Jigsaw permutation banks."""

import torch
import torch.nn.functional as F

from Solver.puzzle import reorder_batch
from Solver.solvers.base import OrderingSolver


class ClassifierSolver(OrderingSolver):
    def __init__(self, model, *, qwen=False, jigsaw=False):
        super().__init__(model)
        self.qwen = qwen
        self.jigsaw = jigsaw

    def _inputs(self, batch):
        inputs = {"panel_images": batch["panel_images"]}
        if not self.qwen:
            inputs["layout_features"] = batch.get("layout_features")
        if not self.jigsaw:
            inputs["dialog_texts"] = batch.get("dialog_texts")
        return inputs

    def prepare_training_batch(self, batch):
        if not self.jigsaw:
            return super().prepare_training_batch(batch)
        return prepare_self_supervised_batch(batch, self.model)

    def training_loss(self, batch):
        return F.cross_entropy(self.model(**self._inputs(batch)), self.model.target_to_class_indices(batch["target_order"]))

    def predict(self, batch):
        return self.model.predict_order(**self._inputs(batch))


def prepare_self_supervised_batch(batch, model):
    target = batch["target_order"]
    ids = torch.randint(model.num_classes, (target.size(0),))
    permutation = model.permutation_bank[ids.to(model.permutation_bank.device)].to(target.device)
    return reorder_batch(batch, target.gather(1, permutation))
