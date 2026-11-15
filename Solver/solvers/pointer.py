"""Autoregressive and parallel pointer training/inference implementations."""

from Solver.solvers.base import OrderingSolver


class PointerSolver(OrderingSolver):
    def __init__(self, model, *, qwen=False, parallel=False, column_loss_weight=1.0):
        super().__init__(model)
        self.qwen = qwen
        self.parallel = parallel
        self.column_loss_weight = column_loss_weight

    def _inputs(self, batch):
        inputs = {"panel_images": batch["panel_images"], "dialog_texts": batch.get("dialog_texts")}
        if not self.qwen:
            inputs["layout_features"] = batch.get("layout_features")
        return inputs

    def training_loss(self, batch):
        logits = self.model(**self._inputs(batch), target_order=batch["target_order"])
        if self.parallel:
            from Solver.agent.mangazero_qwen_vl_nonautoregressive import nonautoregressive_ordering_loss
            return nonautoregressive_ordering_loss(logits, batch["target_order"], self.column_loss_weight)
        from Solver.agent.mangazero_set2seq_solver import pointer_cross_entropy
        return pointer_cross_entropy(logits, batch["target_order"])

    def predict(self, batch):
        function = self.model.predict_order if self.qwen else self.model.greedy_decode
        return function(**self._inputs(batch))
