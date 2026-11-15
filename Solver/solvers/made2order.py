"""Made2Order rank prediction and middle-only supervision."""

from Solver.solvers.base import OrderingSolver


class Made2OrderSolver(OrderingSolver):
    def training_loss(self, batch):
        scores, _ = self.model(batch["panel_images"])
        return self.model.ranking_loss(scores, batch["target_order"])

    def predict(self, batch):
        return self.model.predict_order(batch["panel_images"])
