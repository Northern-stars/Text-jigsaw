"""Vector experts share the unified puzzle and epoch runner."""

from Solver.solvers.base import OrderingSolver


class VectorExpertSolver(OrderingSolver):
    def training_loss(self, batch):
        from Solver.agent.qwen_vl_vector_expert import expert_loss
        return expert_loss(self.model, batch)

    def predict(self, batch):
        return self.model.ensemble_predict(batch["panel_images"], batch.get("dialog_texts"))

    def checkpoint_state(self):
        return {"format": "experts", "weights": self.model.expert_checkpoint_state()}

    def load_checkpoint_state(self, state):
        self.model.load_expert_checkpoint_state(state["weights"])
