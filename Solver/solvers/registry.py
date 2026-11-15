"""Lazy model registry: the CLI chooses a solver, never a training script."""

import importlib
import inspect

from Solver.solvers.classifier import ClassifierSolver
from Solver.solvers.generative import GenerativeSolver
from Solver.solvers.made2order import Made2OrderSolver
from Solver.solvers.pointer import PointerSolver
from Solver.solvers.vector_expert import VectorExpertSolver

SOLVERS = {
    "set2seq": ("mangazero_set2seq_solver", "MangaZeroSetToSequenceSolver", PointerSolver, {}),
    "permutation": ("mangazero_permutation_classifier", "MangaZeroPermutationClassifier", ClassifierSolver, {}),
    "jigsaw": ("mangazero_jigsaw_solver", "MangaZeroJigsawSolver", ClassifierSolver, {"jigsaw": True}),
    "made2order": ("made2order_set2order_solver", "Made2OrderPanelOrderingSolver", Made2OrderSolver, {}),
    "qwen_vl_autoregressive": ("mangazero_qwen_vl_autoregressive", "MangaZeroQwenVLAutoregressiveSolver", PointerSolver, {"qwen": True}),
    "qwen_vl_nonautoregressive": ("mangazero_qwen_vl_nonautoregressive", "MangaZeroQwenVLNonAutoregressiveSolver", PointerSolver, {"qwen": True, "parallel": True}),
    "qwen_vl_classifier": ("mangazero_qwen_vl_lora_classifier", "MangaZeroQwenVLLoRAClassifier", ClassifierSolver, {"qwen": True}),
    "qwen_vl": ("qwen_vl_ordering_solver", "QwenVLOrderingSolver", GenerativeSolver, {}),
    "qwen_vl_vector_expert": ("qwen_vl_vector_expert", "QwenVLVectorExpertSolver", VectorExpertSolver, {}),
}


def create_solver(name, options):
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver {name!r}; choose from {', '.join(SOLVERS)}")
    module, class_name, wrapper, settings = SOLVERS[name]
    model_class = getattr(importlib.import_module(f"Solver.agent.{module}"), class_name)
    signature = inspect.signature(model_class)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        unknown = set(options) - set(signature.parameters)
        if unknown:
            raise ValueError(f"Unsupported {name} model options: {sorted(unknown)}")
    options = dict(options)
    for key in ("lora_target_modules", "image_size"):
        if key in options:
            options[key] = tuple(options[key].split(",")) if isinstance(options[key], str) else tuple(options[key])
    if settings.get("parallel"):
        settings = {**settings, "column_loss_weight": options.get("column_loss_weight", 1.0)}
    return wrapper(model_class(**options), **settings)
