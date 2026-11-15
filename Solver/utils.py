"""Compatibility helpers for research experiment runners.

Puzzle semantics and metrics are implemented in Solver.puzzle.
"""
from Solver.puzzle import ordering_metrics, shuffle_middle_batch


def randomize_training_batch(batch, use_layout=False):
    """Shuffle only middle panels, keeping all modalities aligned."""
    return shuffle_middle_batch(batch)
