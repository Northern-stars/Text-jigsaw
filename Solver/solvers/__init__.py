"""Solver implementations own training objectives and prediction policies."""

from Solver.solvers.registry import SOLVERS, create_solver

__all__ = ["SOLVERS", "create_solver"]
