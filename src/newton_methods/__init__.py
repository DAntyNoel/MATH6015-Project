"""Newton-type methods for the MATH6015 optimization project."""

from .algorithms import (
    classical_newton,
    conjugate_gradient,
    fixed_step_gradient_descent,
    modified_damped_newton,
    newton_cg,
)
from .core import IterationRecord, Objective, OptimizationResult
from .line_search import armijo_backtracking

__all__ = [
    "IterationRecord",
    "Objective",
    "OptimizationResult",
    "armijo_backtracking",
    "classical_newton",
    "conjugate_gradient",
    "fixed_step_gradient_descent",
    "modified_damped_newton",
    "newton_cg",
]
