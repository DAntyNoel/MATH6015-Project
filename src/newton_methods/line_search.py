"""Line-search routines."""

from __future__ import annotations

import numpy as np

from .core import Array, Objective


def armijo_backtracking(
    objective: Objective,
    x: Array,
    direction: Array,
    *,
    initial_step: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    min_step: float = 1e-16,
    max_backtracks: int = 60,
) -> float:
    """Return a step length satisfying the Armijo sufficient decrease condition."""

    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    if not 0.0 < c1 < 1.0:
        raise ValueError("c1 must be in (0, 1)")

    f_x = objective.value(x)
    grad_x = objective.gradient(x)
    directional_derivative = float(np.dot(grad_x, direction))

    if directional_derivative >= 0.0:
        raise ValueError("direction must be a descent direction for Armijo search")

    step = initial_step
    for _ in range(max_backtracks):
        candidate = x + step * direction
        if objective.value(candidate) <= f_x + c1 * step * directional_derivative:
            return step
        step *= rho
        if step < min_step:
            break

    return step
