"""Objective functions used by the assignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import Array


@dataclass(frozen=True)
class FunctionObjective:
    """Wrap value, gradient, and Hessian callables as an objective."""

    value_fn: Callable[[Array], float]
    gradient_fn: Callable[[Array], Array]
    hessian_fn: Callable[[Array], Array]
    name: str = "objective"

    def value(self, x: Array) -> float:
        return float(self.value_fn(np.asarray(x, dtype=float)))

    def gradient(self, x: Array) -> Array:
        return np.asarray(self.gradient_fn(np.asarray(x, dtype=float)), dtype=float)

    def hessian(self, x: Array) -> Array:
        return np.asarray(self.hessian_fn(np.asarray(x, dtype=float)), dtype=float)


def experiment1_objective() -> FunctionObjective:
    """Return f1 from the assignment."""

    def value(x: Array) -> float:
        x0, y0 = x
        return x0**4 + y0**4 + x0**2 + y0**2 + 0.5 * x0 * y0

    def gradient(x: Array) -> Array:
        x0, y0 = x
        return np.array([4.0 * x0**3 + 2.0 * x0 + 0.5 * y0, 4.0 * y0**3 + 2.0 * y0 + 0.5 * x0])

    def hessian(x: Array) -> Array:
        x0, y0 = x
        return np.array([[12.0 * x0**2 + 2.0, 0.5], [0.5, 12.0 * y0**2 + 2.0]])

    return FunctionObjective(value, gradient, hessian, name="f1")


def experiment2_objective() -> FunctionObjective:
    """Return f2 from the assignment."""

    def value(x: Array) -> float:
        x0, y0 = x
        return 0.5 * x0**2 + 0.25 * y0**4 - 0.5 * y0**2

    def gradient(x: Array) -> Array:
        x0, y0 = x
        return np.array([x0, y0**3 - y0])

    def hessian(x: Array) -> Array:
        _, y0 = x
        return np.array([[1.0, 0.0], [0.0, 3.0 * y0**2 - 1.0]])

    return FunctionObjective(value, gradient, hessian, name="f2")
