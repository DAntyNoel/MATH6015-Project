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


@dataclass(frozen=True)
class DiagonalQuarticObjective:
    """High-dimensional diagonal quadratic plus componentwise quartic objective."""

    diagonal: Array
    c: float = 0.5
    name: str = "f3"

    def value(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.dot(self.diagonal * x, x) + 0.25 * self.c * np.sum(x**4))

    def gradient(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        return self.diagonal * x + self.c * x**3

    def hessian(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        return np.diag(self.diagonal + 3.0 * self.c * x**2)

    def hessian_vector_product(self, x: Array, vector: Array) -> Array:
        x = np.asarray(x, dtype=float)
        vector = np.asarray(vector, dtype=float)
        return (self.diagonal + 3.0 * self.c * x**2) * vector


def experiment3_objective(n: int = 200, c: float = 0.5) -> DiagonalQuarticObjective:
    """Return f3 from the assignment."""

    diagonal = np.logspace(-3, 1, n)
    return DiagonalQuarticObjective(diagonal=diagonal, c=c, name="f3")


def experiment3_initial_point(n: int = 200) -> Array:
    """Return x0 = (1, 2, 1, 2, ...)^T for Experiment 3."""

    x0 = np.ones(n)
    x0[1::2] = 2.0
    return x0
