"""Shared data structures for optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


class Objective(Protocol):
    """Interface required by the optimization routines."""

    def value(self, x: Array) -> float:
        """Return f(x)."""

    def gradient(self, x: Array) -> Array:
        """Return grad f(x)."""

    def hessian(self, x: Array) -> Array:
        """Return Hessian matrix at x."""


@dataclass(frozen=True)
class IterationRecord:
    """One row in the optimization history."""

    iteration: int
    x: Array
    value: float
    gradient_norm: float
    step_size: float | None = None
    direction_norm: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by all optimization algorithms."""

    x: Array
    value: float
    gradient_norm: float
    iterations: int
    converged: bool
    reason: str
    history: tuple[IterationRecord, ...]
