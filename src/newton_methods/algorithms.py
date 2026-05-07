"""Basic gradient descent and Newton-type methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .core import Array, IterationRecord, Objective, OptimizationResult
from .line_search import armijo_backtracking


def fixed_step_gradient_descent(
    objective: Objective,
    x0: Array,
    *,
    step_size: float,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> OptimizationResult:
    """Minimize an objective using fixed-step gradient descent."""

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    x = _as_float_vector(x0)
    history: list[IterationRecord] = []

    for iteration in range(max_iterations + 1):
        value = objective.value(x)
        gradient = objective.gradient(x)
        gradient_norm = float(np.linalg.norm(gradient))
        record = IterationRecord(
            iteration=iteration,
            x=x.copy(),
            value=value,
            gradient_norm=gradient_norm,
        )

        if gradient_norm < tolerance:
            history.append(record)
            return _result(objective, x, iteration, True, "gradient tolerance reached", history)
        if iteration == max_iterations:
            history.append(record)
            break

        direction = -gradient
        history.append(_with_step_info(record, step_size=step_size, direction=direction))
        x = x + step_size * direction

    return _result(objective, x, max_iterations, False, "maximum iterations reached", history)


def classical_newton(
    objective: Objective,
    x0: Array,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> OptimizationResult:
    """Minimize an objective using the classical full-step Newton method."""

    x = _as_float_vector(x0)
    history: list[IterationRecord] = []

    for iteration in range(max_iterations + 1):
        value = objective.value(x)
        gradient = objective.gradient(x)
        gradient_norm = float(np.linalg.norm(gradient))
        record = IterationRecord(
            iteration=iteration,
            x=x.copy(),
            value=value,
            gradient_norm=gradient_norm,
        )

        if gradient_norm < tolerance:
            history.append(record)
            return _result(objective, x, iteration, True, "gradient tolerance reached", history)
        if iteration == max_iterations:
            history.append(record)
            break

        hessian = objective.hessian(x)
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError as exc:
            history.append(record)
            return _result(objective, x, iteration, False, f"linear solve failed: {exc}", history)

        history.append(_with_step_info(record, step_size=1.0, direction=direction))
        x = x + direction

    return _result(objective, x, max_iterations, False, "maximum iterations reached", history)


def modified_damped_newton(
    objective: Objective,
    x0: Array,
    *,
    delta: float = 1e-3,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    armijo_rho: float = 0.5,
    armijo_c1: float = 1e-4,
) -> OptimizationResult:
    """Minimize an objective using Hessian modification and Armijo damping."""

    if delta <= 0.0:
        raise ValueError("delta must be positive")

    x = _as_float_vector(x0)
    history: list[IterationRecord] = []

    for iteration in range(max_iterations + 1):
        value = objective.value(x)
        gradient = objective.gradient(x)
        gradient_norm = float(np.linalg.norm(gradient))
        record = IterationRecord(
            iteration=iteration,
            x=x.copy(),
            value=value,
            gradient_norm=gradient_norm,
        )

        if gradient_norm < tolerance:
            history.append(record)
            return _result(objective, x, iteration, True, "gradient tolerance reached", history)
        if iteration == max_iterations:
            history.append(record)
            break

        hessian = objective.hessian(x)
        lambda_min = float(np.linalg.eigvalsh(hessian).min())
        tau = max(0.0, delta - lambda_min)
        modified_hessian = hessian + tau * np.eye(x.size)

        try:
            direction = np.linalg.solve(modified_hessian, -gradient)
        except np.linalg.LinAlgError as exc:
            history.append(record)
            return _result(objective, x, iteration, False, f"linear solve failed: {exc}", history)

        try:
            step = armijo_backtracking(
                objective,
                x,
                direction,
                rho=armijo_rho,
                c1=armijo_c1,
            )
        except ValueError as exc:
            history.append(record)
            return _result(objective, x, iteration, False, f"line search failed: {exc}", history)

        history.append(
            _with_step_info(
                record,
                step_size=step,
                direction=direction,
                metadata={"lambda_min": lambda_min, "tau": tau},
            )
        )
        x = x + step * direction

    return _result(objective, x, max_iterations, False, "maximum iterations reached", history)


def conjugate_gradient(
    matvec: Callable[[Array], Array],
    rhs: Array,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, int, float, bool]:
    """Approximately solve A x = rhs using conjugate gradient."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    x = np.zeros_like(rhs, dtype=float)
    residual = rhs - matvec(x)
    direction = residual.copy()
    residual_norm_squared = float(np.dot(residual, residual))

    if np.sqrt(residual_norm_squared) <= tolerance:
        return x, 0, float(np.sqrt(residual_norm_squared)), True

    for iteration in range(1, max_iterations + 1):
        matvec_direction = matvec(direction)
        curvature = float(np.dot(direction, matvec_direction))
        if curvature <= 0.0:
            return x, iteration - 1, float(np.sqrt(residual_norm_squared)), False

        step = residual_norm_squared / curvature
        x = x + step * direction
        residual = residual - step * matvec_direction
        next_residual_norm_squared = float(np.dot(residual, residual))
        residual_norm = float(np.sqrt(next_residual_norm_squared))

        if residual_norm <= tolerance:
            return x, iteration, residual_norm, True

        beta = next_residual_norm_squared / residual_norm_squared
        direction = residual + beta * direction
        residual_norm_squared = next_residual_norm_squared

    return x, max_iterations, float(np.sqrt(residual_norm_squared)), False


def newton_cg(
    objective: Objective,
    hessian_vector_product: Callable[[Array, Array], Array],
    x0: Array,
    *,
    eta_rule: Callable[[float, int], float],
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    max_cg_iterations: int | None = None,
    armijo_rho: float = 0.5,
    armijo_c1: float = 1e-4,
) -> OptimizationResult:
    """Minimize an objective using inexact Newton steps computed by CG."""

    x = _as_float_vector(x0)
    history: list[IterationRecord] = []
    cumulative_cg_iterations = 0

    for iteration in range(max_iterations + 1):
        value = objective.value(x)
        gradient = objective.gradient(x)
        gradient_norm = float(np.linalg.norm(gradient))
        record = IterationRecord(
            iteration=iteration,
            x=x.copy(),
            value=value,
            gradient_norm=gradient_norm,
            metadata={"cumulative_cg_iterations": cumulative_cg_iterations},
        )

        if gradient_norm < tolerance:
            history.append(record)
            return _result(objective, x, iteration, True, "gradient tolerance reached", history)
        if iteration == max_iterations:
            history.append(record)
            break

        eta = float(eta_rule(gradient_norm, iteration))
        if not 0.0 <= eta < 1.0:
            raise ValueError("eta_rule must return a value in [0, 1)")

        cg_limit = x.size if max_cg_iterations is None else max_cg_iterations
        direction, cg_iterations, residual_norm, cg_converged = conjugate_gradient(
            lambda vector: hessian_vector_product(x, vector),
            -gradient,
            tolerance=eta * gradient_norm,
            max_iterations=cg_limit,
        )
        cumulative_cg_iterations += cg_iterations

        try:
            step = armijo_backtracking(
                objective,
                x,
                direction,
                rho=armijo_rho,
                c1=armijo_c1,
            )
        except ValueError as exc:
            history.append(record)
            return _result(objective, x, iteration, False, f"line search failed: {exc}", history)

        history.append(
            _with_step_info(
                record,
                step_size=step,
                direction=direction,
                metadata={
                    "eta": eta,
                    "cg_iterations": cg_iterations,
                    "cumulative_cg_iterations": cumulative_cg_iterations,
                    "linear_residual_norm": residual_norm,
                    "cg_converged": cg_converged,
                },
            )
        )
        x = x + step * direction

    return _result(objective, x, max_iterations, False, "maximum iterations reached", history)


def _as_float_vector(x: Array) -> Array:
    return np.asarray(x, dtype=float).reshape(-1).copy()


def _result(
    objective: Objective,
    x: Array,
    iterations: int,
    converged: bool,
    reason: str,
    history: list[IterationRecord],
) -> OptimizationResult:
    gradient_norm = float(np.linalg.norm(objective.gradient(x)))
    return OptimizationResult(
        x=x.copy(),
        value=objective.value(x),
        gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged,
        reason=reason,
        history=tuple(history),
    )


def _with_step_info(
    record: IterationRecord,
    *,
    step_size: float | None,
    direction: Array,
    metadata: dict[str, float] | None = None,
) -> IterationRecord:
    return IterationRecord(
        iteration=record.iteration,
        x=record.x,
        value=record.value,
        gradient_norm=record.gradient_norm,
        step_size=step_size,
        direction_norm=float(np.linalg.norm(direction)),
        metadata={} if metadata is None else metadata,
    )
