"""Small sanity checks for the basic optimization framework."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newton_methods import classical_newton, fixed_step_gradient_descent, modified_damped_newton
from newton_methods.problems import FunctionObjective, experiment1_objective


def quadratic_objective() -> FunctionObjective:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])

    def value(x: np.ndarray) -> float:
        return 0.5 * float(x @ matrix @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return matrix @ x

    def hessian(_: np.ndarray) -> np.ndarray:
        return matrix

    return FunctionObjective(value, gradient, hessian, name="quadratic")


def main() -> None:
    quadratic = quadratic_objective()
    x0 = np.array([1.0, -1.0])

    gd = fixed_step_gradient_descent(quadratic, x0, step_size=0.2, max_iterations=200)
    newton = classical_newton(quadratic, x0, max_iterations=10)
    damped = modified_damped_newton(experiment1_objective(), np.array([1.0, 1.0]), max_iterations=30)

    print("gradient_descent", gd.converged, gd.iterations, gd.gradient_norm)
    print("classical_newton", newton.converged, newton.iterations, newton.gradient_norm)
    print("modified_damped_newton_f1", damped.converged, damped.iterations, damped.gradient_norm)

    assert gd.converged
    assert newton.converged
    assert damped.converged


if __name__ == "__main__":
    main()
