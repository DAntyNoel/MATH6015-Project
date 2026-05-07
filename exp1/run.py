"""Run Experiment 1 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newton_methods import classical_newton, fixed_step_gradient_descent
from newton_methods.core import OptimizationResult
from newton_methods.experiment_utils import (
    configure_experiment,
    history_points,
    positive_for_log,
    print_result,
    result_line,
)
from newton_methods.problems import experiment1_objective

OUTPUT_DIR = configure_experiment(__file__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


X_STAR = np.array([0.0, 0.0])
X0 = np.array([1.0, 1.0])
GD_STEP_SIZE = 0.2


def main() -> None:
    objective = experiment1_objective()
    gradient_descent = fixed_step_gradient_descent(objective, X0, step_size=GD_STEP_SIZE, max_iterations=500)
    newton = classical_newton(objective, X0)

    gd_points = history_points(gradient_descent)
    newton_points = history_points(newton)
    gd_errors = errors(gd_points)
    newton_errors = errors(newton_points)
    newton_orders = observed_orders(newton_errors)

    plot_paths(objective.value, gd_points, newton_points, OUTPUT_DIR / "experiment1_paths.png")
    plot_errors(gd_errors, newton_errors, OUTPUT_DIR / "experiment1_errors.png")
    write_order_table(newton_errors, newton_orders, OUTPUT_DIR / "newton_observed_orders.csv")
    write_summary(gradient_descent, newton, newton_orders, OUTPUT_DIR / "summary.txt")

    print(f"Experiment 1 complete. Outputs written to {OUTPUT_DIR}")
    print_result("gradient_descent", gradient_descent)
    print_result("classical_newton", newton)
    finite_orders = [q for q in newton_orders if np.isfinite(q)]
    if finite_orders:
        print(f"last finite Newton q_k: {finite_orders[-1]:.6g}")


def errors(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - X_STAR, axis=1)


def observed_orders(error_values: np.ndarray) -> list[float]:
    orders = [float("nan"), float("nan")]
    for previous_error, current_error, next_error in zip(error_values, error_values[1:], error_values[2:]):
        if min(previous_error, current_error, next_error) <= 0.0:
            orders.append(float("nan"))
            continue
        denominator = np.log(current_error / previous_error)
        orders.append(float("nan") if abs(denominator) < 1e-15 else float(np.log(next_error / current_error) / denominator))
    return orders[: len(error_values)]


def plot_paths(value_fn, gd_points: np.ndarray, newton_points: np.ndarray, output_path: pathlib.Path) -> None:
    grid = np.linspace(-1.25, 1.25, 300)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.vectorize(lambda x, y: value_fn(np.array([x, y])))(xx, yy)

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.contour(xx, yy, zz, levels=np.geomspace(1e-3, float(zz.max()), 35), cmap="viridis", linewidths=0.8)
    ax.plot(gd_points[:, 0], gd_points[:, 1], "o-", markersize=3.5, linewidth=1.5, label=f"Gradient descent, alpha={GD_STEP_SIZE}")
    ax.plot(newton_points[:, 0], newton_points[:, 1], "s-", markersize=4.0, linewidth=1.5, label="Classical Newton")
    ax.scatter(*X_STAR, color="black", marker="*", s=120, label="x*")
    ax.set(title="Experiment 1 Iteration Paths", xlabel="x", ylabel="y", aspect="equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_errors(gd_errors: np.ndarray, newton_errors: np.ndarray, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(positive_for_log(gd_errors), "o-", markersize=3.0, linewidth=1.5, label=f"Gradient descent, alpha={GD_STEP_SIZE}")
    ax.semilogy(positive_for_log(newton_errors), "s-", markersize=4.0, linewidth=1.5, label="Classical Newton")
    ax.set(title="Experiment 1 Error Curves", xlabel="Iteration k", ylabel="||x_k - x*||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_order_table(error_values: np.ndarray, orders: list[float], output_path: pathlib.Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "error", "q_k"])
        for k, (error_value, q_value) in enumerate(zip(error_values, orders, strict=True)):
            writer.writerow([k, f"{error_value:.16e}", f"{q_value:.16e}" if np.isfinite(q_value) else "nan"])


def write_summary(
    gradient_descent: OptimizationResult,
    newton: OptimizationResult,
    newton_orders: list[float],
    output_path: pathlib.Path,
) -> None:
    finite_orders = [q for q in newton_orders if np.isfinite(q)]
    lines = [
        "Experiment 1 Summary",
        "",
        f"Initial point: {X0.tolist()}",
        f"Minimizer: {X_STAR.tolist()}",
        "",
        result_line("Gradient descent", gradient_descent),
        result_line("Classical Newton", newton),
        "",
        "Observed Newton convergence orders:",
        *(f"  q = {q:.8g}" for q in finite_orders),
        f"  last finite q = {finite_orders[-1]:.8g}" if finite_orders else "  No finite q_k values were available.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
