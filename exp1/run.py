"""Run Experiment 1 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from newton_methods import classical_newton, fixed_step_gradient_descent
from newton_methods.core import OptimizationResult
from newton_methods.problems import experiment1_objective


X_STAR = np.array([0.0, 0.0])
X0 = np.array([1.0, 1.0])
GD_STEP_SIZE = 0.2


def main() -> None:
    objective = experiment1_objective()

    gradient_descent = fixed_step_gradient_descent(
        objective,
        X0,
        step_size=GD_STEP_SIZE,
        tolerance=1e-10,
        max_iterations=500,
    )
    newton = classical_newton(
        objective,
        X0,
        tolerance=1e-10,
        max_iterations=100,
    )

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


def history_points(result: OptimizationResult) -> np.ndarray:
    return np.vstack([record.x for record in result.history])


def errors(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - X_STAR, axis=1)


def observed_orders(error_values: np.ndarray) -> list[float]:
    orders: list[float] = [float("nan"), float("nan")]
    for k in range(1, len(error_values) - 1):
        previous_error = error_values[k - 1]
        current_error = error_values[k]
        next_error = error_values[k + 1]
        if previous_error <= 0.0 or current_error <= 0.0 or next_error <= 0.0:
            orders.append(float("nan"))
            continue

        denominator = np.log(current_error / previous_error)
        if abs(denominator) < 1e-15:
            orders.append(float("nan"))
            continue

        orders.append(float(np.log(next_error / current_error) / denominator))
    return orders[: len(error_values)]


def plot_paths(
    value_fn,
    gradient_descent_points: np.ndarray,
    newton_points: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    grid = np.linspace(-1.25, 1.25, 300)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.empty_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = value_fn(np.array([xx[i, j], yy[i, j]]))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    levels = np.geomspace(1e-3, float(zz.max()), 35)
    ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.8)

    ax.plot(
        gradient_descent_points[:, 0],
        gradient_descent_points[:, 1],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        color="#1f77b4",
        label=f"Gradient descent, alpha={GD_STEP_SIZE}",
    )
    ax.plot(
        newton_points[:, 0],
        newton_points[:, 1],
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        color="#d62728",
        label="Classical Newton",
    )
    ax.scatter([X_STAR[0]], [X_STAR[1]], color="black", marker="*", s=120, label="x*")
    ax.set_title("Experiment 1 Iteration Paths")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_errors(
    gradient_descent_errors: np.ndarray,
    newton_errors: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(
        np.arange(len(gradient_descent_errors)),
        positive_for_log(gradient_descent_errors),
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        label=f"Gradient descent, alpha={GD_STEP_SIZE}",
    )
    ax.semilogy(
        np.arange(len(newton_errors)),
        positive_for_log(newton_errors),
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        label="Classical Newton",
    )
    ax.set_title("Experiment 1 Error Curves")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("||x_k - x*||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def write_order_table(error_values: np.ndarray, orders: list[float], output_path: pathlib.Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "error", "q_k"])
        for k, error_value in enumerate(error_values):
            q_value = orders[k] if k < len(orders) else float("nan")
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
        format_result("Gradient descent", gradient_descent),
        format_result("Classical Newton", newton),
        "",
        "Observed Newton convergence orders:",
    ]
    if finite_orders:
        lines.extend(f"  q = {q:.8g}" for q in finite_orders)
        lines.append(f"  last finite q = {finite_orders[-1]:.8g}")
    else:
        lines.append("  No finite q_k values were available.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_result(name: str, result: OptimizationResult) -> None:
    print(
        f"{name}: converged={result.converged}, iterations={result.iterations}, "
        f"f={result.value:.6e}, ||grad||={result.gradient_norm:.6e}, x={result.x}"
    )


def format_result(name: str, result: OptimizationResult) -> str:
    return (
        f"{name}: converged={result.converged}, iterations={result.iterations}, "
        f"f={result.value:.16e}, ||grad||={result.gradient_norm:.16e}, x={result.x.tolist()}"
    )


if __name__ == "__main__":
    main()
