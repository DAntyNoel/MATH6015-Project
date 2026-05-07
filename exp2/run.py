"""Run Experiment 2 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newton_methods import classical_newton, modified_damped_newton
from newton_methods.core import OptimizationResult
from newton_methods.experiment_utils import (
    configure_experiment,
    format_float,
    history_points,
    positive_for_log,
    print_result,
    result_line,
)
from newton_methods.problems import experiment2_objective

OUTPUT_DIR = configure_experiment(__file__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


X0 = np.array([0.1, 0.1])
TARGET_MINIMIZER = np.array([0.0, 1.0])
SADDLE_POINT = np.array([0.0, 0.0])
DELTA = 1e-3


def main() -> None:
    objective = experiment2_objective()
    classical = classical_newton(objective, X0)
    modified = modified_damped_newton(objective, X0, delta=DELTA)

    classical_points = history_points(classical)
    modified_points = history_points(modified)
    classical_distances = distances_to_target(classical_points)
    modified_distances = distances_to_target(modified_points)

    plot_paths(classical_points, modified_points, OUTPUT_DIR / "experiment2_paths.png")
    plot_distances(classical_distances, modified_distances, OUTPUT_DIR / "experiment2_distance_to_minimizer.png")
    write_iteration_history(classical, modified, classical_distances, modified_distances, OUTPUT_DIR / "iteration_history.csv")
    write_summary(classical, modified, OUTPUT_DIR / "summary.txt")

    print(f"Experiment 2 complete. Outputs written to {OUTPUT_DIR}")
    print_result("classical_newton", classical)
    print_result("modified_damped_newton", modified)


def distances_to_target(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - TARGET_MINIMIZER, axis=1)


def plot_paths(classical_points: np.ndarray, modified_points: np.ndarray, output_path: pathlib.Path) -> None:
    x_grid = np.linspace(-0.2, 0.2, 260)
    y_grid = np.linspace(-1.35, 1.35, 360)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = 0.5 * xx**2 + 0.25 * yy**4 - 0.5 * yy**2

    fig, ax = plt.subplots(figsize=(6.2, 7.0), constrained_layout=True)
    ax.contour(xx, yy, zz, levels=np.linspace(float(zz.min()), float(zz.max()), 36), cmap="viridis", linewidths=0.8)
    ax.plot(classical_points[:, 0], classical_points[:, 1], "o-", markersize=4.0, linewidth=1.5, label="Classical Newton")
    ax.plot(modified_points[:, 0], modified_points[:, 1], "s-", markersize=3.8, linewidth=1.5, label=f"Modified damped Newton, delta={DELTA:g}")
    ax.scatter(*SADDLE_POINT, color="black", marker="x", s=70, label="saddle")
    ax.scatter(*TARGET_MINIMIZER, color="black", marker="*", s=130, label="target minimizer")
    ax.scatter([0.0], [-1.0], color="gray", marker="*", s=90, label="other minimizer")
    ax.set(title="Experiment 2 Iteration Paths", xlabel="x", ylabel="y")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_distances(classical_distances: np.ndarray, modified_distances: np.ndarray, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(positive_for_log(classical_distances), "o-", markersize=4.0, linewidth=1.5, label="Classical Newton")
    ax.semilogy(positive_for_log(modified_distances), "s-", markersize=3.8, linewidth=1.5, label="Modified damped Newton")
    ax.set(title="Distance to Local Minimizer (0, 1)", xlabel="Iteration k", ylabel="||x_k - (0, 1)^T||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_iteration_history(
    classical: OptimizationResult,
    modified: OptimizationResult,
    classical_distances: np.ndarray,
    modified_distances: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    columns = [
        "method",
        "k",
        "x",
        "y",
        "f",
        "gradient_norm",
        "distance_to_(0,1)",
        "step_size",
        "direction_norm",
        "lambda_min",
        "tau",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        write_result_rows(writer, "classical_newton", classical, classical_distances)
        write_result_rows(writer, "modified_damped_newton", modified, modified_distances)


def write_result_rows(csv_writer, method: str, result: OptimizationResult, distances: np.ndarray) -> None:
    for record, distance in zip(result.history, distances, strict=True):
        csv_writer.writerow(
            [
                method,
                record.iteration,
                f"{record.x[0]:.16e}",
                f"{record.x[1]:.16e}",
                f"{record.value:.16e}",
                f"{record.gradient_norm:.16e}",
                f"{distance:.16e}",
                format_float(record.step_size),
                format_float(record.direction_norm),
                format_float(record.metadata.get("lambda_min")),
                format_float(record.metadata.get("tau")),
            ]
        )


def write_summary(classical: OptimizationResult, modified: OptimizationResult, output_path: pathlib.Path) -> None:
    lines = [
        "Experiment 2 Summary",
        "",
        f"Initial point: {X0.tolist()}",
        f"Saddle point: {SADDLE_POINT.tolist()}",
        f"Target local minimizer: {TARGET_MINIMIZER.tolist()}",
        "",
        result_line("Classical Newton", classical),
        result_line("Modified damped Newton", modified),
        "",
        "Observation:",
        "Classical Newton converges to the saddle point near (0, 0).",
        "Modified damped Newton converges to the local minimizer near (0, 1).",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
