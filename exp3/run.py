"""Run Experiment 3 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import pathlib
import sys
from collections.abc import Callable

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newton_methods import newton_cg
from newton_methods.core import OptimizationResult
from newton_methods.experiment_utils import configure_experiment, format_float, format_int, positive_for_log
from newton_methods.problems import experiment3_initial_point, experiment3_objective

OUTPUT_DIR = configure_experiment(__file__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 200
C = 0.5
TOLERANCE = 1e-10
MAX_OUTER_ITERATIONS = 100

ETA_RULES: dict[str, Callable[[float, int], float]] = {
    "eta_0.8": lambda _norm, _k: 0.8,
    "eta_0.1": lambda _norm, _k: 0.1,
    "eta_adaptive": lambda norm, _k: min(0.5, norm),
}
COLORS = {"eta_0.8": "#1f77b4", "eta_0.1": "#d62728", "eta_adaptive": "#2ca02c"}
LABELS = {
    "eta_0.8": "eta_k = 0.8",
    "eta_0.1": "eta_k = 0.1",
    "eta_adaptive": "eta_k = min(0.5, ||grad||)",
}


def main() -> None:
    objective = experiment3_objective(n=N, c=C)
    x0 = experiment3_initial_point(n=N)
    results = {
        name: newton_cg(
            objective,
            objective.hessian_vector_product,
            x0,
            eta_rule=eta_rule,
            tolerance=TOLERANCE,
            max_iterations=MAX_OUTER_ITERATIONS,
            max_cg_iterations=N,
        )
        for name, eta_rule in ETA_RULES.items()
    }

    plot_gradient_norms(results, OUTPUT_DIR / "gradient_norms.png")
    plot_cumulative_cg_iterations(results, OUTPUT_DIR / "cumulative_cg_iterations.png")
    write_iteration_history(results, OUTPUT_DIR / "iteration_history.csv")
    write_summary(results, OUTPUT_DIR / "summary.txt")

    print(f"Experiment 3 complete. Outputs written to {OUTPUT_DIR}")
    for name, result in results.items():
        print(
            f"{name}: converged={result.converged}, outer_iterations={result.iterations}, "
            f"total_cg={total_cg(result)}, ||grad||={result.gradient_norm:.6e}, f={result.value:.6e}"
        )


def gradient_norms(result: OptimizationResult) -> np.ndarray:
    return np.array([record.gradient_norm for record in result.history])


def cumulative_cg_iterations(result: OptimizationResult) -> np.ndarray:
    return np.array([record.metadata.get("cumulative_cg_iterations", 0) for record in result.history], dtype=int)


def total_cg(result: OptimizationResult) -> int:
    return int(cumulative_cg_iterations(result)[-1])


def plot_gradient_norms(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, result in results.items():
        values = gradient_norms(result)
        ax.semilogy(positive_for_log(values), "o-", markersize=3.8, linewidth=1.5, color=COLORS[name], label=LABELS[name])
    ax.set(title="Experiment 3 Gradient Norms", xlabel="Outer Newton iteration k", ylabel="||grad f(x_k)||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cumulative_cg_iterations(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, result in results.items():
        ax.plot(cumulative_cg_iterations(result), "o-", markersize=3.8, linewidth=1.5, color=COLORS[name], label=LABELS[name])
    ax.set(title="Experiment 3 Cumulative CG Iterations", xlabel="Outer Newton iteration k", ylabel="Cumulative CG iterations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_iteration_history(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    columns = [
        "eta_rule",
        "outer_iteration",
        "f",
        "gradient_norm",
        "eta",
        "cg_iterations",
        "cumulative_cg_iterations",
        "linear_residual_norm",
        "step_size",
        "direction_norm",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, result in results.items():
            for record in result.history:
                writer.writerow(
                    [
                        name,
                        record.iteration,
                        f"{record.value:.16e}",
                        f"{record.gradient_norm:.16e}",
                        format_float(record.metadata.get("eta")),
                        format_int(record.metadata.get("cg_iterations")),
                        format_int(record.metadata.get("cumulative_cg_iterations")),
                        format_float(record.metadata.get("linear_residual_norm")),
                        format_float(record.step_size),
                        format_float(record.direction_norm),
                    ]
                )


def write_summary(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    lines = ["Experiment 3 Summary", "", f"n: {N}", f"c: {C}", f"tolerance: {TOLERANCE:.1e}", ""]
    lines.extend(format_result(name, result) for name, result in results.items())
    lines.extend(
        [
            "",
            "Observation:",
            "eta_k = 0.8 uses fewer CG iterations per outer step but needs more outer Newton iterations.",
            "eta_k = 0.1 solves the Newton system more accurately and usually reduces the gradient faster per outer step.",
            "The adaptive eta rule starts cheaply and tightens the inner solve as the iterates approach the minimizer.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_result(name: str, result: OptimizationResult) -> str:
    return (
        f"{name}: converged={result.converged}, outer_iterations={result.iterations}, "
        f"total_cg_iterations={total_cg(result)}, f={result.value:.16e}, "
        f"||grad||={result.gradient_norm:.16e}, reason={result.reason}"
    )


if __name__ == "__main__":
    main()
