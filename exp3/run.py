"""Run Experiment 3 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import html
import os
import pathlib
import sys
from collections.abc import Callable

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib-cache"
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
MPLCONFIG_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from newton_methods import newton_cg
from newton_methods.core import OptimizationResult
from newton_methods.problems import experiment3_initial_point, experiment3_objective

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


N = 200
C = 0.5
TOLERANCE = 1e-10
MAX_OUTER_ITERATIONS = 100


ETA_RULES: dict[str, Callable[[float, int], float]] = {
    "eta_0.8": lambda _gradient_norm, _iteration: 0.8,
    "eta_0.1": lambda _gradient_norm, _iteration: 0.1,
    "eta_adaptive": lambda gradient_norm, _iteration: min(0.5, gradient_norm),
}


COLORS = {
    "eta_0.8": "#1f77b4",
    "eta_0.1": "#d62728",
    "eta_adaptive": "#2ca02c",
}


def main() -> None:
    objective = experiment3_objective(n=N, c=C)
    x0 = experiment3_initial_point(n=N)

    results: dict[str, OptimizationResult] = {}
    for name, eta_rule in ETA_RULES.items():
        results[name] = newton_cg(
            objective,
            objective.hessian_vector_product,
            x0,
            eta_rule=eta_rule,
            tolerance=TOLERANCE,
            max_iterations=MAX_OUTER_ITERATIONS,
            max_cg_iterations=N,
        )

    gradient_plot = OUTPUT_DIR / ("gradient_norms.png" if HAS_MATPLOTLIB else "gradient_norms.svg")
    cg_plot = OUTPUT_DIR / ("cumulative_cg_iterations.png" if HAS_MATPLOTLIB else "cumulative_cg_iterations.svg")

    plot_gradient_norms(results, gradient_plot)
    plot_cumulative_cg_iterations(results, cg_plot)
    write_iteration_history(results, OUTPUT_DIR / "iteration_history.csv")
    write_summary(results, OUTPUT_DIR / "summary.txt")

    print(f"Experiment 3 complete. Outputs written to {OUTPUT_DIR}")
    for name, result in results.items():
        total_cg = int(cumulative_cg_iterations(result)[-1])
        print(
            f"{name}: converged={result.converged}, outer_iterations={result.iterations}, "
            f"total_cg={total_cg}, ||grad||={result.gradient_norm:.6e}, f={result.value:.6e}"
        )


def gradient_norms(result: OptimizationResult) -> np.ndarray:
    return np.array([record.gradient_norm for record in result.history], dtype=float)


def cumulative_cg_iterations(result: OptimizationResult) -> np.ndarray:
    return np.array([record.metadata.get("cumulative_cg_iterations", 0) for record in result.history], dtype=int)


def plot_gradient_norms(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    if not HAS_MATPLOTLIB:
        series = {
            name: np.column_stack([np.arange(len(gradient_norms(result))), np.log10(positive_for_log(gradient_norms(result)))])
            for name, result in results.items()
        }
        write_svg_line_plot(
            series,
            output_path,
            title="Experiment 3 Gradient Norms",
            y_label="log10(||grad f(x_k)||_2)",
        )
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, result in results.items():
        values = gradient_norms(result)
        ax.semilogy(
            np.arange(len(values)),
            positive_for_log(values),
            marker="o",
            markersize=3.8,
            linewidth=1.5,
            color=COLORS[name],
            label=label_for(name),
        )
    ax.set_title("Experiment 3 Gradient Norms")
    ax.set_xlabel("Outer Newton iteration k")
    ax.set_ylabel("||grad f(x_k)||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cumulative_cg_iterations(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    if not HAS_MATPLOTLIB:
        series = {
            name: np.column_stack([np.arange(len(cumulative_cg_iterations(result))), cumulative_cg_iterations(result)])
            for name, result in results.items()
        }
        write_svg_line_plot(
            series,
            output_path,
            title="Experiment 3 Cumulative CG Iterations",
            y_label="cumulative CG iterations",
        )
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, result in results.items():
        values = cumulative_cg_iterations(result)
        ax.plot(
            np.arange(len(values)),
            values,
            marker="o",
            markersize=3.8,
            linewidth=1.5,
            color=COLORS[name],
            label=label_for(name),
        )
    ax.set_title("Experiment 3 Cumulative CG Iterations")
    ax.set_xlabel("Outer Newton iteration k")
    ax.set_ylabel("Cumulative CG iterations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_iteration_history(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for name, result in results.items():
            for record in result.history:
                writer.writerow(
                    [
                        name,
                        record.iteration,
                        f"{record.value:.16e}",
                        f"{record.gradient_norm:.16e}",
                        format_optional(record.metadata.get("eta")),
                        format_count(record.metadata.get("cg_iterations")),
                        format_count(record.metadata.get("cumulative_cg_iterations")),
                        format_optional(record.metadata.get("linear_residual_norm")),
                        format_optional(record.step_size),
                        format_optional(record.direction_norm),
                    ]
                )


def write_summary(results: dict[str, OptimizationResult], output_path: pathlib.Path) -> None:
    lines = [
        "Experiment 3 Summary",
        "",
        f"n: {N}",
        f"c: {C}",
        f"tolerance: {TOLERANCE:.1e}",
        "",
    ]
    for name, result in results.items():
        lines.append(format_result(name, result))
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
    total_cg = int(cumulative_cg_iterations(result)[-1])
    return (
        f"{name}: converged={result.converged}, outer_iterations={result.iterations}, "
        f"total_cg_iterations={total_cg}, f={result.value:.16e}, "
        f"||grad||={result.gradient_norm:.16e}, reason={result.reason}"
    )


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.16e}"


def format_count(value: object) -> str:
    if value is None:
        return ""
    return str(int(value))


def label_for(name: str) -> str:
    labels = {
        "eta_0.8": "eta_k = 0.8",
        "eta_0.1": "eta_k = 0.1",
        "eta_adaptive": "eta_k = min(0.5, ||grad||)",
    }
    return labels[name]


def write_svg_line_plot(
    series: dict[str, np.ndarray],
    output_path: pathlib.Path,
    *,
    title: str,
    y_label: str,
) -> None:
    all_points = np.vstack(list(series.values()))
    x_min = 0.0
    x_max = float(np.max(all_points[:, 0]))
    y_min = float(np.min(all_points[:, 1]))
    y_max = float(np.max(all_points[:, 1]))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    canvas = SvgCanvas(output_path, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    canvas.begin(title, y_label)
    for name, points in series.items():
        canvas.polyline(points, color=COLORS[name])
        canvas.markers(points, color=COLORS[name])
    canvas.legend([(label_for(name), COLORS[name]) for name in series])
    canvas.finish()


class SvgCanvas:
    def __init__(
        self,
        output_path: pathlib.Path,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        self.output_path = output_path
        self.x_min = x_min
        self.x_max = x_max if x_max > x_min else x_min + 1.0
        self.y_min = y_min
        self.y_max = y_max if y_max > y_min else y_min + 1.0
        self.width = 780
        self.height = 540
        self.plot_left = 78
        self.plot_top = 58
        self.plot_width = 650
        self.plot_height = 360
        self.elements: list[str] = []

    def begin(self, title: str, y_label: str) -> None:
        self.elements.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
        )
        self.elements.append('<rect width="100%" height="100%" fill="white"/>')
        self.text(self.width / 2, 30, title, size=18, anchor="middle", weight="700")
        self.text(self.plot_left, 50, y_label, size=13, anchor="start")
        self.text(self.plot_left + self.plot_width / 2, self.height - 24, "Outer Newton iteration k", size=13, anchor="middle")
        self.elements.append(
            f'<rect x="{self.plot_left}" y="{self.plot_top}" width="{self.plot_width}" '
            f'height="{self.plot_height}" fill="#fafafa" stroke="#222" stroke-width="1"/>'
        )
        self.draw_grid()

    def finish(self) -> None:
        self.elements.append("</svg>")
        self.output_path.write_text("\n".join(self.elements) + "\n", encoding="utf-8")

    def draw_grid(self) -> None:
        for t in np.linspace(0.0, 1.0, 6):
            x = self.plot_left + t * self.plot_width
            y = self.plot_top + t * self.plot_height
            self.elements.append(f'<line x1="{x:.2f}" y1="{self.plot_top}" x2="{x:.2f}" y2="{self.plot_top + self.plot_height}" stroke="#ddd"/>')
            self.elements.append(f'<line x1="{self.plot_left}" y1="{y:.2f}" x2="{self.plot_left + self.plot_width}" y2="{y:.2f}" stroke="#ddd"/>')

    def to_screen(self, point: np.ndarray) -> tuple[float, float]:
        x_value, y_value = float(point[0]), float(point[1])
        x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * self.plot_width
        y = self.plot_top + (self.y_max - y_value) / (self.y_max - self.y_min) * self.plot_height
        return x, y

    def polyline(self, points: np.ndarray, *, color: str) -> None:
        screen_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (self.to_screen(point) for point in points))
        self.elements.append(
            f'<polyline points="{screen_points}" stroke="{color}" stroke-width="2.2" fill="none" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def markers(self, points: np.ndarray, *, color: str) -> None:
        for point in points:
            x, y = self.to_screen(point)
            self.elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')

    def legend(self, items: list[tuple[str, str]]) -> None:
        x = self.plot_left
        y = self.plot_top + self.plot_height + 34
        for index, (label, color) in enumerate(items):
            item_y = y + 22 * index
            self.elements.append(f'<line x1="{x}" y1="{item_y}" x2="{x + 28}" y2="{item_y}" stroke="{color}" stroke-width="3"/>')
            self.text(x + 38, item_y + 4, label, size=13, anchor="start")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int,
        anchor: str,
        weight: str = "400",
    ) -> None:
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="#111111" text-anchor="{anchor}">{html.escape(text)}</text>'
        )


if __name__ == "__main__":
    main()
