"""Run Experiment 2 from the Newton-type methods assignment."""

from __future__ import annotations

import csv
import html
import os
import pathlib
import sys

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

from newton_methods import classical_newton, modified_damped_newton
from newton_methods.core import OptimizationResult
from newton_methods.problems import experiment2_objective

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


X0 = np.array([0.1, 0.1])
TARGET_MINIMIZER = np.array([0.0, 1.0])
SADDLE_POINT = np.array([0.0, 0.0])
DELTA = 1e-3


def main() -> None:
    objective = experiment2_objective()

    classical = classical_newton(
        objective,
        X0,
        tolerance=1e-10,
        max_iterations=100,
    )
    modified = modified_damped_newton(
        objective,
        X0,
        delta=DELTA,
        tolerance=1e-10,
        max_iterations=100,
    )

    classical_points = history_points(classical)
    modified_points = history_points(modified)
    classical_distances = distances_to_target(classical_points)
    modified_distances = distances_to_target(modified_points)

    paths_output = OUTPUT_DIR / ("experiment2_paths.png" if HAS_MATPLOTLIB else "experiment2_paths.svg")
    distances_output = OUTPUT_DIR / (
        "experiment2_distance_to_minimizer.png" if HAS_MATPLOTLIB else "experiment2_distance_to_minimizer.svg"
    )

    plot_paths(objective.value, classical_points, modified_points, paths_output)
    plot_distances(
        classical_distances,
        modified_distances,
        distances_output,
    )
    write_iteration_history(
        classical,
        modified,
        classical_distances,
        modified_distances,
        OUTPUT_DIR / "iteration_history.csv",
    )
    write_summary(classical, modified, OUTPUT_DIR / "summary.txt")

    print(f"Experiment 2 complete. Outputs written to {OUTPUT_DIR}")
    print_result("classical_newton", classical)
    print_result("modified_damped_newton", modified)


def history_points(result: OptimizationResult) -> np.ndarray:
    return np.vstack([record.x for record in result.history])


def distances_to_target(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - TARGET_MINIMIZER, axis=1)


def plot_paths(
    value_fn,
    classical_points: np.ndarray,
    modified_points: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    if not HAS_MATPLOTLIB:
        plot_paths_svg(value_fn, classical_points, modified_points, output_path)
        return

    x_grid = np.linspace(-0.2, 0.2, 260)
    y_grid = np.linspace(-1.35, 1.35, 360)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = 0.5 * xx**2 + 0.25 * yy**4 - 0.5 * yy**2

    fig, ax = plt.subplots(figsize=(6.2, 7.0), constrained_layout=True)
    levels = np.linspace(float(zz.min()), float(zz.max()), 36)
    ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.8)

    ax.plot(
        classical_points[:, 0],
        classical_points[:, 1],
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        color="#1f77b4",
        label="Classical Newton",
    )
    ax.plot(
        modified_points[:, 0],
        modified_points[:, 1],
        marker="s",
        markersize=3.8,
        linewidth=1.5,
        color="#d62728",
        label=f"Modified damped Newton, delta={DELTA:g}",
    )
    ax.scatter([SADDLE_POINT[0]], [SADDLE_POINT[1]], color="black", marker="x", s=70, label="saddle")
    ax.scatter([TARGET_MINIMIZER[0]], [TARGET_MINIMIZER[1]], color="black", marker="*", s=130, label="target minimizer")
    ax.scatter([0.0], [-1.0], color="gray", marker="*", s=90, label="other minimizer")
    ax.set_title("Experiment 2 Iteration Paths")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_distances(
    classical_distances: np.ndarray,
    modified_distances: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    if not HAS_MATPLOTLIB:
        plot_distances_svg(classical_distances, modified_distances, output_path)
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(
        np.arange(len(classical_distances)),
        positive_for_log(classical_distances),
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label="Classical Newton",
    )
    ax.semilogy(
        np.arange(len(modified_distances)),
        positive_for_log(modified_distances),
        marker="s",
        markersize=3.8,
        linewidth=1.5,
        label="Modified damped Newton",
    )
    ax.set_title("Distance to Local Minimizer (0, 1)")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("||x_k - (0, 1)^T||_2")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def plot_paths_svg(
    value_fn,
    classical_points: np.ndarray,
    modified_points: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    x_grid = np.linspace(-0.2, 0.2, 130)
    y_grid = np.linspace(-1.35, 1.35, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = 0.5 * xx**2 + 0.25 * yy**4 - 0.5 * yy**2
    levels = np.linspace(float(zz.min()), float(zz.max()), 28)
    segments = contour_segments(x_grid, y_grid, zz, levels)

    canvas = SvgCanvas(output_path, x_min=-0.2, x_max=0.2, y_min=-1.35, y_max=1.35, width=720, height=820)
    canvas.begin("Experiment 2 Iteration Paths")
    for level, segment in segments:
        shade = int(190 - 110 * (level - levels.min()) / (levels.max() - levels.min()))
        canvas.line(segment[0], segment[1], color=f"rgb({shade},{shade},{shade})", width=0.8, opacity=0.7)
    canvas.polyline(classical_points, color="#1f77b4", width=2.2)
    canvas.markers(classical_points, color="#1f77b4", shape="circle")
    canvas.polyline(modified_points, color="#d62728", width=2.2)
    canvas.markers(modified_points, color="#d62728", shape="square")
    canvas.marker(SADDLE_POINT, color="#111111", shape="cross", size=8)
    canvas.marker(TARGET_MINIMIZER, color="#111111", shape="star", size=11)
    canvas.marker(np.array([0.0, -1.0]), color="#777777", shape="star", size=8)
    canvas.legend(
        [
            ("Classical Newton", "#1f77b4"),
            ("Modified damped Newton", "#d62728"),
            ("saddle / minimizers marked in black and gray", "#111111"),
        ]
    )
    canvas.finish()


def plot_distances_svg(
    classical_distances: np.ndarray,
    modified_distances: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    all_values = positive_for_log(np.concatenate([classical_distances, modified_distances]))
    log_values = np.log10(all_values)
    y_min = float(np.floor(log_values.min()) - 0.2)
    y_max = float(np.ceil(log_values.max()) + 0.2)
    x_max = max(len(classical_distances), len(modified_distances)) - 1

    canvas = SvgCanvas(output_path, x_min=0.0, x_max=float(x_max), y_min=y_min, y_max=y_max, width=760, height=520)
    canvas.begin("Distance to Local Minimizer (0, 1)")
    classical_points = np.column_stack([np.arange(len(classical_distances)), np.log10(positive_for_log(classical_distances))])
    modified_points = np.column_stack([np.arange(len(modified_distances)), np.log10(positive_for_log(modified_distances))])
    canvas.polyline(classical_points, color="#1f77b4", width=2.2)
    canvas.markers(classical_points, color="#1f77b4", shape="circle")
    canvas.polyline(modified_points, color="#d62728", width=2.2)
    canvas.markers(modified_points, color="#d62728", shape="square")
    canvas.text(canvas.plot_left, 32, "y-axis: log10(||x_k - (0, 1)^T||_2)", size=13, anchor="start")
    canvas.legend([("Classical Newton", "#1f77b4"), ("Modified damped Newton", "#d62728")])
    canvas.finish()


def contour_segments(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    levels: np.ndarray,
) -> list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]]:
    segments: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for level in levels:
        for j in range(len(y_grid) - 1):
            for i in range(len(x_grid) - 1):
                corners = [
                    ((x_grid[i], y_grid[j]), values[j, i]),
                    ((x_grid[i + 1], y_grid[j]), values[j, i + 1]),
                    ((x_grid[i + 1], y_grid[j + 1]), values[j + 1, i + 1]),
                    ((x_grid[i], y_grid[j + 1]), values[j + 1, i]),
                ]
                points: list[tuple[float, float]] = []
                for first, second in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    point = edge_crossing(corners[first], corners[second], level)
                    if point is not None:
                        points.append(point)
                if len(points) >= 2:
                    segments.append((float(level), (points[0], points[1])))
                if len(points) == 4:
                    segments.append((float(level), (points[2], points[3])))
    return segments


def edge_crossing(
    first: tuple[tuple[float, float], float],
    second: tuple[tuple[float, float], float],
    level: float,
) -> tuple[float, float] | None:
    (x0, y0), z0 = first
    (x1, y1), z1 = second
    if (z0 - level) * (z1 - level) > 0.0 or z0 == z1:
        return None
    if z0 == level:
        return (float(x0), float(y0))
    if z1 == level:
        return (float(x1), float(y1))
    t = float((level - z0) / (z1 - z0))
    if not 0.0 <= t <= 1.0:
        return None
    return (float(x0 + t * (x1 - x0)), float(y0 + t * (y1 - y0)))


class SvgCanvas:
    def __init__(
        self,
        output_path: pathlib.Path,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int,
        height: int,
    ) -> None:
        self.output_path = output_path
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = width
        self.height = height
        self.plot_left = 70
        self.plot_top = 58
        self.plot_width = width - 110
        self.plot_height = height - 125
        self.elements: list[str] = []

    def begin(self, title: str) -> None:
        self.elements.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
        )
        self.elements.append('<rect width="100%" height="100%" fill="white"/>')
        self.text(self.width / 2, 28, title, size=18, anchor="middle", weight="700")
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

    def to_screen(self, point: np.ndarray | tuple[float, float]) -> tuple[float, float]:
        x_value, y_value = float(point[0]), float(point[1])
        x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * self.plot_width
        y = self.plot_top + (self.y_max - y_value) / (self.y_max - self.y_min) * self.plot_height
        return x, y

    def line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
        width: float,
        opacity: float = 1.0,
    ) -> None:
        x0, y0 = self.to_screen(start)
        x1, y1 = self.to_screen(end)
        self.elements.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}" fill="none"/>'
        )

    def polyline(self, points: np.ndarray, *, color: str, width: float) -> None:
        screen_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (self.to_screen(point) for point in points))
        self.elements.append(
            f'<polyline points="{screen_points}" stroke="{color}" stroke-width="{width}" fill="none" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def markers(self, points: np.ndarray, *, color: str, shape: str) -> None:
        for point in points:
            self.marker(point, color=color, shape=shape, size=5)

    def marker(self, point: np.ndarray, *, color: str, shape: str, size: float = 5) -> None:
        x, y = self.to_screen(point)
        if shape == "square":
            self.elements.append(
                f'<rect x="{x - size / 2:.2f}" y="{y - size / 2:.2f}" width="{size:.2f}" height="{size:.2f}" fill="{color}"/>'
            )
        elif shape == "cross":
            self.elements.append(f'<line x1="{x - size:.2f}" y1="{y - size:.2f}" x2="{x + size:.2f}" y2="{y + size:.2f}" stroke="{color}" stroke-width="2"/>')
            self.elements.append(f'<line x1="{x - size:.2f}" y1="{y + size:.2f}" x2="{x + size:.2f}" y2="{y - size:.2f}" stroke="{color}" stroke-width="2"/>')
        elif shape == "star":
            self.text(x, y + size / 2, "*", size=int(size * 3), anchor="middle", weight="700", color=color)
        else:
            self.elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size / 2:.2f}" fill="{color}"/>')

    def legend(self, items: list[tuple[str, str]]) -> None:
        x = self.plot_left + 14
        y = self.plot_top + self.plot_height + 34
        for index, (label, color) in enumerate(items):
            item_y = y + 20 * index
            self.elements.append(f'<line x1="{x}" y1="{item_y}" x2="{x + 24}" y2="{item_y}" stroke="{color}" stroke-width="3"/>')
            self.text(x + 32, item_y + 4, label, size=13, anchor="start")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int,
        anchor: str,
        weight: str = "400",
        color: str = "#111111",
    ) -> None:
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(text)}</text>'
        )


def write_iteration_history(
    classical: OptimizationResult,
    modified: OptimizationResult,
    classical_distances: np.ndarray,
    modified_distances: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
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
                format_optional(record.step_size),
                format_optional(record.direction_norm),
                format_optional(record.metadata.get("lambda_min")),
                format_optional(record.metadata.get("tau")),
            ]
        )


def format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.16e}"


def write_summary(classical: OptimizationResult, modified: OptimizationResult, output_path: pathlib.Path) -> None:
    lines = [
        "Experiment 2 Summary",
        "",
        f"Initial point: {X0.tolist()}",
        f"Saddle point: {SADDLE_POINT.tolist()}",
        f"Target local minimizer: {TARGET_MINIMIZER.tolist()}",
        "",
        format_result("Classical Newton", classical),
        format_result("Modified damped Newton", modified),
        "",
        "Observation:",
        "Classical Newton converges to the saddle point near (0, 0).",
        "Modified damped Newton converges to the local minimizer near (0, 1).",
    ]
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
