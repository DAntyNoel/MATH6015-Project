"""Run Experiment 1 from the Newton-type methods assignment."""

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

from newton_methods import classical_newton, fixed_step_gradient_descent
from newton_methods.core import OptimizationResult
from newton_methods.problems import experiment1_objective

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


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

    paths_output = OUTPUT_DIR / ("experiment1_paths.png" if HAS_MATPLOTLIB else "experiment1_paths.svg")
    errors_output = OUTPUT_DIR / ("experiment1_errors.png" if HAS_MATPLOTLIB else "experiment1_errors.svg")

    plot_paths(objective.value, gd_points, newton_points, paths_output)
    plot_errors(gd_errors, newton_errors, errors_output)
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
    if not HAS_MATPLOTLIB:
        plot_paths_svg(value_fn, gradient_descent_points, newton_points, output_path)
        return

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
    if not HAS_MATPLOTLIB:
        plot_errors_svg(gradient_descent_errors, newton_errors, output_path)
        return

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


def plot_paths_svg(
    value_fn,
    gradient_descent_points: np.ndarray,
    newton_points: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    grid = np.linspace(-1.25, 1.25, 150)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.empty_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = value_fn(np.array([xx[i, j], yy[i, j]]))
    levels = np.geomspace(1e-3, float(zz.max()), 28)
    segments = contour_segments(grid, grid, zz, levels)

    canvas = SvgCanvas(output_path, x_min=-1.25, x_max=1.25, y_min=-1.25, y_max=1.25, width=720, height=720)
    canvas.begin("Experiment 1 Iteration Paths", "x", "y")
    for level, segment in segments:
        shade = int(205 - 125 * (np.log(level) - np.log(levels.min())) / (np.log(levels.max()) - np.log(levels.min())))
        canvas.line(segment[0], segment[1], color=f"rgb({shade},{shade},{shade})", width=0.8, opacity=0.75)
    canvas.polyline(gradient_descent_points, color="#1f77b4", width=2.2)
    canvas.markers(gradient_descent_points, color="#1f77b4", shape="circle")
    canvas.polyline(newton_points, color="#d62728", width=2.2)
    canvas.markers(newton_points, color="#d62728", shape="square")
    canvas.marker(X_STAR, color="#111111", shape="star", size=11)
    canvas.legend(
        [
            (f"Gradient descent, alpha={GD_STEP_SIZE}", "#1f77b4"),
            ("Classical Newton", "#d62728"),
            ("x*", "#111111"),
        ]
    )
    canvas.finish()


def plot_errors_svg(
    gradient_descent_errors: np.ndarray,
    newton_errors: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    gd_points = np.column_stack(
        [np.arange(len(gradient_descent_errors)), np.log10(positive_for_log(gradient_descent_errors))]
    )
    newton_points = np.column_stack([np.arange(len(newton_errors)), np.log10(positive_for_log(newton_errors))])
    all_points = np.vstack([gd_points, newton_points])
    y_min = float(np.floor(all_points[:, 1].min()) - 0.2)
    y_max = float(np.ceil(all_points[:, 1].max()) + 0.2)
    x_max = float(max(len(gradient_descent_errors), len(newton_errors)) - 1)

    canvas = SvgCanvas(output_path, x_min=0.0, x_max=x_max, y_min=y_min, y_max=y_max, width=760, height=540)
    canvas.begin("Experiment 1 Error Curves", "iteration k", "log10(||x_k - x*||_2)")
    canvas.polyline(gd_points, color="#1f77b4", width=2.2)
    canvas.markers(gd_points, color="#1f77b4", shape="circle")
    canvas.polyline(newton_points, color="#d62728", width=2.2)
    canvas.markers(newton_points, color="#d62728", shape="square")
    canvas.legend(
        [
            (f"Gradient descent, alpha={GD_STEP_SIZE}", "#1f77b4"),
            ("Classical Newton", "#d62728"),
        ]
    )
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
        self.x_max = x_max if x_max > x_min else x_min + 1.0
        self.y_min = y_min
        self.y_max = y_max if y_max > y_min else y_min + 1.0
        self.width = width
        self.height = height
        self.plot_left = 74
        self.plot_top = 58
        self.plot_width = width - 120
        self.plot_height = height - 140
        self.elements: list[str] = []

    def begin(self, title: str, x_label: str, y_label: str) -> None:
        self.elements.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
        )
        self.elements.append('<rect width="100%" height="100%" fill="white"/>')
        self.text(self.width / 2, 30, title, size=18, anchor="middle", weight="700")
        self.text(self.plot_left + self.plot_width / 2, self.height - 30, x_label, size=13, anchor="middle")
        self.text(self.plot_left, 50, y_label, size=13, anchor="start")
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
        elif shape == "star":
            self.text(x, y + size / 2, "*", size=int(size * 3), anchor="middle", weight="700", color=color)
        else:
            self.elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size / 2:.2f}" fill="{color}"/>')

    def legend(self, items: list[tuple[str, str]]) -> None:
        x = self.plot_left + 14
        y = self.plot_top + self.plot_height + 30
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
