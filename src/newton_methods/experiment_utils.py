"""Small helpers shared by the experiment scripts."""

from __future__ import annotations

import os
import pathlib

import numpy as np

from .core import OptimizationResult


def configure_experiment(script_file: str) -> pathlib.Path:
    """Prepare cache and output directories before importing Matplotlib."""

    project_root = pathlib.Path(script_file).resolve().parents[1]
    output_dir = pathlib.Path(script_file).resolve().parent / "outputs"

    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))

    pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    pathlib.Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def history_points(result: OptimizationResult) -> np.ndarray:
    return np.vstack([record.x for record in result.history])


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def format_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.16e}"


def format_int(value: object) -> str:
    return "" if value is None else str(int(value))


def result_line(name: str, result: OptimizationResult) -> str:
    return (
        f"{name}: converged={result.converged}, iterations={result.iterations}, "
        f"f={result.value:.16e}, ||grad||={result.gradient_norm:.16e}, x={result.x.tolist()}"
    )


def print_result(name: str, result: OptimizationResult) -> None:
    print(
        f"{name}: converged={result.converged}, iterations={result.iterations}, "
        f"f={result.value:.6e}, ||grad||={result.gradient_norm:.6e}, x={result.x}"
    )
