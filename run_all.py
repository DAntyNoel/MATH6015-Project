"""Run all Homework 7 experiments from the repository root."""

from __future__ import annotations

import pathlib
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
EXPERIMENT_SCRIPTS = (
    PROJECT_ROOT / "exp1" / "run.py",
    PROJECT_ROOT / "exp2" / "run.py",
    PROJECT_ROOT / "exp3" / "run.py",
)


def main() -> None:
    for script in EXPERIMENT_SCRIPTS:
        print(f"\n=== Running {script.relative_to(PROJECT_ROOT)} ===")
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
