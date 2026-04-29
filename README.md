# MATH6015 Project

Programming assignment framework for Newton-type optimization methods.

## Current Structure

- `src/newton_methods/core.py`: shared objective interface and result/history records.
- `src/newton_methods/line_search.py`: Armijo backtracking line search.
- `src/newton_methods/algorithms.py`: fixed-step gradient descent, classical Newton, and modified damped Newton.
- `src/newton_methods/problems.py`: reusable objective wrappers and assignment functions.
- `examples/smoke_check.py`: quick sanity check for the basic framework.

## Run The Smoke Check

```bash
python examples/smoke_check.py
```

The main algorithms are implemented directly. Standard linear algebra routines from NumPy
are used for vector operations, linear solves, and Hessian eigenvalue checks.
