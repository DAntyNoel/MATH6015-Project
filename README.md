# MATH6015 Homework 7

本仓库实现 Homework 7 中的 Newton 型优化方法实验。

## 1. 程序结构

|  | 代码片段 |
| --- | --- |
| $f(x)$、$\nabla f(x)$、$\nabla^2 f(x)$ | `src/newton_methods/problems.py` |
| 固定步长梯度下降 | `fixed_step_gradient_descent` in `src/newton_methods/algorithms.py` |
| 经典 Newton 方法 | `classical_newton` in `src/newton_methods/algorithms.py` |
| 修正阻尼 Newton 方法 | `modified_damped_newton` in `src/newton_methods/algorithms.py` |
| Armijo 回溯线搜索 | `armijo_backtracking` in `src/newton_methods/line_search.py` |
| 共轭梯度法 CG | `conjugate_gradient` in `src/newton_methods/algorithms.py` |
| Newton-CG / inexact Newton | `newton_cg` in `src/newton_methods/algorithms.py` |
| 实验脚本与图像生成 | `exp1/run.py`、`exp2/run.py`、`exp3/run.py`、`run_all.py` |

## 2. 仓库结构

```text
.
├── Answer-questions-AI-enhanced.md
├── Project_Report.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_all.py
├── exp1/
│   ├── README.md
│   ├── run.py
│   └── outputs/
├── exp2/
│   ├── README.md
│   ├── run.py
│   └── outputs/
├── exp3/
│   ├── README.md
│   ├── run.py
│   └── outputs/
└── src/newton_methods/
    ├── algorithms.py
    ├── core.py
    ├── experiment_utils.py
    ├── line_search.py
    └── problems.py
```

## 3. 使用

在仓库根目录运行：

```bash
python run_all.py
```

该命令会依次运行：

```text
exp1/run.py
exp2/run.py
exp3/run.py
```

输出写入：

```text
exp1/outputs/
exp2/outputs/
exp3/outputs/
```