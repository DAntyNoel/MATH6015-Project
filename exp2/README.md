# 实验 2：经典 Newton 方法可能收敛到鞍点

本目录实现作业 PDF 中的 Experiment 2，用于比较经典 Newton 方法和修改阻尼 Newton 方法在非凸函数

```text
f2(x, y) = 0.5x^2 + 0.25y^4 - 0.5y^2
```

上的行为。初始点为 `x0 = (0.1, 0.1)^T`。该函数的驻点为 `(0, 0)`、`(0, 1)` 和 `(0, -1)`，其中 `(0, 0)` 是鞍点，`(0, 1)` 与 `(0, -1)` 是局部极小点。

## 已实现内容

- 经典 Newton 方法
- 修改阻尼 Newton 方法，使用 Hessian 修正 `B_k = H_k + tau_k I`
- Armijo 回溯线搜索
- 两种方法在同一张 `f2` 等高线图上的迭代路径
- 两种方法到局部极小点 `(0, 1)^T` 的距离曲线
- 最终点、最终函数值、最终梯度范数的数值摘要

## 运行方式

在项目根目录运行：

```bash
python3 experiment2/run_experiment2.py
```

生成的图像和表格会写入 `experiment2/outputs/`。如果当前 Python 环境安装了 Matplotlib，脚本会生成 PNG；否则会使用内置 SVG 后备绘图生成 SVG。

## 输出文件

- `experiment2_paths.png` 或 `experiment2_paths.svg`：经典 Newton 与修改阻尼 Newton 的迭代路径等高线图
- `experiment2_distance_to_minimizer.png` 或 `experiment2_distance_to_minimizer.svg`：两种方法到 `(0, 1)^T` 的距离曲线
- `iteration_history.csv`：两种方法的逐步迭代数据
- `summary.txt`：最终点、函数值、梯度范数和简短观察

## 实验现象

从初始点 `(0.1, 0.1)^T` 出发，经典 Newton 方法会很快把 `x` 推到 0，并且由于 `y = 0` 附近 Hessian 的 `yy` 方向为负，Newton 步会把 `y` 推向鞍点 `(0, 0)`。这说明经典 Newton 方法只是在求解一阶驻点条件，并不自动区分极小点和鞍点。

修改阻尼 Newton 方法会通过给 Hessian 加上 `tau_k I` 使线性系统矩阵正定，从而得到下降方向；再配合 Armijo 回溯线搜索控制步长，可以避免沿不可靠方向直接全步跳转。本实验中它会收敛到正侧的局部极小点 `(0, 1)`。
