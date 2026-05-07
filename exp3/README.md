# 实验 3：Inexact Newton 与 Newton-CG

本目录实现作业 PDF 中的 Experiment 3，用于比较不同 forcing term `eta_k` 对 Newton-CG 方法的外层收敛速度和内层 CG 成本的影响。

目标函数为

```text
f3(x) = 0.5 x^T A x + c/4 * sum_i x_i^4
```

其中 `c = 0.5`，`A = diag(a_1, ..., a_n)`，对角元素通过 `np.logspace(-3, 1, n)` 生成。本实验默认使用 `n = 200`，初始点为 `(1, 2, 1, 2, ...)^T`，最优点为 `x* = 0`。

## 已实现内容

- 共轭梯度法 CG
- Newton-CG / inexact Newton 方法
- Armijo 回溯线搜索
- 三种 forcing term：
  - `eta_k = 0.8`
  - `eta_k = 0.1`
  - `eta_k = min(0.5, ||grad f(x_k)||_2)`
- 每种方法记录外层 Newton 迭代编号、梯度范数和累计 CG 次数
- 梯度范数关于外层迭代次数的半对数图
- 累计 CG 次数关于外层迭代次数的曲线图

## 运行方式

在项目根目录运行：

```bash
python exp3/run.py
```

生成的图像和表格会写入 `exp3/outputs/`。

## 输出文件

- `gradient_norms.png`：三种 `eta_k` 下的梯度范数半对数曲线
- `cumulative_cg_iterations.png`：累计 CG 次数曲线
- `iteration_history.csv`：逐步迭代数据
- `summary.txt`：每种 `eta_k` 的外层迭代次数、总 CG 次数、最终函数值和最终梯度范数

## 实验现象

`eta_k = 0.8` 的内层线性系统求解较粗糙，因此每一步 Newton 便宜，但通常需要更多外层迭代。`eta_k = 0.1` 的内层求解更准确，外层收敛更稳定，但每一步可能消耗更多 CG 迭代。

自适应规则 `eta_k = min(0.5, ||grad f(x_k)||_2)` 在远离最优点时允许较粗糙的 Newton 方向，靠近最优点时逐渐提高内层精度，因此能体现 inexact Newton 方法平衡单步成本和收敛速度的思想。
