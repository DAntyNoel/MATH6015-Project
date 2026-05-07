# MATH6015 Project

这是 MATH6015 Optimization Methods 编程作业的代码框架，用于实现和比较 Newton-type optimization methods。

## 当前结构

- `src/newton_methods/core.py`：通用目标函数接口、迭代记录和算法返回结果。
- `src/newton_methods/line_search.py`：Armijo 回溯线搜索。
- `src/newton_methods/algorithms.py`：固定步长梯度下降、经典 Newton 方法、修改阻尼 Newton 方法。
- `src/newton_methods/problems.py`：目标函数封装和作业中的测试函数。
- `src/newton_methods/experiment_utils.py`：实验脚本共享的输出目录、绘图缓存和结果格式化工具。
- `examples/smoke_check.py`：基础框架的快速检查脚本。
- `exp1/`：Experiment 1 的完整运行脚本、说明和输出文件。
- `exp2/`：Experiment 2 的完整运行脚本、说明和输出文件。
- `exp3/`：Experiment 3 的完整运行脚本、说明和输出文件。

## 运行环境

推荐使用 conda 的 `main` 环境运行本项目：

```bash
conda activate main
```

## 运行基础检查

```bash
python examples/smoke_check.py
```

## 运行 Experiment 1

```bash
python exp1/run.py
```

输出图像和表格会生成到 `exp1/outputs/`。

## 运行 Experiment 2

```bash
python exp2/run.py
```

输出图像和表格会生成到 `exp2/outputs/`。

## 运行 Experiment 3

```bash
python exp3/run.py
```

输出图像和表格会生成到 `exp3/outputs/`。

## 实现原则

主要优化算法均在本项目中直接实现，不调用 `scipy.optimize.minimize`、`fminunc` 等高层黑盒优化器。项目使用 NumPy 完成向量运算、线性方程求解和 Hessian 特征值检查，使用 Matplotlib 生成实验图像。
