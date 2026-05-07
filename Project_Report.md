# Newton 型优化方法项目报告

## 1. 算法简介

本项目实现了 5 类方法：

$$
\text{Gradient Descent},\quad \text{Classical Newton},\quad
\text{Modified Damped Newton},\quad \text{CG},\quad \text{Newton-CG}.
$$

### 固定步长梯度下降

$$
x_{k+1}=x_k-\alpha \nabla f(x_k).
$$

实验 1 中

$$
\alpha=0.2.
$$

优点：单步便宜。缺点：只用一阶信息，通常线性收敛。

### 经典 Newton 方法

Newton 方向满足

$$
\nabla^2 f(x_k)d_k=-\nabla f(x_k),
$$

迭代为

$$
x_{k+1}=x_k+d_k.
$$

若 $x_k$ 充分接近非退化极小点，且 Hessian 正定，则通常有

$$
\|x_{k+1}-x^*\|=O(\|x_k-x^*\|^2).
$$

### 修正阻尼 Newton 方法

先修正 Hessian：

$$
\tau_k=\max\{0,\delta-\lambda_{\min}(\nabla^2 f(x_k))\},
$$

$$
B_k=\nabla^2 f(x_k)+\tau_k I,\qquad \delta=10^{-3}.
$$

方向：

$$
B_k d_k=-\nabla f(x_k).
$$

若 $B_k\succ 0$ 且 $g_k=\nabla f(x_k)\ne 0$，则

$$
d_k=-B_k^{-1}g_k,
$$

$$
g_k^T d_k=-g_k^T B_k^{-1}g_k<0.
$$

所以 $d_k$ 是下降方向。

Armijo 回溯线搜索选择 $t_k$：

$$
f(x_k+t_kd_k)\le f(x_k)+c_1t_k\nabla f(x_k)^Td_k,
$$

其中代码使用

$$
c_1=10^{-4},\qquad \rho=0.5.
$$

### Newton-CG

Newton-CG 用 CG 近似求解

$$
\nabla^2 f(x_k)d_k=-\nabla f(x_k).
$$

内层停止准则：

$$
\|\nabla^2 f(x_k)d_k+\nabla f(x_k)\|
\le \eta_k\|\nabla f(x_k)\|.
$$

实验 3 比较：

$$
\eta_k=0.8,\qquad
\eta_k=0.1,\qquad
\eta_k=\min\{0.5,\|\nabla f(x_k)\|\}.
$$

## 2. 梯度与 Hessian 公式

### 实验 1

目标函数：

$$
f_1(x,y)=x^4+y^4+x^2+y^2+0.5xy.
$$

梯度：

$$
\nabla f_1(x,y)=
\begin{bmatrix}
4x^3+2x+0.5y\\
4y^3+2y+0.5x
\end{bmatrix}.
$$

Hessian：

$$
\nabla^2 f_1(x,y)=
\begin{bmatrix}
12x^2+2 & 0.5\\
0.5 & 12y^2+2
\end{bmatrix}.
$$

初始点与极小点：

$$
x_0=(1,1)^T,\qquad x^*=(0,0)^T.
$$

### 实验 2

目标函数：

$$
f_2(x,y)=0.5x^2+0.25y^4-0.5y^2.
$$

梯度：

$$
\nabla f_2(x,y)=
\begin{bmatrix}
x\\
y^3-y
\end{bmatrix}.
$$

Hessian：

$$
\nabla^2 f_2(x,y)=
\begin{bmatrix}
1 & 0\\
0 & 3y^2-1
\end{bmatrix}.
$$

驻点：

$$
(0,0),\qquad (0,1),\qquad (0,-1).
$$

分类：

$$
\nabla^2 f_2(0,0)=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix},
$$

所以 $(0,0)$ 是鞍点。

$$
\nabla^2 f_2(0,\pm1)=
\begin{bmatrix}
1&0\\
0&2
\end{bmatrix}\succ0,
$$

所以 $(0,\pm1)$ 是局部极小点。

### 实验 3

目标函数：

$$
f_3(x)=\frac12x^TAx+\frac c4\sum_{i=1}^n x_i^4.
$$

参数：

$$
n=200,\qquad c=0.5,\qquad
A=\operatorname{diag}(a_1,\ldots,a_n).
$$

其中

$$
a_i\in[10^{-3},10^1]
$$

按对数间隔生成。

梯度：

$$
\nabla f_3(x)=Ax+cx^{\circ3}.
$$

分量形式：

$$
[\nabla f_3(x)]_i=a_ix_i+cx_i^3.
$$

Hessian：

$$
\nabla^2 f_3(x)=
\operatorname{diag}(a_i+3cx_i^2)_{i=1}^n.
$$

Hessian-vector product：

$$
\nabla^2 f_3(x)v=
\big((a_i+3cx_i^2)v_i\big)_{i=1}^n.
$$

初始点：

$$
x_0=(1,2,1,2,\ldots)^T.
$$

极小点：

$$
x^*=0.
$$

## 3. 实验 1：Newton 的快速局部收敛

比较：

$$
\text{Gradient Descent}\quad \text{vs.}\quad \text{Classical Newton}.
$$

![实验 1 迭代路径](exp1/outputs/experiment1_paths.png)

![实验 1 误差曲线](exp1/outputs/experiment1_errors.png)

结果：

| 方法 | 迭代数 | $f(x_k)$ | $\|\nabla f(x_k)\|$ | 最终点 |
| --- | ---: | ---: | ---: | --- |
| Gradient Descent | 35 | $5.2076\times10^{-22}$ | $5.1027\times10^{-11}$ | $(-1.4433\times10^{-11},-1.4433\times10^{-11})$ |
| Newton | 5 | $1.6881\times10^{-24}$ | $2.9053\times10^{-12}$ | $(8.2173\times10^{-13},8.2173\times10^{-13})$ |

Newton 误差与观测阶：

| $k$ | $\|x_k-x^*\|_2$ | $q_k$ |
| ---: | ---: | ---: |
| 0 | $1.4142$ | - |
| 1 | $7.8026\times10^{-1}$ | - |
| 2 | $3.0882\times10^{-1}$ | $1.5585$ |
| 3 | $3.8345\times10^{-2}$ | $2.2507$ |
| 4 | $8.9889\times10^{-5}$ | $2.9029$ |
| 5 | $1.1621\times10^{-12}$ | $2.9994$ |

问题回答：

1. Newton 为什么更快？

$$
d_k=-(\nabla^2 f(x_k))^{-1}\nabla f(x_k).
$$

它使用曲率信息。靠近极小点时，二次模型很准，所以局部收敛很快。

2. 梯度下降为什么 zigzag？

狭长等高线表示 Hessian 条件数大：

$$
\kappa(\nabla^2 f)\gg1.
$$

负梯度方向常横穿谷底，固定步长容易左右震荡。

3. 是否支持 Newton 的快速局部收敛？

支持。实验中

$$
35\ \text{iterations}\quad \text{vs.}\quad 5\ \text{iterations}.
$$

观测阶最后接近 $3$，原因是本实验有对称结构：

$$
x_0=y_0,\qquad f_1(x,y)=f_1(y,x).
$$

迭代保持在

$$
x=y
$$

上，因此出现特殊的近三阶现象。一般 Newton 理论结论仍是局部二次收敛。

## 4. 实验 2：鞍点与 Hessian 修正

比较：

$$
\text{Classical Newton}\quad \text{vs.}\quad \text{Modified Damped Newton}.
$$

初始点：

$$
x_0=(0.1,0.1)^T.
$$

![实验 2 迭代路径](exp2/outputs/experiment2_paths.png)

![实验 2 到极小点距离](exp2/outputs/experiment2_distance_to_minimizer.png)

结果：

| 方法 | 迭代数 | $f(x_k)$ | $\|\nabla f(x_k)\|$ | 最终点 |
| --- | ---: | ---: | ---: | --- |
| Classical Newton | 3 | $\approx0$ | $9.9262\times10^{-24}$ | $(0,-9.9262\times10^{-24})$ |
| Modified Damped Newton | 5 | $-0.25$ | $4.6107\times10^{-11}$ | $(0,1.0000000000230533)$ |

问题回答：

1. Classical Newton 为什么会被鞍点吸引？

Newton 解的是

$$
\nabla f(x)=0.
$$

但

$$
\nabla f(x)=0
$$

只说明 $x$ 是驻点，不保证是极小点。

本实验中

$$
(0,0)
$$

是鞍点，但 classical Newton 收敛到它。

2. Hessian 不定时，Newton 方向为什么不一定下降？

Newton 方向：

$$
d=-H^{-1}g.
$$

下降条件：

$$
g^Td<0.
$$

若 $H\succ0$：

$$
g^Td=-g^TH^{-1}g<0.
$$

若 $H$ 不定，则

$$
g^TH^{-1}g
$$

可正可负，所以 $g^Td$ 不一定小于 $0$。

3. 为什么修正方向是下降方向？

若

$$
B_k\succ0,\qquad B_kd_k=-g_k,
$$

则

$$
d_k=-B_k^{-1}g_k.
$$

所以

$$
g_k^Td_k=-g_k^TB_k^{-1}g_k<0.
$$

4. Armijo line search 的作用是什么？

Hessian 修正确保

$$
d_k\ \text{is descent}.
$$

Armijo 确保

$$
f(x_{k+1})<f(x_k)
$$

具有充分下降。二者结合避免收敛到鞍点，并收敛到

$$
(0,1).
$$

## 5. 实验 3：Inexact Newton / Newton-CG

目标：比较不同 forcing term。

![实验 3 梯度范数](exp3/outputs/gradient_norms.png)

![实验 3 累计 CG 次数](exp3/outputs/cumulative_cg_iterations.png)

结果：

| $\eta_k$ | 外层迭代数 | 总 CG 次数 | $f(x_k)$ | $\|\nabla f(x_k)\|$ |
| --- | ---: | ---: | ---: | ---: |
| $0.8$ | 79 | 961 | $1.8633\times10^{-19}$ | $9.8512\times10^{-11}$ |
| $0.1$ | 16 | 861 | $2.5440\times10^{-20}$ | $4.0393\times10^{-11}$ |
| $\min(0.5,\|\nabla f(x_k)\|)$ | 15 | 1149 | $3.7214\times10^{-24}$ | $8.2076\times10^{-13}$ |

问题回答：

1. 为什么 $\eta_k=0.8$ 单步便宜但外层迭代多？

停止条件宽松：

$$
\|Hd+g\|\le0.8\|g\|.
$$

CG 早停，单步便宜；但方向粗糙，外层下降慢。

2. 为什么 $\eta_k=0.1$ 更准确但可能更贵？

停止条件更严格：

$$
\|Hd+g\|\le0.1\|g\|.
$$

方向更接近精确 Newton 方向，外层迭代少；但每步 CG 可能更多。

3. 为什么自适应规则能平衡成本？

$$
\eta_k=\min(0.5,\|\nabla f(x_k)\|).
$$

远离解时：

$$
\|\nabla f(x_k)\|\ \text{large}\Rightarrow \eta_k\approx0.5.
$$

接近解时：

$$
\|\nabla f(x_k)\|\to0\Rightarrow \eta_k\to0.
$$

因此早期少算，后期精算。

4. 本实验说明了 inexact Newton 的什么思想？

不必精确解 Newton 方程：

$$
Hd=-g.
$$

只需控制残差：

$$
\|Hd+g\|\le\eta_k\|g\|.
$$

核心权衡：

$$
\text{inner CG cost}\quad \leftrightarrow\quad \text{outer Newton convergence}.
$$

## 6. Newton 型方法的优点与局限

优点：

$$
\text{uses curvature information}
$$

$$
\Rightarrow\quad \text{fast local convergence}.
$$

在极小点附近：

$$
\|x_{k+1}-x^*\|=O(\|x_k-x^*\|^2).
$$

Hessian 可自动缩放方向：

$$
d_k=-H_k^{-1}g_k.
$$

因此比梯度下降更少 zigzag。

局限：

1. 可能收敛到非极小驻点：

$$
\nabla f(x)=0
\nRightarrow x\ \text{is minimizer}.
$$

2. Hessian 不定时方向可能非下降：

$$
H_k\not\succ0
\Rightarrow
g_k^Td_k\not<0.
$$

3. 精确 Newton 单步代价高：

$$
H_kd_k=-g_k
$$

需要解线性系统。

4. 大规模问题需要近似：

$$
\text{Newton-CG},\quad \text{Hessian-vector product},\quad \text{forcing term}.
$$

总结：

$$
\text{Newton is fast but needs safeguards.}
$$

实际可靠实现通常需要

$$
\text{Hessian modification}+\text{line search}+\text{inexact solve}.
$$
