# Derivation Package

## Target

核验第5章式（5.5）—式（5.12）的推导主线：在同一生成器参数集合上测量重建梯度与对抗梯度，以目标范数比构造GAAW权重；同时明确该权重能控制什么、不能控制什么。

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

推导在“共同参数集合、权重从计算图分离、梯度范数显著大于稳定项”的条件下连贯。式（5.7）的比例关系和式（5.12）的尺度抵消均为近似性质，不应写成无条件定理。

## Invariant Object

贯穿推导的量是生成器参数空间中，加权对抗梯度相对重建梯度的范数比：

$$
R_t=\frac{\|w_t\mathbf g_{\mathrm{adv}}\|_2}
{\|\mathbf g_{\mathrm{cd}}\|_2+\varepsilon}.
$$

GAAW的直接目标是使 $R_t$ 接近预设的 $r_{\mathrm{target}}$，而不是保证两个损失值的比例、优化器输出位移的比例或最终评价指标的改善。

## Assumptions

- 两条梯度在同一参数集合 $\Theta\subseteq\theta$ 上计算。
- $\Theta$ 非空；未参与某条计算图的参数梯度被显式识别并排除。
- $r_{\mathrm{target}}>0$，稳定项 $\varepsilon>0$。
- 动态权重在总损失反向传播前执行 `detach` 或 `.item()`，本步反向中把 $w_t$ 视为常数。
- 式（5.7）的近似解释要求 $g_{\mathrm{adv}}\gg\varepsilon$，且范数定义中的稳定项不占主导。
- 式（5.12）讨论正常数 $c>0$ 对生成器对抗损失的纯常数重标度；它不覆盖判别器目标、裁剪、回退路径或优化器状态的变化。

## Notation

- $G_\theta$：点云上采样生成器。
- $D_\phi$：点云判别器。
- $\mathcal L_{\mathrm{CD}}$：双向Chamfer重建损失。
- $\mathcal L_{\mathrm{adv}}$：生成器对抗损失。
- $\Theta$：用于比较两条梯度的共同生成器参数集合。
- $\mathbf g_{\mathrm{cd}}=\nabla_\Theta\mathcal L_{\mathrm{CD}}$。
- $\mathbf g_{\mathrm{adv}}=\nabla_\Theta\mathcal L_{\mathrm{adv}}$。
- $g_{\mathrm{cd}}$、$g_{\mathrm{adv}}$：上述向量的稳定化L2范数。
- $w_t$：第 $t$ 个训练批次的动态对抗权重。
- $a(e)$：由epoch决定的外生线性引入因子。
- $r_{\mathrm{target}}$：目标梯度范数比。
- $\alpha_t$：两条梯度之间的夹角。

## Derivation Strategy

先从生成器总损失的梯度分解出发，把“损失权重”转写为“参数空间中的梯度尺度”；再以目标范数比求解动态权重。随后加入外生引入因子，并展开合成梯度范数以说明范数控制不等于方向控制。最后检验对抗损失常数缩放时的有限抵消性质。

## Derivation Map

1. 式（5.1）—式（5.3）定义对抗训练目标，并给出固定权重基线。
2. 式（5.5）—式（5.6）在共同参数集合上定义两条梯度及其范数。
3. 式（5.7）规定期望的加权对抗梯度范数。
4. 式（5.8）由式（5.7）代数求解得到可计算权重；稳定项使比例仅近似成立。
5. 式（5.9）把训练日程与批次自适应量分开。
6. 式（5.10）构成实际生成器目标。
7. 式（5.11）展开合成梯度范数，暴露夹角项这一未受控变量。
8. 式（5.12）说明正常数重标度在限定条件下近似抵消。

## Main Derivation

### Step 1：固定权重只决定代数系数

固定权重生成器目标为

$$
\mathcal L_G^{\mathrm{fixed}}
=\mathcal L_{\mathrm{CD}}+w_f\mathcal L_{\mathrm{adv}}.
$$

对共同参数集合 $\Theta$ 求梯度，得到精确恒等式

$$
\nabla_\Theta\mathcal L_G^{\mathrm{fixed}}
=\mathbf g_{\mathrm{cd}}+w_f\mathbf g_{\mathrm{adv}}.
$$

因此，固定 $w_f$ 并不意味着第二项的实际梯度贡献固定，因为两条未加权梯度会随参数与批次变化。

### Step 2：在同一参数集合上定义可比尺度

定义

$$
\mathbf g_{\mathrm{cd}}=\nabla_\Theta\mathcal L_{\mathrm{CD}},
\qquad
\mathbf g_{\mathrm{adv}}=\nabla_\Theta\mathcal L_{\mathrm{adv}}.
$$

实现采用稳定化范数

$$
g_k=\left(\sum_{\vartheta\in\Theta}
\|\nabla_\vartheta\mathcal L_k\|_2^2+\varepsilon\right)^{1/2},
\qquad k\in\{\mathrm{cd},\mathrm{adv}\}.
$$

这是定义，不是经验结论。若两条范数使用不同参数集合，后续比值会混入参数作用域差异，推导失去可比对象。

### Step 3：从目标范数比求解动态权重

希望加权对抗梯度的范数约为重建梯度的 $r_{\mathrm{target}}$ 倍：

$$
\|w_t\mathbf g_{\mathrm{adv}}\|_2
\approx r_{\mathrm{target}}\|\mathbf g_{\mathrm{cd}}\|_2.
$$

在 $w_t\ge0$ 时，利用范数的正齐次性，形式上得到

$$
w_t\,g_{\mathrm{adv}}
\approx r_{\mathrm{target}}g_{\mathrm{cd}},
$$

从而构造

$$
w_t=r_{\mathrm{target}}
\frac{g_{\mathrm{cd}}}{g_{\mathrm{adv}}+\varepsilon}.
$$

这里前两步是目标设定与代数求解；把稳定项加入分母是数值近似。代回后得到

$$
\frac{w_tg_{\mathrm{adv}}}{g_{\mathrm{cd}}}
=r_{\mathrm{target}}
\frac{g_{\mathrm{adv}}}{g_{\mathrm{adv}}+\varepsilon},
$$

所以只有当 $g_{\mathrm{adv}}\gg\varepsilon$ 时，该比值才接近 $r_{\mathrm{target}}$。

### Step 4：把外生日程与内生权重分开

实际目标写为

$$
\mathcal L_G^{\mathrm{GAAW}}
=\mathcal L_{\mathrm{CD}}+a(e)w_t\mathcal L_{\mathrm{adv}}.
$$

$a(e)$ 由epoch预先规定，$w_t$ 由当前批次梯度计算。忽略稳定项时，实际目标范数比约为 $a(e)r_{\mathrm{target}}$。因此机制图和日志应分别记录 $a(e)$、$w_t$ 及乘积，不能把三者混为一条“自适应权重”曲线。

### Step 5：范数平衡不能消除方向冲突

把 $w_t$ 视为本步常数，合成梯度为

$$
\mathbf g_t
=\mathbf g_{\mathrm{cd}}+a(e)w_t\mathbf g_{\mathrm{adv}}.
$$

平方范数的精确展开为

$$
\|\mathbf g_t\|_2^2
=g_{\mathrm{cd}}^2+a(e)^2w_t^2g_{\mathrm{adv}}^2
+2a(e)w_tg_{\mathrm{cd}}g_{\mathrm{adv}}\cos\alpha_t,
$$

若忽略稳定化范数与真实向量范数之间的微小差异。GAAW只调节第二项尺度，未控制 $\cos\alpha_t$；因而不能从范数比例推出共同下降方向。

### Step 6：常数重标度的有限抵消

令 $\mathcal L'_{\mathrm{adv}}=c\mathcal L_{\mathrm{adv}}$，其中 $c>0$。则

$$
\nabla_\Theta\mathcal L'_{\mathrm{adv}}
=c\nabla_\Theta\mathcal L_{\mathrm{adv}},
\qquad
g'_{\mathrm{adv}}\approx cg_{\mathrm{adv}}.
$$

忽略稳定项时，重新计算的权重满足 $w'_t\approx w_t/c$，于是

$$
\left(\frac{w_t}{c}\right)
\nabla_\Theta\!\left(c\mathcal L_{\mathrm{adv}}\right)
=w_t\nabla_\Theta\mathcal L_{\mathrm{adv}}.
$$

等式本身是代数恒等式；把实际 $w'_t$ 替换为 $w_t/c$ 则依赖前述近似条件。论文正文应保留“忽略稳定项时”的限定。

## Remarks and Interpretation

- $r_{\mathrm{target}}=0.1$ 表示优化器输入前、指定参数集合上的梯度范数目标，不表示损失值占比为10%。
- Adam会逐坐标重新缩放合成梯度，所以该比例不是参数位移的精确分解。
- 全参数范数与最后一层范数对应不同的观测对象，不能仅凭公式断言前者更优。
- `.item()`或`detach`把方法限定为“先测量、再以常数加权”；若保留权重计算图，算法会包含二阶项。
- 第6章的11条交集结果只能评价当前训练结果的探索性方向，不能反向证明上述机制假设普遍成立。

## Boundaries and Non-Claims

- 不声称梯度范数比自适应判别权重为本文首创；VQGAN已存在直接先例。
- 不声称范数达到目标比后两项梯度方向一致。
- 不声称GAAW对任意损失变换不变；这里只讨论正常数缩放的限定情形。
- 不声称动态权重有界；当前实现没有上限裁剪。
- 不以公式性质替代与无对抗、固定权重、最后一层作用域和显式均匀性方法的实验比较。

## Open Risks

- 原始逐批次 $g_{\mathrm{cd}}$、$g_{\mathrm{adv}}$ 与回退标志尚未形成完整可审计轨迹，H1仍待重跑。
- 全参数聚合可能被参数量较大的层主导，需要分层范数对照。
- 未裁剪权重在 $g_{\mathrm{adv}}$ 接近零时可能出现极端值。
- 当前正式独立测试集与多训练seed评价未完成，方法的泛化与稳定性仍不能确认。
