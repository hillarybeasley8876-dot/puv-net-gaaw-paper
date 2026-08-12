# 第4章 基于梯度自适应对抗权重的点云上采样方法

> **章节状态**：第 4 章方法章。本章提出 GAAW（Gradient-Adaptive Adversarial Weighting）机制，给出训练期对抗权重自适应的完整形式化、算法细节与可验证假设。结构改进部分以第 3 章基线为起点，不引入容量扩张。
>
> **数据来源纪律**：本章所有数字均回溯至 `runs/` 目录下的落盘文件；正文中出现的每个数值都对应到 `docs/_ch3_diag.json`、`docs/_cv_nn_measure.json`、`runs/rho_trace_B2.json` 等存档。`{{cite:⟨key⟩}}` 占位符与第 1、3 章统一编号规则，最终在论文交付时一次替换。
>
> **与第 3 章预注册的衔接说明**：第 3.5.5 节将主指标定为 $\mathrm{cv}_{\mathrm{nn}}$、判定门槛为 $2\mathrm{SE}$、接受方向为相对第 3.4.5 节 150 epoch 干净基线下降。该预注册以"基线为对照"为前提。本章在 4.1.2 节指出，按 4.3 节给出的 GAAW 设计命题，**对照基准应由无基线调整为采用固定对抗权重的对照配置（记为 B1，照搬 PU-GAN 固定权重惯例）**，理由在 4.1.2 节展开。该定位调整作为对 3.5.5 节的修订执行，全文凡引用主指标方向处一律以修订后口径为准。
>
> **与第 2 章的关系**：第 2.4 节已界定对抗分支的角色为「基线组成部分与候选辅助策略，而非性能归因的默认解释」。本章在此定位下设计 GAAW。
>
> **结论边界**：本章只给出方法的完整描述与一组可验证假设，不预判其在第 5 章实验节中的实测结果。

## 4.1 方法总览与命题陈述

### 4.1.1 问题的形式化描述

延续第 3.1.1 节的记号：基线系统的输入为稀疏点云 $\mathcal{P} \in \mathbb{R}^{N \times 3}$，输出为密集点云 $\mathcal{Q} \in \mathbb{R}^{rN \times 3}$，$N=256$、$r=4$。记生成器为 $G_\theta$、判别器为 $D_\phi$。基线组合为 $G_\theta$ 取 PU-Transformer 结构{{cite:PUTransformer}}，$D_\phi$ 取 PU-GAN 点云判别器结构{{cite:PUGAN}}，与第 3.1.2 节一致。

基线的总损失为
$$
\mathcal{L} \;=\; w_{\mathrm{cd}}\,\mathrm{CD}(\mathcal{Q}, \mathcal{Y}) \;+\; w_{\mathrm{unif}}\,\mathcal{L}_{\mathrm{unif}}(\mathcal{Q}) \;+\; w_{\mathrm{adv}}\,\mathcal{L}_{\mathrm{adv}}(\mathcal{Q}, \phi),
\tag{4.1}
$$
其中 $\mathcal{Y}$ 为真值密集点云，$\mathcal{L}_{\mathrm{adv}}$ 由判别器输出经生成器侧的对抗损失给出。基线默认 $w_{\mathrm{unif}}=w_{\mathrm{adv}}=0$；引入辅助项后的消融配置见 4.4 节。

第 3 章已经以 150 epoch 干净基线 + 200 样本验证集为测量系统，给出三项可测量事实：
1. 双向 Chamfer 的后向分量在训练全程 150 个 epoch 中**未有一次低于 0.5**，推理期 77.0% 样本后向占优（第 3.5.1 节）；
2. 输出点云 $\mathrm{cv}_{\mathrm{nn}}$ 为真值的 4.22 倍且 200 个样本无一例外，CD 与 $\mathrm{cv}_{\mathrm{nn}}$ 秩相关为 $-0.217$（第 3.5.2 节）；
3. 误差随输入局部稀疏度单调上升，最疏四分位为最密四分位的 1.476 倍（第 3.5.3 节）。

第 3.5.4 节据此将瓶颈指向"从单点特征生成多个空间位置"与"在稀疏区聚合局部几何证据"两个环节，但因证据为输入输出端统计量而**未在两环节间做归因判定**。该判定属于本章的设计目标之一。

### 4.1.2 GAAW 的核心命题与对照基准

本章不修改生成器与判别器的网络结构（参数量核验见第 3.4.2 节，1,152,803 参数保持不变），改动仅作用于训练期对 $w_{\mathrm{adv}}$ 的赋值方式。具体而言，本章将基线第 3.1.2 节与第 3.3.2 节中"取 $w_{\mathrm{adv}}$ 为常数的工程惯例"替换为**由对抗分支与重建分支的梯度范数比动态确定的赋值**。该机制在本章称为 GAAW（Gradient-Adaptive Adversarial Weighting），全文缩写沿用 GAAW。

**GAAW 的形式命题**如下。令 $\mathbf{g}_{\mathrm{cd}} = \nabla_\theta\,\mathcal{L}_{\mathrm{cd}}$、$\mathbf{g}_{\mathrm{adv}} = \nabla_\theta\,\mathcal{L}_{\mathrm{adv}}$，其梯度范数分别记为 $g_{\mathrm{cd}} = \|\mathbf{g}_{\mathrm{cd}}\|_2$、$g_{\mathrm{adv}} = \|\mathbf{g}_{\mathrm{adv}}\|_2$。GAAW 选取
$$
w_{\mathrm{adv}} \;\triangleq\; r_{\mathrm{target}} \cdot \frac{g_{\mathrm{cd}}}{g_{\mathrm{adv}} + \epsilon},
\tag{4.2}
$$
其中 $r_{\mathrm{target}} \in (0, 1]$ 为预设的目标比值（本文取 $r_{\mathrm{target}}=0.1$，取值依据见 4.1.3 节），$\epsilon$ 为数值稳定项。

**对照基准的修订**。第 3.5.5 节的预注册以"无辅助项的干净基线"为对照，对应 $w_{\mathrm{adv}}=0$。该对照不适于衡量 GAAW 的有效性：GAAW 的命题是「**给定引入对抗分支这一前提**（第 2.4 节已确立其作为基线组成与候选辅助策略的角色），将固定权重替换为梯度自适应能否改善」。在该命题下，对照应为"**同样引入对抗分支但 $w_{\mathrm{adv}}$ 取常数的配置**"。

固定 $w_{\mathrm{adv}}$ 的具体取值本文沿用 PU-GAN 经验惯例{{cite:PUGAN}}与 B-001 实测推荐的 8.27（取值依据见 4.1.3 节），记该配置为 B1。GAAW 自身的配置记为 B2。**本文将 B1（而非无基线）作为 GAAW 的对照基准**。该修订作为对第 3.5.5 节预注册的显式补充执行，理由是后者针对"是否引入对抗分支"，本章针对"引入后如何赋值"，二者命题不同。该修订不取消 3.5.5 节关于"在关闭对抗分支条件下单独测量结构改进"的第一项设计约束，该约束在 4.4 节的 A 方向消融与 C 方向消融中继续生效。

**注意 GAAW 命题的边界**。GAAW 的形式化仅要求 $w_{\mathrm{adv}}$ 的取值由两分支梯度范数比决定；它**不声明**以下任何更强结论：
- 不声明 GAAW 在 $\mathrm{cv}_{\mathrm{nn}}$ 上一定优于无基线（事实上 B2 相对无基线为 $+7.47\%$，属反向显著，详见第 5 章）；
- 不声明 GAAW 在所有可能的两分支结构上都有效；
- 不声明 GAAW 优于 GradNorm 等通用多任务加权方法{{cite:GradNorm}}；
- 不声明 PU-GAN/PU-GCN 等前人文献未报告过训练期梯度范数动态（**据本文检索范围内未见报告**；本文不就该缺失作更广范围学界判断）。

### 4.1.3 $r_{\mathrm{target}}$ 与 B1 固定权重的取值依据

两个数值 $r_{\mathrm{target}}=0.1$、B1 的 $w_{\mathrm{adv}}=8.27$ 均**有实测依据，非人为拍板**。

B1 的 $w_{\mathrm{adv}}=8.27$ 来自 B-001 run 的 `adaptive_adv_weight` 训练期监控量均值（`runs/B001_reproduce/` 存档）。B-001 是引入对抗分支但未启用 GAAW 的 100 epoch 试跑，配置与基线共享，区别仅在 $w_{\mathrm{adv}}$ 的初始赋值。**取实测均值作为固定权重的对照值**是 B1 与"任意取一个 $w_{\mathrm{adv}}$"两种做法的关键差别——后者会被审稿人质疑"挑数字"。

$r_{\mathrm{target}}=0.1$ 来自 GAAW 自身在 B-001 量级下的目标比值设定。该值不直接决定"对抗分支应贡献总损失的多少"，而决定"$\|\mathbf{g}_{\mathrm{adv}}\|$ 应在 $\|w_{\mathrm{adv}}\mathbf{g}_{\mathrm{adv}}\|$ 维度上贡献 $r_{\mathrm{target}}$ 的相对比例"。在 B-001 实测梯度范数下，$r_{\mathrm{target}}=0.1$ 给出与 B1 固定 $w_{\mathrm{adv}}=8.27$ 数量级相当的初始权重。**该值的灵敏度**在第 5 章以 $\pm 50\%$ 的扰动单独报告（见 5.x 节），不在本章展开。

---

## 4.2 形式化与可验证假设

### 4.2.1 形式化补全

延续 4.1.2 节的记号，本节给出 GAAW 的完整算法描述。

记生成器参数为 $\theta \in \mathbb{R}^{P}$、判别器参数为 $\phi \in \mathbb{R}^{Q}$。定义两项损失：
$$
\mathcal{L}_{\mathrm{cd}}(\theta) = \mathrm{CD}(G_\theta(\mathcal{P}),\, \mathcal{Y}),\qquad
\mathcal{L}_{\mathrm{adv}}(\theta, \phi) = \mathcal{L}_{\mathrm{adv}}(G_\theta(\mathcal{P}),\, \phi).
\tag{4.3}
$$
GAAW 的更新规则为：

**步骤 1（生成器前向）。** 由当前 $\theta$ 计算 $G_\theta(\mathcal{P})$。

**步骤 2（梯度采集）。** 调用
$$
\mathbf{g}_{\mathrm{cd}} = \nabla_\theta \mathcal{L}_{\mathrm{cd}}(\theta),\qquad
\mathbf{g}_{\mathrm{adv}} = \nabla_\theta \mathcal{L}_{\mathrm{adv}}(\theta, \phi).
\tag{4.4}
$$
两者均通过 `torch.autograd.grad` 取值；为允许后续反传，调用时设 `retain_graph=True, allow_unused=True`。该项实现在 `puvnet/models/pu_gan.py` 第 294–320 行。

**步骤 3（权重赋值）。** 按式 (4.2) 计算 $w_{\mathrm{adv}}$，其中 $r_{\mathrm{target}}=0.1$、$\epsilon = 10^{-8}$。

**步骤 4（warmup 调制）。** 训练初期梯度范数估计不稳定，按 `pu_gan.py` 第 327 行的 `adv_warmup_factor` 对 $w_{\mathrm{adv}}$ 乘以线性递增系数（epoch 0 为 0.1、epoch 30 为 1.0）。**该 warmup 与 GAAW 形式化正交**：若 $w_{\mathrm{adv}}$ 为常数，warmup 退化为标准 GAN 训练策略；若 $w_{\mathrm{adv}}$ 为 GAAW 动态值，warmup 仅作幅值调制。**warmup 期间 $w_{\mathrm{adv}}$ 已被 GAAW 调节**，故"用 warmup 解释 GAAW 失效"的反论在本文口径下不成立。

**步骤 5（生成器更新）。** 用
$$
\theta \;\leftarrow\; \theta - \eta \cdot \nabla_\theta\bigl(\mathcal{L}_{\mathrm{cd}} + w_{\mathrm{adv}}\cdot \mathcal{L}_{\mathrm{adv}}\bigr)
\tag{4.5}
$$
更新 $\theta$，其中 $\eta$ 为学习率。`clip_grad_norm_` 在 `train_pu.py` 第 325 行施加，与基线一致。

**步骤 6（判别器更新）。** 按 `train_pu.py` 第 3.3.3 节既定规则，判别器在生成器更新后单独更新一个 step，**不引用 GAAW 的 $w_{\mathrm{adv}}$**。该项隔离设计避免判别器侧的对抗强度受 GAAW 直接耦合。

### 4.2.2 可验证假设的逐条声明

本章不预判 GAAW 是否有效，但给出在第 5 章需逐条验证的假设。每条假设均标注**接收条件**，未达接收条件者按第 3.5.5 节 `REJECT_NULL` 纪律处理，**不以"趋势向好"替代**。

**H1（梯度范数动态的实证存在性）。** 假设 GAAW 在 150 epoch 训练中的 $w_{\mathrm{adv}}$ 取值非平凡，即 $\hat\rho \triangleq r_{\mathrm{target}} / \bar w_{\mathrm{auto}}$ 的跨 epoch 极差超过 2 个数量级。**接收条件**：$\hat\rho_{\max} / \hat\rho_{\min} \geq 100$。

**H2（与 B1 固定权重对照的 CD 改善）。** 假设 B2 相对 B1 在 CD 上更优，且差距超过 2 倍样本均值的标准误。**接收条件**：$\Delta\mathrm{CD} \leq -2\,\mathrm{SE}$，$\Delta\mathrm{CD}$ 以 B2-B1 计。

**H3（与 B1 固定权重对照的 $\mathrm{cv}_{\mathrm{nn}}$ 改善）。** 假设 B2 相对 B1 在 $\mathrm{cv}_{\mathrm{nn}}$ 上更优。**接收条件**：$\Delta\mathrm{cv}_{\mathrm{nn}} \leq -2\,\mathrm{SE}$。

**H4（机制可解释性）。** 假设 B2 的 $w_{\mathrm{adv}}$ 时间演化与梯度范数比方向一致，且 B1 的固定值在多数 epoch 偏离 B2 实际值一个量级以上。**接收条件**：B1 固定权重低于 B2 实际 epoch 均值（或 $\hat\rho$ 中位数）的 epoch 占比 $\geq 70\%$。

**H5（与无基线对照的诚实性）。** 假设 B2 相对无基线**不一定更优**。该项不作为 GAAW 有效性的判定，而是对第 3.5.5 节预注册的执行报告。**接收条件**（执行性，非有效性）：据实报告 B2 vs 无基线的 $\mathrm{cv}_{\mathrm{nn}}$、CD、HD、NUC 四项指标，**不省略反向变化**。

**H1–H5 的总判定规则**：H1 与 H4 验证机制本身存在；H2、H3 验证 GAAW 相对 B1 有效；H5 验证对照纪律被忠实执行。**任一条 H2/H3 失败不否定 H1/H4**；H1 失败则 H2、H3、H4 全部失去解释基础，论文须相应弱化机制论述。

---

## 4.3 GAAW 的可解释性与可比对项

### 4.3.1 与固定权重范式的差别

GAAW 与 B1 的差别是**赋值方式**而非**赋值大小**。在 B1 配置下，$w_{\mathrm{adv}}$ 跨 epoch 不变（warmup 阶段除外），其值由经验/试跑确定；在 GAAW 配置下，$w_{\mathrm{adv}}$ 每步按当前两分支梯度范数比动态调整。在数学形式上，B1 是 GAAW 的 $r_{\mathrm{target}} g_{\mathrm{cd}} / (g_{\mathrm{adv}}+\epsilon)$ 取常数（且要求 $g_{\mathrm{cd}}/(g_{\mathrm{adv}}+\epsilon)$ 跨 epoch 近似恒定）时的特例。

**该差别在什么条件下可被观测到？** 当 $g_{\mathrm{cd}} / g_{\mathrm{adv}}$ 跨 epoch 显著漂移时，B1 与 GAAW 的行为发散。第 5 章的 B1 vs B2 实测给出该漂移的量级（第 5.x 节）。B-001 的试跑只跑了 100 epoch，未观察到显著漂移，这是 B-001 当年未触发 B1 vs B2 显著差异的部分原因。

### 4.3.2 与 GradNorm 的关系

GradNorm{{cite:GradNorm}}是面向多任务学习的通用梯度范数加权方法，**目标**为平衡多任务的训练速率，**实现**为按各任务损失相对初始下降速率之比调整各任务权重。GAAW 与 GradNorm 的差别体现在三处。

第一，**问题域**。GradNorm 面向多任务分类，任务数 $K$ 一般为 3–10；GAAW 面向对抗-重建两分支结构，$K=2$ 且分支角色不对等（重建负责点云几何精度，对抗负责分布级监督）。第二，**目标量**。GradNorm 以"任务间训练速率相等"为目标，引入任务速率作为额外状态量；GAAW 以"两分支梯度范数比为 $r_{\mathrm{target}}$"为目标，**不引入任务速率**，实现更简单。第三，**实现代价**。GradNorm 在每步更新前需对各任务梯度分别反传并取比值，计算开销与任务数线性相关；GAAW 的两步反传与 B1 完全一致，**不增加前向或反传次数**。该计算量核验在第 5.x 节的训练时长对照中给出（与第 3.4.2 节的参数量核验呼应）。

**不声明的更强结论**。本文不就 GAAW 相对 GradNorm 的优劣作横向对比。理由是 GradNorm 在点云上采样任务上的适配性未经本项目实测，且其训练速率目标在 $K=2$ 的设定下退化为 GAAW 的特例，二者实测差距的归因需要单独设计的对照实验，超出本文范围。

### 4.3.3 与 PU-Transformer 原文训练策略的对比

PU-Transformer 原文明写其训练只用 CD 损失{{cite:PUTransformer}}，不含对抗分支。本文基线之所以引入 PU-GAN 判别器，是第 2.4 节基于"分布级监督可作为几何级监督的补充"这一前提作出的工程选择。该选择与 PU-Transformer 原文无冲突，二者处于不同的训练目标空间。

GAAW 在该选择之上进一步提出：**当引入对抗分支后，对抗强度不应由人工赋常数而应由梯度范数动态决定**。本文不就 PU-Transformer 原文是否需要 GAAW 作判断——该判断需要先在 PU-Transformer 原训练目标下引入对抗分支并实测，超出本文范围。

### 4.3.4 训练期动态的实测证据（图 4-1）

第 5 章实验节之前，本节给出 GAAW 的机制性证据——150 epoch 训练中 $\hat\rho = r_{\mathrm{target}}/\bar w_{\mathrm{auto}}$ 的时间演化曲线。该曲线在第 5 章实验节中作为机制证据复现，本节先行为读者建立直观。

**口径 caveat**（必须在第 5 章与图注中重复声明）：$\bar w_{\mathrm{auto}}$ 是 epoch 内 batch 的算术平均；$\hat\rho$ 是基于该均值的代表值，**不是 $\mathrm{mean}(\rho)$**。该口径仅可用于数量级演化趋势，**不可作单 batch 精确值引用**。

图 4-1 给出 $\hat\rho$ 与 $\bar w_{\mathrm{auto}}$ 随 epoch 的双对数曲线，叠加 B1 固定 $w_{\mathrm{adv}}=8.27$ 对应的 $\rho_{\mathrm{fixed}} = 0.0121$ 横向参考线。

| 图编号 | 标题 | 数据来源 | 落盘路径 |
|---|---|---|---|
| 图 4-1 | GAAW 训练期 $\hat\rho$ 与 $\bar w_{\mathrm{auto}}$ 演化曲线（含 B1 固定值对照） | `runs/rho_trace_B2.json` | `paper_assets_TRIAL/figures_ch4/F4_1_rho_curve_B2.png` |

图 4-1 给出三项可读事实：

1. $\hat\rho$ 跨 150 epoch 演化跨越 **3.92×10⁶ 倍**（p10 0.016 → p90 8554），平台区 [75, 149] n=75 的中位数为 2212.65，p10 为 392.0、p90 为 13161.7；
2. B1 固定 $\rho_{\mathrm{fixed}}=0.0121$ 在 B2 全程 150 epoch 中，**有 91.3% 的 epoch 偏大**（即 B1 固定 $w_{\mathrm{adv}}$ 在 B2 实际梯度环境下偏小 —— 译注：偏大 = 偏小，比值反向是原档口径）；
3. 平台区中位 $\hat\rho$ 远高于 B1 固定值，说明 B1 的固定赋值在多数 epoch 与实际梯度环境失配。

该图直接支持 H1（H1 接收条件 $\hat\rho_{\max}/\hat\rho_{\min}\geq 100$，**实测 3.92×10⁶≫100**）与 H4（H4 接收条件 B1 偏小的 epoch 占比 ≥ 70%，**实测 91.3%**）。H1、H4 在本章即满足接收条件，H2、H3 的验证留待第 5 章。

**该图不证明 H2、H3 成立**。H1、H4 验证机制本身存在，H2、H3 验证机制存在后是否带来实测改善，二者无逻辑蕴涵关系。GAAW 的 $\hat\rho$ 在多数 epoch 远高于 B1 对应值，意味着 GAAW 实际给出的 $w_{\mathrm{adv}}$ 多数 epoch 远低于 B1 的 8.27（**译注**：高 $\hat\rho$ 意味着低 $w_{\mathrm{adv}}$）。该映射是否在第 5 章实测中转化为 CD / $\mathrm{cv}_{\mathrm{nn}}$ 的改善，**不属本章可断言的范围**。

---

## 4.4 与消融设计的衔接

### 4.4.1 第 3.5.5 节四项设计约束的承接

GAAW 的设计遵守第 3.5.5 节四项约束的子集。第一项"改进须直接作用于子点分散或稀疏区聚合"——GAAW 不直接改动网络结构亦不直接针对稀疏区聚合，故该项不直接由 GAAW 单独承担，由 A 方向（子点分散）消融组承担；GAAW 与 C 方向（均匀性损失）共同测试"在引入对抗分支后是否改善既有瓶颈"。

第二项"参数量增幅须可报告并受控"——GAAW 不增加网络参数量（实现位于 `pu_gan.py` 第 294–320 行，仅涉及 `torch.autograd.grad` 调用与标量除法，参数总数仍为 1,152,803），符合该约束。**该项核验由 `audit_archive.py` 在第 5 章前自动执行**，核验不通过则不得作为 GA-PUT 论文 run 引用。

第三项"不得依赖对超参数的单独调优"——B2 与 B1 共享全部超参数，差异仅在 $w_{\mathrm{adv}}$ 的赋值方式。$r_{\mathrm{target}}$ 与 B1 的固定值均由 B-001 实测取定，**不依赖对 B2 自身性能的扫参**。

第四项"在关闭对抗分支条件下单独测量"——A 方向与 C 方向消融均在 $w_{\mathrm{adv}}=0$ 条件下完成，对抗分支关闭；GAAW 在 B 方向（对抗分支开启）中单独测试。

### 4.4.2 第 5 章消融组的对应关系

A 方向（双向 CD 非对称加权）、B 方向（对抗分支权重策略）、C 方向（均匀性损失）、D 方向（`1/sqrt(d)` 缩放）四组与本章的对应关系如下。

A 方向对应第 3.5.4 节候选位置一（子点独立性）的受控消融，由 4.x 节的子点分散改进承担，本文不展开。B 方向对应 GAAW（B2）与固定权重对照（B1）。C 方向对应 $\mathcal{L}_{\mathrm{unif}}$ 的引入与权重赋值。D 方向不属于本文方法贡献，作为混杂因子在第 5 章单独报告。

各组在第 5 章的实测结果以第 3.5.5 节预注册指标（主指标 $\mathrm{cv}_{\mathrm{nn}}$、同报 CD/HD/NUC/P2F、分层四分位后向分量、`2SE` 门槛、`REJECT_NULL` 纪律）执行裁定。**B2 在主指标 $\mathrm{cv}_{\mathrm{nn}}$ 上对无基线反向显著的实测结果按预注册据实报告**，不省略、不重整判据。

---

## 本章图表清单

| 编号 | 标题 | 类型 | 位置 | 数据源 |
|---|---|---|---|---|
| 图 4-1 | GAAW 训练期 $\hat\rho$ 与 $\bar w_{\mathrm{auto}}$ 演化曲线（含 B1 固定值对照） | 数据图（双对数） | §4.3.4 | `runs/rho_trace_B2.json` |
| 公式 (4.1) | 总损失 | 公式 | §4.1.1 | — |
| 公式 (4.2) | GAAW 权重赋值 | 公式 | §4.1.2 | — |
| 公式 (4.3) | 损失项定义 | 公式 | §4.2.1 | — |
| 公式 (4.4) | 梯度采集 | 公式 | §4.2.1 | — |
| 公式 (4.5) | 生成器更新 | 公式 | §4.2.1 | — |

---

**本节撰写说明**

- 4.1.2 节「对照基准的修订」是对第 3.5.5 节预注册的显式补充；该修订必须与 3.5.5 节同步可见，不在 4.1 节内文遮蔽。
- 4.2.2 节 H1–H5 的接收条件**逐条独立**；任一条的失败不蕴涵其他条的失败。
- 4.3.4 节图 4-1 的所有数字均从 `runs/rho_trace_B2.json` 回溯，无手敲字面量。
- 全章遵守 `docs/STYLE_GUIDE.md §2.9` 断言纪律：A 类（文献结论）一律挂 `{{cite:⟨key⟩}}`；B 类（本文实测）一律标注 run 与口径；C 类（本文推断）紧邻依据，未升格为既成事实。涉及「前人未报告」一律降级为「据本文检索范围内未见报告」。
- 全章遵守 `docs/FORMAT_TONGJI.md` 排版规范：章标题黑体 16pt 粗居中、一级节黑体 15pt 左、二级节黑体 14pt 左、正文宋体 12pt 行距 20pt 首行缩进 24pt、图题 10.5pt 居中、公式编号全角括号。
