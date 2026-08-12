# PU-Transformer 复现规格（一手证据：arXiv TeX 源码）

> **证据来源**：`https://arxiv.org/src/2111.12242`（15,291,074 B，已解包至 `refs/pu_transformer/`）
> 逐层结构与超参全部来自论文 TeX 原文，非二手解读、非记忆推测。
>
> 关键文件：`sections/methodology.tex`（算法）/ `sections/implementation.tex`（超参）/ `sections/experiments.tex`（结果表）
>
> **PU-Transformer 无官方开源代码**，本项目为 PyTorch 从零复现。原文用 TensorFlow + 单张 GeForce 2080 Ti。

---

## 1. 整体架构

输入稀疏点云 `P ∈ R^{N×3}` → 输出密集点云 `S ∈ R^{rN×3}`，`r` = 上采样倍率。

三段式：**Head → Body（L 个 Transformer Encoder）→ Tail**

```
# PU-Transformer Head
F_0 = MLP(P)

# PU-Transformer Body        l = 1 ... L
for each Transformer Encoder:
    G_l  = PosFus(P, F_{l-1})                      # 位置融合块
    G_l' = SC-MSA(Norm(G_l)) + G_l                 # 残差 1
    F_l  = MLP(Norm(G_l')) + G_l'                  # 残差 2

# PU-Transformer Tail
S = MLP(Shuffle(F_L))
```

**关键实现要点**：
- `Norm` = LayerNorm（论文引 Ba et al. 2016）
- **pre-norm 结构**（Norm 在子层之前，残差加原始输入），不是 post-norm
- `Shuffle` = PixelShuffle 周期性重排（Shi et al. 2016），**不引入额外参数**
- **knn 只算一次**，所有 PosFus 块共用同一组邻居索引 —— 这是原文强调的效率优势之一，实现时必须复用，不要每层重算

---

## 2. Positional Fusion（PosFus）块

### 2.1 公式（methodology.tex Eq.1-5）

给定坐标 `P ∈ R^{N×3}` 和上一层特征 `F ∈ R^{N×C}`：

**第 1 步 — knn 分组**（基于 **3D 欧氏距离**，非特征空间距离）
```
P_j ∈ R^{N×k×3}     # k 个邻居的坐标
F_j ∈ R^{N×k×C}     # k 个邻居的特征
```

**第 2 步 — 几何上下文**
```
ΔP    = P_j - P                        ∈ R^{N×k×3}      (Eq.1)
G_geo = concat[dup_k(P) ; ΔP]          ∈ R^{N×k×6}      (Eq.2)
```

**第 3 步 — 特征上下文**（与几何完全对称的操作）
```
ΔF     = F_j - F                       ∈ R^{N×k×C}      (Eq.3)
G_feat = concat[dup_k(F) ; ΔF]         ∈ R^{N×k×2C}     (Eq.4)
```

**第 4 步 — 融合**
```
G = max_k( concat[ M_Φ(G_geo) ; M_Θ(G_feat) ] )   ∈ R^{N×C'}   (Eq.5)
```
- `M_Φ` / `M_Θ` 是两个**独立的** MLP
- `max_k` 是在邻域维度（k）上的 max-pooling
- 输出通道 `C'` 由本层指定（见 §4）
- 两个 MLP 的输出通道各占 `C'/2`，concat 后为 `C'`

### 2.2 与 DGCNN 的关键区别（不能实现错）

> 原文：DGCNN 的局部图需要在每个 encoder 里基于**嵌入空间的动态关系**更新；而 `G_geo` 和 `G_feat` **都按固定的 3D 几何关系**构建和编码。

**实现红线**：
- ❌ 不要用动态图（每层在特征空间重新 knn）
- ✅ 用**固定的** 3D 欧氏距离 knn，索引全程复用
- 原文给的两个理由：(i) 昂贵的 knn 只需算一次；(ii) 几何和特征上下文用同样的方式表示，有利于**公平融合**

---

## 3. Shifted Channel Multi-head Self-Attention（SC-MSA）块

### 3.1 动机

常规 MSA 只算 **point-wise** 依赖，各 head 之间**互相独立**，缺少 **channel 相关信息**的整合。
而基于 shuffle 的上采样恰恰依赖 channel 维度信息 → 所以要让 head 之间在 channel 上有重叠。

### 3.2 算法（methodology.tex Alg.2）

输入 `I ∈ R^{N×C'}`，输出 `O ∈ R^{N×C'}`。参数：split 宽度 `w`、shift 间隔 `d`（`d < w`）、head 数 `M`。

```
Q = Linear(I)      # 1×1 conv, ∈ R^{N×C'}
K = Linear(I)
V = Linear(I)

for m in 1..M:
    Q_m = Q[:, (m-1)*d : (m-1)*d + w]      # 沿 channel 开窗，窗口滑动
    K_m = K[:, (m-1)*d : (m-1)*d + w]
    V_m = V[:, (m-1)*d : (m-1)*d + w]
    A_m = softmax(Q_m @ K_m^T)             # ∈ R^{N×N}
    O_m = A_m @ V_m                        # ∈ R^{N×w}

O = Linear( concat[O_1, ..., O_M] )        # ∈ R^{N×C'}
```

### 3.3 核心差异：滑窗 vs 独立切分

- **常规 MSA**：把 `C'` 切成 `C'/w` 个**互不重叠**的 split
- **SC-MSA**：窗口沿 channel **滑动**，任意两个连续 split 有 `(w-d)` 个**重叠 channel**
- 因此 head 数更多：`M > C'/w`

**注意**：`Linear` 论文明确说「implement as a 1×1 convolution」。
**注意**：`softmax(Q_m K_m^T)` 原文**没有写 `/√d` 缩放因子** —— 按原文实现（不加缩放），若训练不稳定再作为消融项记录，不要默认偷偷加。

---

## 4. 超参（implementation.tex，逐字）

### 4.1 Body 配置

| 项 | 值 | 出处 |
|---|---|---|
| Transformer Encoder 层数 | **L = 5** | implementation.tex §pu_body |
| 各层输出通道 `C'` | **32 → 64 → 128 → 256 → 256** | 同上 |
| PosFus 邻居数 | **k = 20** | 同上（沿用 DGCNN / PU-GCN） |

> 「in each Transformer Encoder, we only use the Positional Fusion block to encode the corresponding channel dimension (i.e. `C'` in Eq.5), which remains the same in the subsequent operations」
>
> → **通道升维只发生在 PosFus 内部**，SC-MSA 和后续 MLP 都保持 `C'` 不变。

### 4.2 SC-MSA 配置

| 项 | 公式 | ψ=4 时的值 |
|---|---|---|
| reduction ratio | `ψ` | **ψ = 4**（所有 SC-MSA 块统一） |
| split 通道宽度 | `w = C'/ψ` | C'=32→w=8；64→16；128→32；256→64 |
| channel shift 间隔 | `d = w/2` | 4 / 8 / 16 / 32 |
| head 数 | `M = 2ψ - 1` | **M = 7**（所有层都是 7） |

**自检**：`M=7, d=w/2` 时最后一个窗口结束位置 = `(7-1)*(w/2) + w = 3w + w = 4w = C'` ✅ 刚好覆盖满，不越界。

### 4.3 Tail 配置

```
输入：N × 256              （最后一个 encoder 输出）
Shuffle：rN × (256/r)      （周期性重排 channel）
MLP：   rN × 3             （回归 3D 坐标）
```
- r=4 → shuffle 后 `4N × 64`
- r=16 → shuffle 后 `16N × 16`

### 4.4 训练超参（experiments.tex）

| 项 | 值 |
|---|---|
| batch size | **64** |
| epochs | **100** |
| 初始学习率 | **1×10⁻³** |
| lr decay rate | **0.7** |
| 损失函数 | **仅** modified Chamfer Distance（MPU/3PU 版本）|
| 框架 / 硬件 | TensorFlow / 单张 2080 Ti |

> 「we **only** use the modified Chamfer Distance loss to train the PU-Transformer」
>
> ⚠️ **这一点对本项目至关重要**：原版 PU-Transformer **不含任何对抗损失、不含 repulsion/uniform 损失**。
> 用户的创新点（+ PU-GAN）正是要在这里加东西 —— 所以「纯 PU-Transformer」消融基线的 loss 必须严格只有 CD，
> 否则消融证明不了「GAN 带来的增益」。

超参「heavily adopt」自 PU-GCN（PU1K 实验）和 Dis-PU（PU-GAN 数据集实验）。

---

## 5. 模型复杂度（experiments.tex Tab.5，用于复现校验）

| L | 参数量 | 模型大小 | 训练速度/batch | 推理/sample | CD↓ | HD↓ | P2F↓ |
|---|---|---|---|---|---|---|---|
| 3 | 438.3k | 8.5M | 12.2s | 6.9ms | 0.487 | 4.081 | 1.362 |
| 4 | 547.3k | 11.5M | 15.9s | 8.2ms | 0.472 | 4.010 | 1.284 |
| **5**（采用） | **969.9k** | **18.4M** | 23.5s | 9.9ms | **0.451** | **3.843** | **1.277** |
| 6 | 2634.4k | 39.8M | 40.3s | 11.0ms | 0.434 | 3.996 | 1.210 |

**🔑 复现验收第一关**：我们的 PyTorch 实现在 `L=5, ψ=4, k=20, r=4` 下，
**参数量应接近 969.9k**。差太多说明结构理解有误，必须回头查，不能带着错结构往下训。

（注：参数量不会完全一致 —— TF 和 PyTorch 的 BN/LayerNorm、conv bias 习惯略有差异。
量级和主要层的分布对上即可，偏差超过 ±15% 需排查。）

---

## 6. 论文报告的结果（文献值，供对表，**不得据为己有**）

### 6.1 PU1K 4× 上采样（Tab.1，单位 ×10⁻³）

| 方法 | 参数量(×10³) | CD↓ | HD↓ | P2F↓ |
|---|---|---|---|---|
| PU-Net | 812.0 | 1.155 | 15.170 | 4.834 |
| MPU | 76.2 | 0.935 | 13.327 | 3.551 |
| PU-GACNet | 50.7 | 0.665 | 9.053 | 2.429 |
| PU-GCN | 76.0 | 0.585 | 7.577 | 2.499 |
| Dis-PU * | 1047.0 | 0.485 | 6.145 | 1.802 |
| **PU-Transformer** | 969.9 | **0.451** | **3.843** | **1.277** |

`*` = 原文标注的自行复现结果

### 6.2 PU-GAN 数据集（Tab.2，单位 ×10⁻³）

| 方法 | 4× CD↓ | 4× HD↓ | 4× P2F↓ | 16× CD↓ | 16× HD↓ | 16× P2F↓ |
|---|---|---|---|---|---|---|
| PU-Net | 0.844 | 7.061 | 9.431 | 0.699 | 8.594 | 11.619 |
| MPU | 0.632 | 6.998 | 6.199 | 0.348 | 7.187 | 6.822 |
| PU-GAN | 0.483 | 5.323 | 5.053 | 0.269 | 7.127 | 6.306 |
| PU-GCN * | 0.357 | 5.229 | 3.628 | 0.256 | 5.938 | 3.945 |
| Dis-PU | 0.315 | 4.201 | 4.149 | **0.199** | 4.716 | 4.249 |
| **PU-Transformer** | **0.273** | **2.605** | **1.836** | 0.241 | **2.310** | **1.687** |

**注意**：16× 时 PU-Transformer 的 CD 输给 Dis-PU（0.241 vs 0.199）。
原文解释：Dis-PU 用了两个 CD 相关损失项，所以只在 CD 指标上占优。**这是诚实报告负面结果的范例，值得学。**

### 6.3 组件消融（Tab.3，PU1K，单位 ×10⁻³）

| 模型 | PosFus | Attention | Tail | CD↓ | HD↓ | P2F↓ |
|---|---|---|---|---|---|---|
| A₁ | None | SC-MSA | Shuffle | 0.605 | 6.477 | 2.038 |
| A₂ | 仅 G_geo | SC-MSA | Shuffle | 0.558 | 5.713 | 1.751 |
| A₃ | 仅 G_feat | SC-MSA | Shuffle | 0.497 | 4.164 | 1.511 |
| B₁ | 两者 | SA (Non-local) | Shuffle | 0.526 | 4.689 | 1.492 |
| B₂ | 两者 | OSA (PCT) | Shuffle | 0.509 | 4.823 | 1.586 |
| B₃ | 两者 | 常规 MSA | Shuffle | 0.498 | 4.218 | 1.427 |
| C₁ | 两者 | SC-MSA | MLPs (PU-Net) | 1.070 | 8.732 | 2.467 |
| C₂ | 两者 | SC-MSA | DupGrid (MPU) | 0.485 | 3.966 | 1.380 |
| C₃ | 两者 | SC-MSA | NodeShuffle (PU-GCN) | 0.505 | 4.157 | 1.404 |
| **Full** | 两者 | SC-MSA | Shuffle | **0.451** | **3.843** | **1.277** |

**可读出的信息**：
- PosFus 贡献最大（A₁ 0.605 → Full 0.451，降 25%）
- SC-MSA vs 常规 MSA：0.498 → 0.451（降 9.4%）—— 增益真实但不算巨大
- Tail 用简单 Shuffle 反而最好（C₂/C₃ 用更复杂的方法都更差）

### 6.4 噪声鲁棒性（Tab.4，PU1K，噪声 ~ β·N(0,1)）

| 方法 | β=0.5% CD/HD/P2F | β=1% CD/HD/P2F | β=2% CD/HD/P2F |
|---|---|---|---|
| PU-Net | 1.006 / 14.640 / 5.253 | 1.017 / 14.998 / 6.851 | 1.333 / 19.964 / 10.378 |
| MPU | 0.869 / 12.524 / 4.069 | 0.907 / 13.019 / 5.625 | 1.130 / 16.252 / 9.291 |
| PU-GCN | 0.621 / 8.011 / 3.524 | 0.762 / 9.553 / 5.585 | 1.107 / 13.130 / 9.378 |
| Dis-PU | 0.496 / 6.268 / 2.604 | **0.591** / 7.944 / 4.417 | **0.858** / 10.960 / 7.759 |
| **PU-Transformer** | **0.453 / 4.052 / 2.127** | 0.610 / **5.787 / 3.965** | 1.058 / **9.948 / 7.551** |

⚠️ **重要观察**：噪声越大，PU-Transformer 的 CD 优势越差（β=2% 时 1.058 输给 Dis-PU 的 0.858），
但 HD/P2F 始终领先。**说明纯 Transformer + 纯 CD 损失在强噪声下会产生偏移。**

**→ 这正好是「结合 PU-GAN」的理论切入点**：
PU-GAN 的对抗损失 + uniform 损失本身就是为了抑制离群点和非均匀分布。
如果我们的融合方案能在 **β=2% 时把 CD 从 1.058 压下去**，那就是一个有机制解释、有数据支撑的真实贡献，
而不是「把两个模型拼起来」。**这条假设记为 H-GAN-1，必须在实验中验证。**

---

## 7. 评测协议（experiments.tex，patch-based 推理）

测试流程（沿用 MPU/PU-GAN/PU-GCN 通用做法）：

1. 把输入点云切成**多个 seed patch**，覆盖全部 N 个点
2. 用训练好的模型对每个 patch 做 r× 上采样
3. 用**最远点采样（FPS）**把所有上采样后的 patch 合并成 rN 点的密集输出

**4× 实验规格**：输入 2048 点 → 输出/GT 8192 点
（注：原文 experiments.tex 第 19 行写「8,196 points」，第 21 行注释写「8,096」，
均为 **8192 的笔误**。本项目统一用 **8192** = 2048×4。这是原论文的一处小错，我们不继承。）

指标：CD / HD / P2F 三项，**都需要原始 3D mesh**（P2F 必需）。

---

## 8. 数据集规格（experiments.tex）

### PU1K
- **1,020 个训练 mesh + 127 个测试 mesh**
  （⚠️ 与常见「1147」说法一致：1020 + 127 = 1147）
- 多数来自 **ShapeNetCore**，覆盖 **50 个类别**
- 训练数据：**69,000 个样本**，每样本 **256 输入点 → 1,024 GT 点**（4×）
- 由 mesh 的 patch 经**泊松盘采样**生成

### PU-GAN 数据集
- 训练：**24,000 个 patch**，来自 **120 个 mesh**
- 测试：**27 个 mesh**
- 规模比 PU1K 小，用于 4× 和 16× 两种倍率

---

## 9. 复现待确认项（诚实标注）

| 项 | 状态 | 影响 |
|---|---|---|
| Head 的 MLP 输出通道数 | ✅ **已定案：32** —— 实测影响仅 ±1,800 参数（见 §9.2） | 小 |
| Encoder 内 MLP 隐层扩张比 `mlp_ratio` | ✅ **已定案：1** —— 实测 ratio=2/4 使参数量偏离到 1.50/2.13 倍（见 §9.2） | 已解决 |
| SC-MSA 的 `proj` 层输入维 | ⚠️ **已定位为主要差距来源**，见 §9.3 | **高** |
| PosFus 中两个 MLP 的层数/是否带 BN+ReLU | `[待验证]` 只说「encoded via two MLPs」；本实现取单层 Linear+BN+ReLU | 中 |
| lr decay 的 step 间隔 | `[待验证]` 只给 decay rate 0.7 | 中；参考 PU-GCN 配置 |
| modified CD 的具体形式 | `[待验证]` 引 MPU/3PU；需查 MPU 论文 | **高**；这是唯一的损失函数 |

### 9.1 复现验收实测结果

`L=5, dims=(32,64,128,256,256), k=20, psi=4, r=4, mlp_ratio=1, head_dim=32`

| 项 | 本实现 | 论文 | 比例 |
|---|---|---|---|
| 参数量 | **1,152,803** | 969,900 | **1.189** |

其余自检全部通过：4× 前向形状正确、SC-MSA 窗口在 dim=32/64/128/256 下均刚好铺满且 M=7、
四种消融变体（A₁/A₃/B₃/C₁）可正常构建与前向、r=4/8/16 均可用、102/102 参数收到非零梯度、
knn 第一近邻为自身。

### 9.2 mlp_ratio 与 head_dim 扫描（实测，`scripts/param_audit.py`）

| mlp_ratio | 参数量 | 比例 | 与目标差 |
|---|---|---|---|
| **1** | **1,152,803** | **1.189** | **+182,903** |
| 2 | 1,458,691 | 1.504 | +488,791 |
| 4 | 2,070,467 | 2.135 | +1,100,567 |

| head_dim | 参数量 | 比例 |
|---|---|---|
| 16 | 1,152,211 | 1.188 |
| **32** | 1,152,803 | 1.189 |
| 64 | 1,153,987 | 1.190 |

**结论**：`mlp_ratio=1` 明确最优 —— 点云上采样工作不沿用图像 Transformer 的 4× FFN 惯例。
`head_dim` 影响可忽略（±1,800）。两项均已定案，不再是不确定源。

### 9.3 差距归因：SC-MSA 的 proj 层（实测）

参数量分布显示 **encoders.3.attn 与 encoders.4.attn 各占 311,552（合计 54.0%）**，
是最大开销。逐层拆解（psi=4 → w=C/4, M=7）：

| dim | w | heads | QKV | proj（concat 输入 = M·w = 1.75C） | proj（若输入 = C） |
|---|---|---|---|---|---|
| 32 | 8 | 7 | 3,072 | 1,824 | 1,056 |
| 64 | 16 | 7 | 12,288 | 7,232 | 4,160 |
| 128 | 32 | 7 | 49,152 | 28,800 | 16,512 |
| 256 | 64 | 7 | 196,608 | 114,944 | 65,792 |
| 256 | 64 | 7 | 196,608 | 114,944 | 65,792 |
| **合计** | | | | **725,472** | **611,040**（省 114,432）|

若把 proj 输入维改为 `C`：总参数 **1,152,803 → 1,038,371**，与论文差距从 **18.9% 收窄到 7.1%**。

**⚠️ 本项目的决定：不改，保持忠于原文。**

理由：论文 Alg.2 第 12 行明写
`O = Linear(concat[{O_1, O_2, ..., O_M}])`，
concat 后维度就是 `M·w = 1.75C`。**为了凑参数量而违背论文明写的算法，是本末倒置。**

因此：
- **默认实现** = 忠于原文（proj 输入 `M·w`），参数量 1,152,803
- 剩余 7.1%（约 68k）差距的可能来源：PosFus MLP 层数、TF 与 PyTorch 的 bias/BN 习惯差异、
  原文可能在某处做了未记录的通道压缩
- 论文写作时**如实报告**：「本文复现版参数量 1.15M，原文报告 0.97M，
  差异源于原文未公开代码，SC-MSA 输出投影层与 PosFus 内 MLP 的具体配置无法逐层核对」

### 9.4 处理原则

- 忠于论文**明写**的参数（L、dims、k、psi、batch、lr、loss），一律不动
- 原文**未写**的项取最合理默认值，并在论文中显式标注为「复现实现细节」
- **绝不为了凑数字扭曲结构**；参数量差异如实报告，不隐藏
- 所有校准尝试记入 `docs/EXPERIMENT_LOG.md`

---

## 10. 引用

```bibtex
@inproceedings{qiu2022pu,
  title={PU-Transformer: Point Cloud Upsampling Transformer},
  author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2022}
}
```
