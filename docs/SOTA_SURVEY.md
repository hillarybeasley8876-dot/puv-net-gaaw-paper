# 点云上采样 SOTA 调研与实验协议锁定

> 本文件是本项目**实验设计的唯一依据**。所有 baseline 数字、协议参数、数据集划分都以此为准。
> 证据等级标注：`[官方仓库]` = 从官方 GitHub README/代码确认；`[论文]` = 从论文正文确认；`[待验证]` = 尚未取得一手证据。
>
> 最后更新：2026-08-10

---

## 0. 命题定位

**创新点（用户指定）**：PU-Transformer 与 PU-GAN 结合，完成点云整体上采样优化。

这是**标准点云上采样任务**（point cloud upsampling），不是「二维表面推理三维内部结构」。
赛道确认后，以下旧设计全部下线：

| 已下线 | 原因 |
|---|---|
| `VolumetricHead` / interior 分支 | 标准上采样任务无内部结构 GT，无法评测 |
| interior 专用指标 | 同上 |
| 合成空腔数据集（hollow_sphere / pipe_block / porous_block） | 标准赛道用 PU1K / PU-GAN 公开数据集 |
| PSNR / SSIM 作主指标 | 投影成二维会压掉三维结构；无法与 SOTA 同表对比 |

---

## 1. 方法谱系（时间线）

| 年份 | 方法 | 会议 | 框架 | 关键机制 | 官方代码 |
|---|---|---|---|---|---|
| 2018 | PU-Net | CVPR | TF 1.x | 多尺度特征 + 特征扩张 | `yulequan/PU-Net` |
| 2019 | MPU (3PU) | CVPR | TF 1.x | 渐进式 patch 上采样 | `yifita/3PU` |
| 2019 | PU-GAN | ICCV | TF 1.x | 对抗训练 + up-down-up 单元 + 均匀性损失 | `liruihui/PU-GAN` |
| 2021 | PU-GCN | CVPR | TF 1.13.1 | Inception DenseGCN + NodeShuffle；**提出 PU1K 数据集** | `guochengqian/PU-GCN` |
| 2022 | **PU-Transformer** | ACCV | — | 位置融合块 + point-wise/channel-wise 多头自注意力；**首个上采样 Transformer** | 无官方代码（见 §5 风险） |
| 2023 | Grad-PU | CVPR | **PyTorch** | 先插值再用学到的距离函数做梯度下降细化；**任意倍率** | `yunhe20/Grad-PU` |
| 2024 | **RepKPU** | CVPR | **PyTorch** | 核点表示 + 形变（KPConv 思路）；当前 SOTA 位置 | `EasyRy/RepKPU` |

**关键结论**：2023 年起该领域**主流已从 TF 迁移到 PyTorch**（Grad-PU / RepKPU 都是 PyTorch），且两者互为基线、共用同一套评测代码。这为我们「全部自己跑」提供了可行路径。

---

## 2. 数据集（已锁定：PU1K 为主 + PU-GAN 为辅）

### 2.1 PU1K `[官方仓库: PU-GCN]`

- 来源：PU-GCN (CVPR'21) 提出
- 规模：1147 个 3D 模型（训练 + 测试）
- 训练集 h5：`pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5`
  - 命名可解读：输入 poisson 256 点 → GT poisson 1024 点，2500 个点云，每个 50 个 patch，**并入了 PU-GAN 数据**
- 测试集目录结构：`data/PU1K/test/input_2048/{input_2048, gt_8192}` → **4× 上采样**
  - 另有 `input_1024` 等其他分辨率
- 下载：
  - 主数据集（Google Drive）：`https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view`
  - 数据集文件夹：`https://drive.google.com/drive/folders/1k1AR_oklkupP8Ssw6gOrIve0CmXJaSH3`
  - 原始 mesh（可选）：`https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view`
  - 预训练模型（PU-GCN on PU1K random + 其他方法）：`https://drive.google.com/file/d/1vusBIw7sd69gnyaeoWMiGaPHfkyHM5Qb/view`

### 2.2 PU-GAN 数据集 `[官方仓库: Grad-PU / RepKPU]`

- 训练集 h5：`PUGAN_poisson_256_poisson_1024.h5`
  - 下载：`https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ`
- **测试集只提供 mesh 文件**，点云需自己用**泊松盘采样（Poisson disk sampling）**生成
  - 测试 mesh 下载：`https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC`
  - 生成脚本（Grad-PU 提供）：`prepare_pugan.py --input_pts_num 2048 --gt_pts_num 8192`（4×）
- RepKPU 提供了**预处理好的多分辨率 GT 测试集**（省掉自己采样的一致性风险）：
  - `https://drive.google.com/drive/folders/14Rd1jaRvGQHJAWM7q_FgJiL9U8_M30qf`

### 2.3 ⚠️ 协议陷阱（必须记死）

**PU1K 训练有 random input 和 uniform(FPS) input 两种协议，二者不能混用。**

PU-GCN README 原文警告：
> "If you favor uniform inputs, you have to retrain all models. Otherwise, the results might be really bad."

- PU-GCN 在 PU1K 上默认用 **random inputs**
- 加 `--fps` 才是 uniform inputs
- PU-GCN 在 **PU-GAN 数据集**上的预训练模型用的是 **uniform inputs**

**我们的决定**：PU1K 主实验统一用 **random input**（与 PU-GCN/Grad-PU/RepKPU 默认一致），并在论文实验章节**显式声明输入协议**。这一条是能否与文献数字对齐的前提。

---

## 3. 评测协议（已锁定）

### 3.1 主指标（4 项）

| 指标 | 全称 | 方向 | 说明 |
|---|---|---|---|
| **CD** | Chamfer Distance | ↓ | 双向最近邻平均距离 |
| **HD** | Hausdorff Distance | ↓ | 最坏情况距离，对离群点敏感 |
| **P2F** | Point-to-Surface | ↓ | 点到原始 mesh 表面的距离，**需要原始 mesh** |
| **Uniformity / NUC** | Normalized Uniformity Coefficient | ↓ | 多个 disk 半径下的均匀性 |

### 3.2 ⚠️ 评测工具链陷阱

**官方 P2F 依赖 CGAL C++ 库**，在 Windows 上是硬坑。链条如下 `[官方仓库]`：

1. Grad-PU / RepKPU 自己只算 **CD**
2. **HD 和 P2F 必须回到 PU-GCN 仓库**用其 `evaluate.py`（TF 1.13.1 环境）
3. PU-GCN 的评测需要编译 CGAL（`evaluation_code/compile.sh`），在 Ubuntu 16.04 上测试过
4. Grad-PU README 明确写：`the evaluate.py script isn't compatible with our environment` —— **官方自己都要切两个虚拟环境**

RepKPU README 原文：
> "You can use our code to get CD value. To calculate HD and P2F value, please refer to [PU-GCN]."

**我们的决定**：
- **自己用 Python 实现全部 4 项指标**（numpy 精确版），不啃 CGAL
- P2F 用 `trimesh` 的最近点查询（`trimesh.proximity.closest_point`）替代 CGAL
- **必须做交叉验证**：用 PU-GCN 预训练模型的输出，跑我们的指标 vs 文献报告值，误差在可接受范围内才认为我们的实现可信
- 这一步是「全部自己跑」路线的**信誉锚点**，不能省

### 3.3 上采样倍率

| 倍率 | 输入点数 | GT 点数 | 数据集 |
|---|---|---|---|
| **4×**（主实验） | 2048 | 8192 | PU1K + PU-GAN |
| 16× | 2048 | 32768 | PU-GAN |
| 5× / 19×（任意倍率） | 2048 | 10240 / — | PU-GAN |

- **4× 是必做主表**，所有方法都报
- 16× 用于展示泛化能力
- 任意倍率是 Grad-PU / RepKPU 的卖点；我们的方法（固定倍率 shuffle 类）若不支持，需在 Limitations 明确写出

### 3.4 鲁棒性实验

Grad-PU 提供 `--noise_level` 参数（如 0.01）生成带噪输入 `[官方仓库]`。
我们应做 **noise 0.005 / 0.01 / 0.02** 三档，这是审稿人必问项。

---

## 4. Baseline 名单（用户决定：全部自己跑）

| 方法 | 原框架 | 我们的策略 | 风险 |
|---|---|---|---|
| PU-Net | TF 1.x | **PyTorch 复现** | 中：结构简单，复现可靠 |
| MPU (3PU) | TF 1.x | PyTorch 复现（可选，非必需） | 中 |
| PU-GAN | TF 1.x | **PyTorch 复现** | 中高：对抗训练超参敏感 |
| PU-GCN | TF 1.13.1 | PyTorch 复现（可选） | 中 |
| **PU-Transformer** | 无官方代码 | **PyTorch 从论文复现** | **高：见 §5** |
| Grad-PU | PyTorch | **直接跑官方代码 + 官方预训练权重** | 低 |
| RepKPU | PyTorch | **直接跑官方代码 + 官方预训练权重** | 低 |

### 4.1 「全部自己跑」的诚实边界

用户选择了「全部自己跑」，但必须在论文里区分三种情况，否则是学术不端：

1. **官方代码 + 官方权重跑出来的**（Grad-PU / RepKPU）→ 标注「官方实现」
2. **我们 PyTorch 复现并自己训练的**（PU-Net / PU-GAN / PU-Transformer）→ **必须标注「本文复现」**，并报告复现值与原论文值的差距
3. **无法复现只能引用的** → 标注文献来源

**红线**：复现值与原论文有差距是正常的、可接受的、必须如实报告的。
**绝不允许**把复现不到的数字直接抄原论文然后声称是自己跑的。

---

## 5. ⚠️ 最大风险：PU-Transformer 无官方代码

- 论文：arXiv 2111.12242，ACCV 2022，作者 Shi Qiu / Saeed Anwar / Nick Barnes（ANU）`[论文]`
- **arXiv 页面无 code 链接**，`[待验证]` 是否存在第三方可信复现
- 摘要确认的两个核心机制 `[论文]`：
  1. **多头自注意力的新变体** —— 同时增强特征图的 point-wise 和 channel-wise 关系
  2. **Positional Fusion Block（位置融合块）** —— 捕获局部上下文，提供散乱点的位置相关信息

**这直接影响创新点的可辩护性**：用户的创新点是「PU-Transformer + PU-GAN 结合」，
如果 PU-Transformer 本身要我们从论文复现，那么：

- **好处**：我们对主干有完全控制权，融合方案可以做得更深（不只是拼接两个黑盒）
- **代价**：复现保真度会被审稿人质疑；必须拉 TeX 源码逐层核对结构参数

**下一步动作**：拉 arXiv TeX 源码（`https://arxiv.org/src/2111.12242`）取得**逐层结构与超参**，
这是复现保真度的唯一可靠依据，比任何二手解读都准。

---

## 6. 创新点的定位风险（必须正视）

用户创新点 = PU-Transformer（2022 ACCV）+ PU-GAN（2019 ICCV）结合。

**问题**：两者分别是 2022 和 2019 的工作，而 2023 的 Grad-PU、2024 的 RepKPU 已经在同一 benchmark 上更强。
如果只是「A + B 拼起来」，主表打不过 RepKPU，论文的说服力会崩。

**可辩护的方向**（需在实验中验证，不能空说）：

1. **不是简单拼接，而是解决具体矛盾** —— Transformer 主干的全局建模 vs GAN 判别器的局部真实性，找出二者的实际冲突点（如梯度尺度失配、判别器过强导致 Transformer 塌缩）并给出机制性解法。这才是「结合」的学术价值。
2. **报告公平的对比位置** —— 如果打不过 RepKPU，就诚实报告，并论证本方法在**其他维度**的优势（参数量 / 推理速度 / 噪声鲁棒性 / 训练稳定性）。
3. **消融必须能证明「1+1 > 2」** —— 必做三组：纯 Transformer 主干（无 GAN）、纯 PU-GAN、二者结合。如果结合后没有提升，这个创新点在数据上就不成立，必须如实说。

**这一条会在实验有数据后重新评估。现在不预设结论。**

---

## 7. 待办（按依赖顺序）

- [ ] 拉 arXiv TeX 源码取 PU-Transformer 逐层结构与超参
- [ ] 确认 PU1K / PU-GAN 数据集在本机网络能否下载（Google Drive，可能需用户手动）
- [ ] 实现 4 项指标（numpy 精确版）+ torch 可微版，交叉验证
- [ ] 用 PU-GCN 预训练模型输出校准我们的指标实现（信誉锚点）
- [ ] 实现数据管线（h5 读取 / patch / 归一化 / 泊松盘采样）
- [ ] 实现 PU-Transformer 主干 + PU-GAN 判别器 + 融合方案
- [ ] baseline 复现
- [ ] 本地小规模跑通 → 实测显存/时长 → 上云正式训练

---

## 8. 一手证据来源清单

| 来源 | URL | 取得内容 |
|---|---|---|
| PU-GCN README | `raw.githubusercontent.com/guochengqian/PU-GCN/master/README.md` | PU1K 下载链接、random/uniform 协议陷阱、baseline 训练命令、CGAL 评测链条 |
| Grad-PU README | `raw.githubusercontent.com/yunhe20/Grad-PU/main/README.md` | 目录结构、泊松采样生成脚本、多倍率参数、noise 参数、跨环境评测坑 |
| RepKPU README | `raw.githubusercontent.com/EasyRy/RepKPU/main/README.md` | CVPR'24 SOTA 位置、预处理测试集、任意倍率、CD 自算/HD+P2F 外借 |
| PU-Transformer arXiv | `arxiv.org/abs/2111.12242` | ACCV 2022、作者、两个核心机制、**无官方代码** |
