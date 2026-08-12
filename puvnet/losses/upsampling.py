"""PUTG-Net 训练损失 —— 新命题（点云上采样）专用。

与 `losses/reconstruction.py` 的关系
-----------------------------------
`reconstruction.py` 属旧命题（「二维表面推理三维内部结构」）遗留，且存在三处不适配：
  1. 通道序为 `(B,3,N)`，与新模型 `pu_transformer.py` / `pu_gan.py` 的 `(B,N,3)` 不一致
  2. 含 `interior_containment_loss`（内部结构项），新命题已无此概念
  3. 其 `chamfer_loss` docstring 写「评测时用非平方版本报告」——**该说法已被证伪**，
     PU-GCN 官方评测用的是平方距离（见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md）
**本模块为新命题唯一损失入口，旧文件保留但不再用于新命题。**

训练 loss 与评测指标的口径分离（重要）
-------------------------------------
两者是**独立决策**，不可互相推导：

| | 训练 loss | 评测指标 |
|---|---|---|
| 实现 | 本模块，torch，可微，求快 | `puvnet/metrics/pointcloud.py`，numpy+KD-tree，求准 |
| CD 口径 | **modified CD**（PU-Transformer 原文） | 官方 squared CD（双向 mean 求和） |
| 归一化 | patch 内，input/gt **共享** gt 的 center/scale | pred/gt **各自独立** |
| 用途 | 产生梯度 | 产生论文数字 |

> 之所以要分离：训练要的是稳定梯度，评测要的是可对表数字。
> 但**必须交叉验证**，否则会出现「训练 loss 降了而真实指标没降」的静默失效。
> 本模块 `self_check` 中与 numpy 参考实现逐值对比。

PU-Transformer 原文的损失配置（一手依据）
----------------------------------------
来源：arXiv 2111.12242 §4.1 与 refs/pu_transformer/（TeX 源码）
- 损失**仅用 modified CD**，无对抗项、无 repulsion 项
- batch 64，100 epochs，lr 1e-3，decay 0.7

→ 故 **B-001 纯复现必须只开 cd 项**（`w_adv=0, w_uniform=0, w_repulsion=0`），
  这是复现保真度的前提；任何额外项都会使"复现"失去意义。

modified CD 是什么
------------------
PU-Net/MPU(3PU)/PU-Transformer 系列使用的 CD 变体，对双向项做点数归一化：

    CD_mod = (1/N) * sum_{x in pred} min_y ||x-y||^2
           + (1/M) * sum_{y in gt}   min_x ||y-x||^2

当 pred 与 gt 点数不同时（上采样场景恒成立），逐方向取均值而非全局均值，
可避免点数多的一方主导梯度。数值上等于本模块 `chamfer_loss(..., squared=True)`。

⚠️ 未在一手 TeX 中找到 modified CD 的逐式定义（原文仅引用 MPU），
   故此处按 PU-Net/MPU 系列的通行实现（双向各自取 mean）执行，
   并在 docs/EVIDENCE_LEDGER.md 标注为「通行实现推断，非原文逐式引用」。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =============================================================================
# 重建项
# =============================================================================

def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor,
                 squared: bool = True) -> torch.Tensor:
    """双向 Chamfer（modified CD），逐方向取均值。

    Parameters
    ----------
    pred : (B, N, 3)
    gt   : (B, M, 3)
    squared : True 用平方距离（梯度更平滑，且与 PU-GCN 评测口径同型）

    Returns
    -------
    标量 loss（已对 batch 取均值）
    """
    if pred.dim() != 3 or gt.dim() != 3 or pred.shape[-1] != 3:
        raise ValueError(f"期望 (B,N,3)，收到 pred={tuple(pred.shape)} "
                         f"gt={tuple(gt.shape)}")
    d = torch.cdist(pred, gt)                    # (B,N,M)
    if squared:
        d = d ** 2
    fwd = d.min(dim=2).values.mean(dim=1)        # (B,) pred->gt
    bwd = d.min(dim=1).values.mean(dim=1)        # (B,) gt->pred
    return (fwd + bwd).mean()


def chamfer_loss_split(pred: torch.Tensor, gt: torch.Tensor,
                       squared: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (fwd, bwd) 双向分项，用于诊断精度项与覆盖项的相对走势。

    E-000 附属诊断已证明：复制型输出的误差主要来自 bwd（覆盖项，占 74%）。
    训练时逐项监控可及早发现「精度在降但覆盖没改善」这类问题。
    """
    d = torch.cdist(pred, gt)
    if squared:
        d = d ** 2
    return (d.min(dim=2).values.mean(dim=1).mean(),
            d.min(dim=1).values.mean(dim=1).mean())


def repulsion_loss(pred: torch.Tensor, k: int = 8,
                   radius: float = 0.05) -> torch.Tensor:
    """排斥损失（PU-Net/PU-GAN 标准做法）—— 防止生成点堆叠。

    只惩罚距离小于 radius 的邻居对，用衰减权重避免过度推开。

    ⚠️ PU-Transformer 原文**不含**此项。仅用于 A-B 组消融（B₃/B₄）。

    Parameters
    ----------
    pred : (B, N, 3)
    k : 近邻数（不含自身）
    radius : 影响半径；坐标系为 patch 归一化后（最大半径 1），故 0.05 是相对量
    """
    d = torch.cdist(pred, pred)                              # (B,N,N)
    n = pred.shape[1]
    d = d + torch.eye(n, device=pred.device).unsqueeze(0) * 1e10
    knn = d.topk(k, dim=-1, largest=False).values            # (B,N,k)
    w = torch.exp(-(knn ** 2) / (radius ** 2))
    return (F.relu(radius - knn) * w).mean()


# =============================================================================
# 组合损失
# =============================================================================

class UpsamplingLoss:
    """PUTG-Net 组合损失 —— 所有项由 config 开关控制，消融只改 config 不改代码。

    设计原则（已定为项目铁律）
    --------------------------
    每个假设做成独立开关，A/B 消融时**只动 config**。这样：
      - B₁ (CD only)  = w_cd=1, 其余 0        <- 即 PU-Transformer 纯复现
      - B₂ (+对抗)    = w_cd=1, w_adv>0
      - B₃ (+uniform) = w_cd=1, w_uniform>0
      - B₄ (全)       = 三者皆开
    代码路径完全一致，排除「改代码引入的混淆变量」。

    三个融合矛盾（M1/M2/M3）的应对，同样是开关
    ------------------------------------------
    - **M1 梯度尺度失配**：`adaptive_adv_weight` 按梯度范数比自适应缩放对抗权重
    - **M2 判别器过强**：`d_steps` 控制判别器更新频率
    - **M3 早期对抗噪声**：`adv_warmup_epochs` 让对抗项延迟introduce并线性 ramp

    这三项均可独立开关，供 D2/D3 诊断实验单独验证。
    """

    def __init__(
        self,
        w_cd: float = 1.0,
        w_adv: float = 0.0,
        w_uniform: float = 0.0,
        w_repulsion: float = 0.0,
        squared_cd: bool = True,
        # 改进 A：双向 CD 加权。默认 1.0/1.0 = 原始对称 CD（与 B-001 逐位等价）
        # 依据：B-001 满 100 epoch 实测 cd_bwd_share mean=0.5446（min .5285/max .5678），
        #       反向项（覆盖）全程大于正向项（精度），无一次交叉。
        # 用法：A 组设 w_cd_fwd>1 提高「精度」项权重，或 w_cd_bwd>1 强化「覆盖」项。
        # ⚠️ 二者不做归一化，故改变它们会同时改变 CD 项的总尺度；
        #    A 组配置须显式声明是否用 (w_fwd+w_bwd)/2 = 1 的等和约束。
        w_cd_fwd: float = 1.0,
        w_cd_bwd: float = 1.0,
        # M3：对抗 warmup
        adv_warmup_epochs: int = 0,
        adv_ramp_epochs: int = 10,
        # M1：梯度自适应
        adaptive_adv: bool = False,
        adv_target_ratio: float = 0.1,
        # uniform_loss 参数
        uniform_percentages: tuple[float, ...] = (0.004, 0.006, 0.008,
                                                  0.010, 0.012),
        repulsion_k: int = 8,
        repulsion_radius: float = 0.05,
    ) -> None:
        self.w_cd = w_cd
        self.w_adv = w_adv
        self.w_uniform = w_uniform
        self.w_repulsion = w_repulsion
        self.squared_cd = squared_cd
        self.w_cd_fwd = w_cd_fwd
        self.w_cd_bwd = w_cd_bwd
        self.adv_warmup_epochs = adv_warmup_epochs
        self.adv_ramp_epochs = adv_ramp_epochs
        self.adaptive_adv = adaptive_adv
        self.adv_target_ratio = adv_target_ratio
        self.uniform_percentages = uniform_percentages
        self.repulsion_k = repulsion_k
        self.repulsion_radius = repulsion_radius

    @property
    def needs_gan(self) -> bool:
        """是否需要判别器 —— 决定训练脚本要不要建 D 与 D 的 optimizer。"""
        return self.w_adv > 0

    def config(self) -> dict:
        """导出配置，供 run 目录落盘（复现性要求）。"""
        return {
            "w_cd": self.w_cd, "w_adv": self.w_adv,
            "w_uniform": self.w_uniform, "w_repulsion": self.w_repulsion,
            "w_cd_fwd": self.w_cd_fwd, "w_cd_bwd": self.w_cd_bwd,
            "squared_cd": self.squared_cd,
            "adv_warmup_epochs": self.adv_warmup_epochs,
            "adv_ramp_epochs": self.adv_ramp_epochs,
            "adaptive_adv": self.adaptive_adv,
            "adv_target_ratio": self.adv_target_ratio,
            "uniform_percentages": list(self.uniform_percentages),
            "repulsion_k": self.repulsion_k,
            "repulsion_radius": self.repulsion_radius,
        }

    @staticmethod
    def _default_shared_params(model) -> list:
        """M1 用的「共享参数」默认取法。

        `adaptive_adv_weight` 比较的是 CD 梯度与对抗梯度**在同一组参数上**的范数比，
        因此这组参数必须同时被两条损失路径覆盖 —— 即生成器参数（判别器不参与 G 的 CD 路径）。

        默认取生成器全部 `requires_grad` 参数。
        训练脚本可显式传 `shared_params` 覆盖（例如只取最后的坐标回归层，
        使权重对浅层特征分布变化不那么敏感）；两种取法均属可消融的实现选择，
        须在 run config 中记录 `shared_params_scope`。
        """
        gen = getattr(model, "generator", None)
        if gen is None:
            return []
        return [p for p in gen.parameters() if p.requires_grad]

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        model=None,
        epoch: int = 0,
        shared_params: list | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """计算生成器总损失。

        Parameters
        ----------
        pred  : (B, N_out, 3) 生成点云
        gt    : (B, M, 3) 真值点云
        model : PUTransGAN 实例（需要对抗项时必传，用于取判别器）
        epoch : 当前 epoch（用于 M3 warmup）
        shared_params : M1 自适应权重所用的共享参数组。
            None 时回落到 `_default_shared_params(model)`（生成器全部可训参数）。

        Returns
        -------
        (loss, logs) —— logs 逐项记录标量，便于 TensorBoard 与过程图
        """
        logs: dict[str, float] = {}
        device = pred.device
        loss = torch.zeros((), device=device)

        # --- CD（主项，恒开）---
        # 改进 A：双向加权。w_cd_fwd=w_cd_bwd=1.0 时退化为原始对称 CD，
        # 与 B-001/B-002 逐位等价（见 self_check 的等价性用例）。
        fwd, bwd = chamfer_loss_split(pred, gt, squared=self.squared_cd)
        l_cd = self.w_cd_fwd * fwd + self.w_cd_bwd * bwd
        loss = loss + self.w_cd * l_cd
        logs["cd"] = float(l_cd.item())
        logs["cd_fwd"] = float(fwd.item())      # 精度项（未加权原值）
        logs["cd_bwd"] = float(bwd.item())      # 覆盖项（未加权原值）
        # share 用未加权原值算，保证与 B-001 的历史记录同定义、可跨 run 比较
        logs["cd_bwd_share"] = float((bwd / (fwd + bwd + 1e-12)).item())
        if self.w_cd_fwd != 1.0 or self.w_cd_bwd != 1.0:
            # 加权生效时额外留痕，便于事后区分 A 组与对照组
            logs["cd_w_fwd"] = float(self.w_cd_fwd)
            logs["cd_w_bwd"] = float(self.w_cd_bwd)

        # --- 对抗项（M3 warmup + M1 自适应）---
        if self.w_adv > 0:
            if model is None or not getattr(model, "use_gan", False):
                raise ValueError("w_adv>0 但未提供带判别器的 model")
            from puvnet.models.pu_gan import adv_warmup_factor

            l_adv = model.g_adv_loss(pred)
            factor = adv_warmup_factor(epoch, self.adv_warmup_epochs,
                                       self.adv_ramp_epochs)
            w_eff = self.w_adv * factor

            # M1：按梯度范数比自适应缩放（仅在对抗项已激活且需要梯度时）
            if self.adaptive_adv and factor > 0:
                sp = (shared_params if shared_params is not None
                      else self._default_shared_params(model))
                if not sp:
                    # 拿不到共享参数 → 退回固定权重，但必须留痕（不静默降级）
                    logs["adv_adaptive_error"] = "shared_params 为空，无法计算梯度比"
                else:
                    from puvnet.models.pu_gan import adaptive_adv_weight
                    try:
                        w_auto = adaptive_adv_weight(
                            l_cd, l_adv, sp,
                            target_ratio=self.adv_target_ratio)
                        w_eff = w_auto * factor
                        logs["adv_w_adaptive"] = float(w_auto)
                    except RuntimeError as e:
                        # 梯度图不可用时（如 no_grad 环境）退回固定权重，但必须留痕
                        logs["adv_adaptive_error"] = f"{type(e).__name__}: {e}"

            loss = loss + w_eff * l_adv
            logs["adv"] = float(l_adv.item())
            logs["adv_warmup_factor"] = float(factor)
            logs["adv_w_effective"] = float(w_eff)

        # --- uniform 项 ---
        if self.w_uniform > 0:
            from puvnet.models.pu_gan import uniform_loss
            # uniform_loss 返回 (B,)，需对 batch 取均值
            l_uni = uniform_loss(pred,
                                 percentages=self.uniform_percentages).mean()
            loss = loss + self.w_uniform * l_uni
            logs["uniform"] = float(l_uni.item())

        # --- repulsion 项 ---
        if self.w_repulsion > 0:
            l_rep = repulsion_loss(pred, k=self.repulsion_k,
                                   radius=self.repulsion_radius)
            loss = loss + self.w_repulsion * l_rep
            logs["repulsion"] = float(l_rep.item())

        logs["total"] = float(loss.item())
        return loss, logs


# =============================================================================
# 自检
# =============================================================================

def self_check(verbose: bool = True) -> bool:
    """自检：与 numpy 参考实现交叉验证 + 损失性质 + 开关行为。

    验证项
    ------
    1. chamfer_loss(squared=True) 与 metrics 官方 squared CD 数值一致
    2. 相同点集 CD ≈ 0
    3. fwd + bwd == cd（分项自洽）
    4. repulsion：堆叠点 > 均匀点（方向性）
    5. 梯度可回传
    6. B₁ 配置（CD only）不需要判别器，且 logs 里无 adv 项
    7. w_adv>0 但无 model 时必须报错（防静默跳过对抗项）
    8. M3 warmup：epoch < warmup 时对抗有效权重为 0
    9. 通道序错误（B,3,N）必须报错，不静默算错
    10. M1 自适应权重的**真实路径**（真建判别器真跑真回传，不只测报错分支）
    11. shared_params 显式传入确实生效（取不同参数组 → 权重不同）
    """
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from puvnet.metrics.pointcloud import chamfer_distance

    ok = True

    def log(*a):
        if verbose:
            print(*a)

    log("=" * 66)
    log("losses/upsampling.py 自检")
    log("=" * 66)

    torch.manual_seed(0)
    p = torch.randn(1, 200, 3, dtype=torch.float64)
    g = torch.randn(1, 300, 3, dtype=torch.float64)

    # --- 1. 与 numpy 官方 squared CD 交叉验证 ---
    mine = chamfer_loss(p, g, squared=True).item()
    ref = chamfer_distance(p[0].numpy(), g[0].numpy(), squared=True)
    d1 = abs(mine - ref)
    c1 = d1 < 1e-9
    ok &= c1
    log(f"[{'PASS' if c1 else 'FAIL'}] 1. torch vs numpy(squared): "
        f"{mine:.12f} vs {ref:.12f} (diff={d1:.2e})")

    # --- 2. 自身 CD ≈ 0 ---
    z = chamfer_loss(p, p).item()
    c2 = z < 1e-12
    ok &= c2
    log(f"[{'PASS' if c2 else 'FAIL'}] 2. 自身 CD = {z:.3e} (期望 ~0)")

    # --- 3. 分项自洽 ---
    fwd, bwd = chamfer_loss_split(p, g, squared=True)
    d3 = abs((fwd + bwd).item() - mine)
    c3 = d3 < 1e-12
    ok &= c3
    log(f"[{'PASS' if c3 else 'FAIL'}] 3. fwd+bwd == cd: "
        f"{fwd.item():.6f}+{bwd.item():.6f} (diff={d3:.2e})")

    # --- 3b. 改进 A 双向加权：默认值必须与原始对称 CD 逐位等价 ---
    # 这条是 B-001/B-002 结果有效性的护栏：若默认值改变了 CD，
    # 则已跑完的 baseline 全部失效。必须逐位相等（diff == 0.0），不是"接近"。
    lf_def = UpsamplingLoss(w_cd=1.0)
    _, lg_def = lf_def(p, g, epoch=0)
    lf_expl = UpsamplingLoss(w_cd=1.0, w_cd_fwd=1.0, w_cd_bwd=1.0)
    _, lg_expl = lf_expl(p, g, epoch=0)
    d3b = abs(lg_def["cd"] - (fwd + bwd).item())
    c3b = (d3b == 0.0) and (lg_def["cd"] == lg_expl["cd"])
    ok &= c3b
    log(f"[{'PASS' if c3b else 'FAIL'}] 3b. A 组默认 (1.0,1.0) 与对称 CD 逐位等价: "
        f"cd={lg_def['cd']:.9f} diff={d3b:.1e}")

    # 加权确实生效，且方向正确
    lf_a = UpsamplingLoss(w_cd=1.0, w_cd_fwd=0.5, w_cd_bwd=1.5)
    _, lg_a = lf_a(p, g, epoch=0)
    expect_a = 0.5 * fwd.item() + 1.5 * bwd.item()
    d3c = abs(lg_a["cd"] - expect_a)
    c3c = d3c < 1e-9 and abs(lg_a["cd"] - lg_def["cd"]) > 1e-9
    ok &= c3c
    log(f"[{'PASS' if c3c else 'FAIL'}] 3c. A 组加权生效 (0.5,1.5): "
        f"cd={lg_a['cd']:.6f} 期望={expect_a:.6f} (diff={d3c:.1e})")

    # cd_bwd_share 必须用未加权原值，加权后不得改变（跨 run 可比性）
    d3d = abs(lg_a["cd_bwd_share"] - lg_def["cd_bwd_share"])
    c3d = d3d < 1e-12
    ok &= c3d
    log(f"[{'PASS' if c3d else 'FAIL'}] 3d. cd_bwd_share 与加权无关: "
        f"{lg_def['cd_bwd_share']:.6f} vs {lg_a['cd_bwd_share']:.6f} (diff={d3d:.1e})")

    # 加权时必须留痕，默认时不留（便于事后区分 A 组）
    c3e = ("cd_w_fwd" in lg_a and lg_a["cd_w_bwd"] == 1.5
           and "cd_w_fwd" not in lg_def)
    ok &= c3e
    log(f"[{'PASS' if c3e else 'FAIL'}] 3e. 加权留痕: A 组有 cd_w_fwd/bwd，对照组无")

    # --- 4. repulsion 方向性 ---
    n_side = 5
    lin = torch.linspace(-0.5, 0.5, n_side)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1)
    uniform = grid.reshape(1, -1, 3)
    clumped = torch.randn(1, n_side ** 3, 3) * 0.01
    r_c = repulsion_loss(clumped).item()
    r_u = repulsion_loss(uniform).item()
    c4 = r_c > r_u
    ok &= c4
    log(f"[{'PASS' if c4 else 'FAIL'}] 4. repulsion 堆叠={r_c:.6f} > "
        f"均匀={r_u:.6f}")

    # --- 5. 梯度回传 ---
    q = torch.randn(1, 50, 3, requires_grad=True)
    chamfer_loss(q, g.float()).backward()
    gn = q.grad.norm().item()
    c5 = gn > 0
    ok &= c5
    log(f"[{'PASS' if c5 else 'FAIL'}] 5. 梯度范数={gn:.4f} (期望 >0)")

    # --- 6. B₁ 配置（CD only）---
    lf_b1 = UpsamplingLoss(w_cd=1.0)
    l_b1, logs_b1 = lf_b1(q.detach().float(), g.float())
    c6 = (not lf_b1.needs_gan) and ("adv" not in logs_b1) \
        and ("uniform" not in logs_b1) and ("repulsion" not in logs_b1)
    ok &= c6
    log(f"[{'PASS' if c6 else 'FAIL'}] 6. B1(CD only) needs_gan="
        f"{lf_b1.needs_gan} logs={sorted(logs_b1.keys())}")

    # --- 7. w_adv>0 无 model 必须报错 ---
    lf_bad = UpsamplingLoss(w_cd=1.0, w_adv=0.1)
    try:
        lf_bad(q.detach().float(), g.float(), model=None)
        c7 = False
    except ValueError:
        c7 = True
    ok &= c7
    log(f"[{'PASS' if c7 else 'FAIL'}] 7. w_adv>0 缺 model 时报错 = {c7} "
        f"(防静默跳过对抗项)")

    # --- 8. M3 warmup 行为 ---
    from puvnet.models.pu_gan import adv_warmup_factor
    f_in = adv_warmup_factor(2, 5, 10)
    f_out = adv_warmup_factor(20, 5, 10)
    c8 = abs(f_in) < 1e-12 and f_out > 0
    ok &= c8
    log(f"[{'PASS' if c8 else 'FAIL'}] 8. warmup: epoch2(warmup=5)="
        f"{f_in:.4f} (期望 0), epoch20={f_out:.4f} (期望 >0)")

    # --- 9. 通道序错误必须报错 ---
    try:
        chamfer_loss(torch.randn(1, 3, 200), torch.randn(1, 3, 300))
        c9 = False
    except ValueError:
        c9 = True
    ok &= c9
    log(f"[{'PASS' if c9 else 'FAIL'}] 9. (B,3,N) 通道序报错 = {c9} "
        f"(防静默算错)")

    # --- 10. M1 自适应权重真实路径（必须真跑，不能只测报错分支）---
    # 判据（预注册）：logs 必须含 adv_w_adaptive 且不含 adv_adaptive_error，
    #                且 adv_w_effective > 0，且 loss 可回传。
    from puvnet.models.pu_gan import PUTransGAN, PUGANDiscriminator
    from puvnet.models.pu_transformer import PUTransformer

    torch.manual_seed(1)
    # dims 取小值加速；shuffle 模式要求 dims[-1] % up_ratio == 0（32%4==0 合法）
    gen = PUTransformer(up_ratio=4, dims=(16, 32), k=8)
    dis = PUGANDiscriminator()
    m = PUTransGAN(gen, discriminator=dis)
    xin = torch.randn(1, 64, 3) * 0.3
    xgt = torch.randn(1, 256, 3) * 0.3
    out = m(xin)
    lf10 = UpsamplingLoss(w_cd=1.0, w_adv=1.0, adaptive_adv=True,
                          adv_warmup_epochs=0)
    l10, logs10 = lf10(out, xgt, model=m, epoch=5)
    l10.backward()
    c10 = ("adv_w_adaptive" in logs10
           and "adv_adaptive_error" not in logs10
           and logs10["adv_w_effective"] > 0
           and any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in gen.parameters()))
    ok &= c10
    log(f"[{'PASS' if c10 else 'FAIL'}] 10. M1 自适应真实路径: "
        f"w_adaptive={logs10.get('adv_w_adaptive')} "
        f"w_eff={logs10.get('adv_w_effective')} "
        f"err={logs10.get('adv_adaptive_error', 'none')}")

    # --- 11. shared_params 显式传入应生效且与默认取法不同 ---
    # 判据：只取最后一层参数时，梯度比与取全部参数时不同（说明参数选择确有影响，
    #      不是被内部忽略）。这条也提醒：shared_params_scope 是须记录的消融维度。
    tail = [p for p in list(gen.parameters())[-2:] if p.requires_grad]
    out2 = m(xin)
    _, logs11 = lf10(out2, xgt, model=m, epoch=5, shared_params=tail)
    w_all = logs10.get("adv_w_adaptive")
    w_tail = logs11.get("adv_w_adaptive")
    c11 = (w_tail is not None and "adv_adaptive_error" not in logs11
           and abs(w_tail - w_all) > 1e-12)
    ok &= c11
    log(f"[{'PASS' if c11 else 'FAIL'}] 11. shared_params 生效: "
        f"全部={w_all} 尾层={w_tail}")

    log("-" * 66)
    log(f"总体: {'PASS' if ok else 'FAIL'}")
    log("=" * 66)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
