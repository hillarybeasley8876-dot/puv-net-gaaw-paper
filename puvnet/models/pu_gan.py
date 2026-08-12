"""PU-GAN 判别器 + PU-Transformer 融合方案（本文核心创新）。

设计依据
--------
1. PU-GAN (Li et al. ICCV 2019) 的判别器思路：自注意力增强的 PointNet 式判别器，
   对上采样结果的**局部真实性与均匀性**施加对抗约束。
2. 原版 PU-Transformer **只用 modified CD 损失**（见 docs/PU_TRANSFORMER_SPEC.md §4.4），
   无对抗、无 uniform、无 repulsion 约束。

创新点的机制假设（H-GAN-1）
--------------------------
来自 PU-Transformer 论文 Tab.4（噪声鲁棒性，见 SPEC §6.4）的实测观察：

    噪声 β=2% 时，PU-Transformer 的 CD=1.058，输给 Dis-PU 的 0.858；
    但 HD=9.948 / P2F=7.551 仍显著领先。

解读：纯 Transformer + 纯 CD 损失在强噪声下会产生**整体分布偏移**（CD 敏感），
      但极端离群点控制得好（HD 好）。
      而 CD 损失本身对"点堆在一起"是不敏感的 —— 只要平均最近邻距离小就行。

假设：PU-GAN 的对抗损失 + uniform 损失正是针对分布均匀性设计的，
      与 Transformer 的全局建模能力**互补而非重复**。
      若融合后能在 β=2% 时压低 CD 且不牺牲 HD/P2F，则创新点在数据上成立。

⚠️ 这是**待验证假设**，不是结论。实验若不支持，必须如实报告（见 SOTA_SURVEY §6）。

融合中的真实矛盾（这才是"结合"的学术内容）
-----------------------------------------
简单把判别器接到 Transformer 输出上会遇到三个具体问题，本模块逐一给出可开关的解法：

  M1 梯度尺度失配：CD 损失量级 ~1e-3，对抗损失量级 ~1e0，直接相加会让 CD 被淹没。
     → 解法：`adaptive_weight` 动态平衡（按梯度范数归一化），可开关对比固定权重。

  M2 判别器过强导致生成器塌缩：Transformer 参数量近 1.2M，判别器若过强会主导训练。
     → 解法：`n_critic` 控制更新频率 + 谱归一化稳定判别器。

  M3 训练早期对抗信号是噪声：Transformer 尚未学会基本形状时，对抗梯度是有害的。
     → 解法：`warmup_epochs` 内只用 CD，之后线性引入对抗损失。

以上三项都是 config 开关，消融可直接对比，不改代码。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 自注意力单元（PU-GAN 判别器用）
# =============================================================================

class SelfAttention(nn.Module):
    """PU-GAN 判别器中的自注意力单元。

    结构依 PU-GAN 论文：f/g/h 三路 1x1 conv，f·g 得 attention map，
    加权 h 后以可学习标量 gamma 残差融合（gamma 初始化为 0，
    使训练初期退化为恒等映射 —— 这对稳定性很重要）。
    """

    def __init__(self, in_ch: int, reduction: int = 8):
        super().__init__()
        mid = max(1, in_ch // reduction)
        self.f = nn.Conv1d(in_ch, mid, 1)
        self.g = nn.Conv1d(in_ch, mid, 1)
        self.h = nn.Conv1d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) -> (B, C, N)"""
        fx = self.f(x)                                   # (B, mid, N)
        gx = self.g(x)                                   # (B, mid, N)
        hx = self.h(x)                                   # (B, C, N)
        attn = F.softmax(torch.bmm(fx.transpose(1, 2), gx), dim=-1)  # (B,N,N)
        out = torch.bmm(hx, attn.transpose(1, 2))        # (B, C, N)
        return self.gamma * out + x


# =============================================================================
# PU-GAN 判别器
# =============================================================================

class PUGANDiscriminator(nn.Module):
    """PU-GAN 风格判别器。

    结构：MLP 升维 → 全局特征拼接 → 自注意力 → MLP 降维 → 全局池化 → 标量分数

    Parameters
    ----------
    in_ch : 输入通道（3 = xyz）
    dims : 逐层通道
    use_attention : 是否用自注意力（消融开关）
    spectral_norm : 是否对 conv 加谱归一化。
        针对 M2（判别器过强）—— 谱归一化限制 Lipschitz 常数，是稳定 GAN 的标准手段。
    """

    def __init__(
        self,
        in_ch: int = 3,
        dims: tuple[int, ...] = (64, 128, 256),
        use_attention: bool = True,
        spectral_norm: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        def conv1d(i: int, o: int) -> nn.Module:
            layer = nn.Conv1d(i, o, 1)
            if spectral_norm:
                layer = nn.utils.parametrizations.spectral_norm(layer)
            return layer

        # 第一段：逐点特征提取
        self.mlp1 = nn.Sequential(
            conv1d(in_ch, dims[0]), nn.LeakyReLU(0.2, inplace=True),
            conv1d(dims[0], dims[1]), nn.LeakyReLU(0.2, inplace=True),
        )
        # 全局特征拼接后通道翻倍
        self.attn = SelfAttention(dims[1] * 2) if use_attention else None
        # 第二段
        self.mlp2 = nn.Sequential(
            conv1d(dims[1] * 2, dims[2]), nn.LeakyReLU(0.2, inplace=True),
            conv1d(dims[2], dims[2]), nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(dims[2], 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pc : (B, N, 3) 点云

        Returns
        -------
        (B,) 判别分数（未过 sigmoid，配合 BCEWithLogits 或 hinge loss 使用）
        """
        if pc.dim() != 3 or pc.size(-1) != 3:
            raise ValueError(f"期望 (B,N,3)，收到 {tuple(pc.shape)}")

        x = pc.transpose(1, 2)                           # (B, 3, N)
        feat = self.mlp1(x)                              # (B, d1, N)
        # 拼接全局最大池化特征（PointNet 式）
        glob = feat.max(dim=2, keepdim=True).values.expand_as(feat)
        feat = torch.cat([feat, glob], dim=1)            # (B, 2*d1, N)
        if self.attn is not None:
            feat = self.attn(feat)
        feat = self.mlp2(feat)                           # (B, d2, N)
        pooled = feat.max(dim=2).values                  # (B, d2)
        return self.head(pooled).squeeze(-1)             # (B,)


# =============================================================================
# 均匀性损失（PU-GAN）
# =============================================================================

def uniform_loss(
    pred: torch.Tensor,
    percentages: tuple[float, ...] = (0.004, 0.006, 0.008, 0.010, 0.012),
    radius: float = 1.0,
    n_seeds: int = 20,
) -> torch.Tensor:
    """PU-GAN 的均匀性损失（可微版）。

    思路：在点云上随机取 seed，统计各 seed 半径 r 邻域内的点数，
    与期望点数比较（imbalance 项）；同时惩罚邻域内点距偏离理想间距（clutter 项）。

    这是 PU-GAN 相对 PU-Transformer 的**关键增量约束** ——
    CD 损失对"点分布不均"不敏感，而这一项直接优化均匀性。

    Parameters
    ----------
    pred : (B, N, 3) 归一化到单位球的点云
    percentages : 各档邻域面积占比 p
    radius : 点云半径（归一化后为 1）
    n_seeds : 每档采样的 seed 数

    Returns
    -------
    (B,) 每个样本的均匀性损失
    """
    b, n, _ = pred.shape
    device = pred.device
    total = pred.new_zeros(b)

    for p in percentages:
        n_expect = n * p
        r = (p ** 0.5) * radius          # disk 半径
        # 随机选 seed
        idx = torch.randint(0, n, (n_seeds,), device=device)
        seeds = pred[:, idx, :]                                  # (B, S, 3)
        dist = torch.cdist(seeds, pred, p=2)                     # (B, S, N)
        mask = (dist < r).float()
        cnt = mask.sum(dim=2)                                    # (B, S)

        # imbalance: 实际点数与期望的相对偏差
        imbalance = (cnt - n_expect) ** 2 / (n_expect + 1e-8)

        # clutter: 邻域内最近邻距离应接近理想间距
        ideal = (2 * 3.1415926 * r ** 2 / (cnt.clamp_min(1) * (3 ** 0.5))).sqrt()
        # 邻域内的最近非零距离
        big = dist + (1 - mask) * 1e6
        # 排除自身（距离 0），取次小
        nn_d = big.topk(k=2, dim=2, largest=False).values[:, :, 1]  # (B, S)
        nn_d = torch.where(nn_d > 1e5, ideal, nn_d)
        clutter = (nn_d - ideal) ** 2 / (ideal + 1e-8)

        total = total + (imbalance * clutter).mean(dim=1)

    return total / len(percentages)


# =============================================================================
# 融合模型：PU-Transformer 生成器 + PU-GAN 判别器
# =============================================================================

class PUTransGAN(nn.Module):
    """本文方法：PU-Transformer 主干 + PU-GAN 对抗约束。

    这不是简单拼接 —— 三个机制性设计针对融合中的真实矛盾（见模块 docstring）：
      M1 梯度尺度失配 → adaptive_weight
      M2 判别器过强   → n_critic + spectral_norm
      M3 早期对抗噪声 → warmup_epochs

    每一项都是可开关的消融维度。

    Parameters
    ----------
    generator : PUTransformer 实例
    discriminator : PUGANDiscriminator 实例
    gan_mode : 'hinge' | 'bce' | 'lsgan' 对抗损失形式
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module | None = None,
        gan_mode: str = "hinge",
    ):
        super().__init__()
        if gan_mode not in ("hinge", "bce", "lsgan"):
            raise ValueError(f"gan_mode 只能是 hinge|bce|lsgan，收到 {gan_mode}")
        self.generator = generator
        self.discriminator = discriminator
        self.gan_mode = gan_mode

    @property
    def use_gan(self) -> bool:
        return self.discriminator is not None

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.generator(xyz)

    # ---------- 对抗损失 ----------

    def d_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """判别器损失。real/fake 均为 (B, N, 3)。"""
        if self.discriminator is None:
            raise RuntimeError("未配置判别器")
        s_real = self.discriminator(real)
        s_fake = self.discriminator(fake.detach())
        if self.gan_mode == "hinge":
            return (F.relu(1.0 - s_real).mean() + F.relu(1.0 + s_fake).mean())
        if self.gan_mode == "lsgan":
            return ((s_real - 1) ** 2).mean() + (s_fake ** 2).mean()
        return (F.binary_cross_entropy_with_logits(s_real, torch.ones_like(s_real))
                + F.binary_cross_entropy_with_logits(s_fake, torch.zeros_like(s_fake)))

    def g_adv_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """生成器的对抗损失项。"""
        if self.discriminator is None:
            raise RuntimeError("未配置判别器")
        s_fake = self.discriminator(fake)
        if self.gan_mode == "hinge":
            return -s_fake.mean()
        if self.gan_mode == "lsgan":
            return ((s_fake - 1) ** 2).mean()
        return F.binary_cross_entropy_with_logits(s_fake, torch.ones_like(s_fake))

    def count_parameters(self) -> dict[str, int]:
        g = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d = (sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
             if self.discriminator is not None else 0)
        return {"generator": g, "discriminator": d, "total": g + d}


# =============================================================================
# M1 解法：自适应损失权重
# =============================================================================

def adaptive_adv_weight(
    cd_loss: torch.Tensor,
    adv_loss: torch.Tensor,
    shared_params: list[torch.nn.Parameter],
    target_ratio: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """按梯度范数动态平衡 CD 与对抗损失（针对 M1 梯度尺度失配）。

    做法：分别计算两个损失对共享参数的梯度范数，
    令 adv 权重 = target_ratio * ||grad_cd|| / ||grad_adv||，
    使对抗梯度的量级稳定维持在 CD 梯度的 target_ratio 倍。

    这比手调固定权重更可靠 —— CD 损失量级在训练中会下降 3 个数量级，
    固定权重必然在某个阶段失衡。

    Returns
    -------
    float 建议的对抗损失权重（标量，detach 后使用）
    """
    g_cd = torch.autograd.grad(cd_loss, shared_params,
                               retain_graph=True, allow_unused=True)
    g_adv = torch.autograd.grad(adv_loss, shared_params,
                                retain_graph=True, allow_unused=True)
    n_cd = torch.sqrt(sum((g ** 2).sum() for g in g_cd if g is not None) + eps)
    n_adv = torch.sqrt(sum((g ** 2).sum() for g in g_adv if g is not None) + eps)
    return float((target_ratio * n_cd / (n_adv + eps)).item())


# =============================================================================
# M3 解法：对抗损失 warmup
# =============================================================================

def adv_warmup_factor(epoch: int, warmup_epochs: int, ramp_epochs: int = 10) -> float:
    """对抗损失的引入系数（针对 M3 早期对抗噪声）。

    epoch < warmup_epochs           → 0.0（纯 CD，先学基本形状）
    warmup ≤ epoch < warmup+ramp    → 线性上升
    之后                            → 1.0
    """
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + ramp_epochs:
        return (epoch - warmup_epochs + 1) / ramp_epochs
    return 1.0


# =============================================================================
# 自检
# =============================================================================

def self_check() -> bool:
    ok = True
    torch.manual_seed(0)

    print("=" * 70)
    print("models/pu_gan.py 自检")
    print("=" * 70)

    b, n = 2, 128

    # --- 1. 判别器前向 ---
    d = PUGANDiscriminator()
    pc = torch.randn(b, n, 3)
    s = d(pc)
    p1 = s.shape == (b,)
    ok &= p1
    print(f"[{'PASS' if p1 else 'FAIL'}] 判别器: {tuple(pc.shape)} -> {tuple(s.shape)} "
          f"(期望 {(b,)}), 参数 {sum(p.numel() for p in d.parameters()):,}")

    # --- 2. 自注意力 gamma 初始为 0 -> 恒等映射 ---
    sa = SelfAttention(64)
    x = torch.randn(b, 64, n)
    out = sa(x)
    p2 = torch.allclose(out, x, atol=1e-6)
    ok &= p2
    print(f"[{'PASS' if p2 else 'FAIL'}] 自注意力 gamma=0 时为恒等映射 "
          f"(max diff={float((out-x).abs().max()):.2e})")

    # --- 3. 三种 GAN 损失 ---
    from puvnet.models.pu_transformer import PUTransformer
    for mode in ("hinge", "bce", "lsgan"):
        try:
            g = PUTransformer(up_ratio=4)
            m = PUTransGAN(g, PUGANDiscriminator(), gan_mode=mode)
            xin = torch.randn(b, 32, 3)
            fake = m(xin)
            real = torch.randn(b, 128, 3)
            dl = m.d_loss(real, fake)
            gl = m.g_adv_loss(fake)
            assert dl.numel() == 1 and gl.numel() == 1
            assert torch.isfinite(dl) and torch.isfinite(gl)
            print(f"[PASS] gan_mode={mode:<6} d_loss={dl.item():+.4f} "
                  f"g_adv={gl.item():+.4f}")
        except Exception as e:
            ok = False
            print(f"[FAIL] gan_mode={mode}: {type(e).__name__}: {e}")

    # --- 4. uniform_loss 能区分均匀 vs 聚集 ---
    nn_pts = 512
    idx = torch.arange(nn_pts, dtype=torch.float64) + 0.5
    phi = torch.acos(1 - 2 * idx / nn_pts)
    theta = 3.14159265 * (1 + 5 ** 0.5) * idx
    sphere = torch.stack([
        (theta.cos() * phi.sin()), (theta.sin() * phi.sin()), phi.cos()
    ], dim=1).float().unsqueeze(0)
    clustered = sphere.clone()
    clustered[:, : nn_pts // 2] = sphere[:, : nn_pts // 2] * 0.05
    lu = uniform_loss(sphere).item()
    lc = uniform_loss(clustered).item()
    p4 = lu < lc
    ok &= p4
    print(f"[{'PASS' if p4 else 'FAIL'}] uniform_loss 均匀={lu:.4f} < 聚集={lc:.4f}")

    # --- 5. warmup 系数 ---
    vals = [adv_warmup_factor(e, warmup_epochs=5, ramp_epochs=10) for e in
            (0, 4, 5, 9, 14, 15, 100)]
    expect_zero = vals[0] == 0.0 and vals[1] == 0.0
    expect_one = vals[-1] == 1.0 and vals[-2] == 1.0
    monotonic = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    p5 = expect_zero and expect_one and monotonic
    ok &= p5
    print(f"[{'PASS' if p5 else 'FAIL'}] warmup 系数 epoch(0,4,5,9,14,15,100) -> "
          f"{[round(v,2) for v in vals]} (单调={monotonic})")

    # --- 6. 自适应权重 ---
    try:
        g = PUTransformer(up_ratio=4)
        m = PUTransGAN(g, PUGANDiscriminator())
        xin = torch.randn(b, 32, 3)
        fake = m(xin)
        gt = torch.randn(b, 128, 3)
        from puvnet.metrics.pointcloud import chamfer_distance_torch
        cd = chamfer_distance_torch(fake, gt).mean()
        adv = m.g_adv_loss(fake)
        params = [p for p in g.parameters() if p.requires_grad]
        w = adaptive_adv_weight(cd, adv, params, target_ratio=0.1)
        p6 = w > 0 and w == w  # 正数且非 NaN
        ok &= p6
        print(f"[{'PASS' if p6 else 'FAIL'}] 自适应权重 = {w:.6f} "
              f"(cd={cd.item():.6f}, adv={adv.item():+.4f})")
    except Exception as e:
        ok = False
        print(f"[FAIL] 自适应权重: {type(e).__name__}: {e}")

    # --- 7. 消融：无判别器（纯 PU-Transformer 基线）---
    m_nogan = PUTransGAN(PUTransformer(up_ratio=4), discriminator=None)
    p7 = not m_nogan.use_gan and m_nogan.count_parameters()["discriminator"] == 0
    ok &= p7
    cnt = m_nogan.count_parameters()
    print(f"[{'PASS' if p7 else 'FAIL'}] 纯 Transformer 基线（无 GAN）: "
          f"G={cnt['generator']:,} D={cnt['discriminator']:,}")

    # --- 8. 完整模型参数量 ---
    m_full = PUTransGAN(PUTransformer(up_ratio=4), PUGANDiscriminator())
    cnt = m_full.count_parameters()
    print(f"[INFO] 完整模型: G={cnt['generator']:,} "
          f"D={cnt['discriminator']:,} 合计={cnt['total']:,}")

    print("-" * 70)
    print(f"总体: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
