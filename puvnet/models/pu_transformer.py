"""PU-Transformer 主干（PyTorch 复现）。

规格来源：docs/PU_TRANSFORMER_SPEC.md（从 arXiv TeX 源码 2111.12242 提取）
原论文：Qiu, Anwar, Barnes. "PU-Transformer: Point Cloud Upsampling Transformer". ACCV 2022.

⚠️ 原论文无官方开源代码。本文件为 PyTorch 从零复现，
   保真度验收标准：L=5, psi=4, k=20, r=4 时参数量应接近 969.9k（论文 Tab.5）。

架构（spec §1）
--------------
    Head : F_0 = MLP(P)
    Body : for l in 1..L:
               G_l  = PosFus(P, F_{l-1})
               G_l' = SC-MSA(Norm(G_l)) + G_l        # pre-norm 残差
               F_l  = MLP(Norm(G_l')) + G_l'         # pre-norm 残差
    Tail : S = MLP(Shuffle(F_L))

张量布局约定
-----------
本模块内部统一用 **(B, N, C)** 布局（channel-last），
与论文公式的 R^{N×C} 一致，便于逐行对照。
接口输入输出点云为 (B, N, 3)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 工具：knn（固定 3D 欧氏距离，全程复用）
# =============================================================================

def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """基于 3D 欧氏距离的 k 近邻索引（含自身）。

    spec §2.2 红线：PU-Transformer 用**固定**的 3D 几何 knn，
    不是 DGCNN 的动态特征空间图。索引只算一次，所有 PosFus 层复用。

    Parameters
    ----------
    xyz : (B, N, 3)
    k : 邻居数（含自身）

    Returns
    -------
    (B, N, k) 索引
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"期望 (B,N,3)，收到 {tuple(xyz.shape)}")
    n = xyz.size(1)
    k = min(k, n)
    dist = torch.cdist(xyz, xyz, p=2)          # (B, N, N)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return idx


def group_by_index(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引收集邻居特征。

    Parameters
    ----------
    feat : (B, N, C)
    idx : (B, N, k)

    Returns
    -------
    (B, N, k, C)
    """
    b, n, c = feat.shape
    k = idx.size(-1)
    # (B, N*k) -> gather -> (B, N*k, C) -> (B, N, k, C)
    flat_idx = idx.reshape(b, n * k, 1).expand(-1, -1, c)
    out = torch.gather(feat, dim=1, index=flat_idx)
    return out.reshape(b, n, k, c)


# =============================================================================
# MLP 构件
# =============================================================================

class PointMLP(nn.Module):
    """作用在最后一维的 MLP（等价于 1x1 conv），带 BN + ReLU。

    输入 (..., in_ch) → 输出 (..., out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, bn: bool = True, act: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=not bn)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn is not None:
            # BatchNorm1d 要求 channel 在 dim=1，需临时换轴
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            x = self.bn(x)
            x = x.reshape(shape)
        if self.act is not None:
            x = self.act(x)
        return x


# =============================================================================
# Positional Fusion 块（spec §2）
# =============================================================================

class PositionalFusion(nn.Module):
    """位置融合块。

    spec §2.1 公式：
        ΔP     = P_j - P                          (N, k, 3)
        G_geo  = concat[dup_k(P) ; ΔP]            (N, k, 6)
        ΔF     = F_j - F                          (N, k, C)
        G_feat = concat[dup_k(F) ; ΔF]            (N, k, 2C)
        G      = max_k( concat[M_Φ(G_geo) ; M_Θ(G_feat)] )   (N, C')

    两个 MLP 输出各占 C'/2，concat 后为 C'。

    Parameters
    ----------
    in_ch : 输入特征通道 C
    out_ch : 输出通道 C'（必须为偶数）
    use_geo : 消融开关 —— 是否使用 G_geo（对应论文 Tab.3 的 A 组）
    use_feat : 消融开关 —— 是否使用 G_feat
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_geo: bool = True,
        use_feat: bool = True,
    ):
        super().__init__()
        if not (use_geo or use_feat):
            raise ValueError("use_geo 和 use_feat 不能同时为 False")
        if out_ch % 2 != 0:
            raise ValueError(f"out_ch 必须为偶数，收到 {out_ch}")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_geo = use_geo
        self.use_feat = use_feat

        # 两路都启用时各占一半；只启用一路时该路占满
        if use_geo and use_feat:
            geo_out = out_ch // 2
            feat_out = out_ch - geo_out
        elif use_geo:
            geo_out, feat_out = out_ch, 0
        else:
            geo_out, feat_out = 0, out_ch

        # M_Φ: 编码几何上下文 (k, 6) -> (k, geo_out)
        self.mlp_geo = PointMLP(6, geo_out) if geo_out > 0 else None
        # M_Θ: 编码特征上下文 (k, 2C) -> (k, feat_out)
        self.mlp_feat = PointMLP(2 * in_ch, feat_out) if feat_out > 0 else None

    def forward(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : (B, N, 3) 原始坐标（全程不变，spec §2.2）
        feat : (B, N, C) 上一层特征
        knn_idx : (B, N, k) 预计算的固定 knn 索引

        Returns
        -------
        (B, N, C')
        """
        parts = []

        if self.mlp_geo is not None:
            xyz_j = group_by_index(xyz, knn_idx)                # (B,N,k,3)
            d_xyz = xyz_j - xyz.unsqueeze(2)                    # Eq.1
            xyz_dup = xyz.unsqueeze(2).expand_as(xyz_j)
            g_geo = torch.cat([xyz_dup, d_xyz], dim=-1)         # Eq.2 (B,N,k,6)
            parts.append(self.mlp_geo(g_geo))

        if self.mlp_feat is not None:
            feat_j = group_by_index(feat, knn_idx)              # (B,N,k,C)
            d_feat = feat_j - feat.unsqueeze(2)                 # Eq.3
            feat_dup = feat.unsqueeze(2).expand_as(feat_j)
            g_feat = torch.cat([feat_dup, d_feat], dim=-1)      # Eq.4 (B,N,k,2C)
            parts.append(self.mlp_feat(g_feat))

        g = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        # Eq.5: 邻域维 max-pooling
        return g.max(dim=2).values                              # (B, N, C')


# =============================================================================
# Shifted Channel Multi-head Self-Attention（spec §3）
# =============================================================================

class SCMSA(nn.Module):
    """通道偏移多头自注意力。

    spec §3.2 算法：Q/K/V 由 1x1 conv 得到，
    沿 channel 维用滑动窗口切 M 个 split（相邻 split 重叠 w-d 个通道），
    每个 split 独立做自注意力，最后 concat + Linear。

    参数关系（spec §4.2）：
        w = C' / psi          split 通道宽度
        d = w / 2             shift 间隔
        M = 2*psi - 1         head 数

    Parameters
    ----------
    dim : 通道数 C'
    psi : reduction ratio（论文用 4）
    attention_type : 'sc-msa' | 'msa' —— 消融开关（论文 Tab.3 B 组）
        'msa' 为常规多头注意力（各 head 互不重叠）
    scale_qk : 是否对 QK^T 做 1/sqrt(w) 缩放。
        ⚠️ 原论文算法**未写缩放**，默认 False 以忠于原文；
        若训练不稳定，作为消融项显式记录，不要默认偷偷开启。
    """

    def __init__(
        self,
        dim: int,
        psi: int = 4,
        attention_type: str = "sc-msa",
        scale_qk: bool = False,
    ):
        super().__init__()
        if attention_type not in ("sc-msa", "msa"):
            raise ValueError(f"attention_type 只能是 sc-msa|msa，收到 {attention_type}")
        if dim % psi != 0:
            raise ValueError(f"dim({dim}) 必须能被 psi({psi}) 整除")

        self.dim = dim
        self.psi = psi
        self.attention_type = attention_type
        self.scale_qk = scale_qk

        self.w = dim // psi                     # split 宽度

        if attention_type == "sc-msa":
            self.d = self.w // 2                # shift 间隔
            if self.d < 1:
                raise ValueError(f"dim={dim}, psi={psi} 导致 shift 间隔 d<1")
            self.n_heads = 2 * psi - 1
            # 自检：最后一个窗口不越界
            last_end = (self.n_heads - 1) * self.d + self.w
            if last_end != dim:
                raise ValueError(
                    f"SC-MSA 窗口未铺满: 最后窗口结束于 {last_end}，dim={dim}"
                )
        else:
            self.d = self.w                     # 常规 MSA：不重叠
            self.n_heads = psi

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # concat 后的通道数 = n_heads * w
        self.proj = nn.Linear(self.n_heads * self.w, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, C')

        Returns
        -------
        (B, N, C')
        """
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        outs = []
        for m in range(self.n_heads):
            s = m * self.d
            e = s + self.w
            q_m, k_m, v_m = q[..., s:e], k[..., s:e], v[..., s:e]
            attn = torch.matmul(q_m, k_m.transpose(-2, -1))     # (B, N, N)
            if self.scale_qk:
                attn = attn / (self.w ** 0.5)
            attn = F.softmax(attn, dim=-1)
            outs.append(torch.matmul(attn, v_m))                # (B, N, w)

        return self.proj(torch.cat(outs, dim=-1))


# =============================================================================
# Transformer Encoder
# =============================================================================

class TransformerEncoder(nn.Module):
    """单个 Transformer Encoder（spec §1 Body 循环体）。

        G   = PosFus(P, F_in)
        G'  = SC-MSA(Norm(G)) + G
        out = MLP(Norm(G')) + G'

    注意是 **pre-norm**：Norm 在子层之前，残差加未归一化的输入。

    Parameters
    ----------
    mlp_ratio : Encoder 内 MLP 的隐层扩张比。
        ⚠️ spec §9 标注为待验证项：原文未给出。
        Transformer 惯例是 4，点云工作常用 1。默认 1，用参数量对齐校准。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        psi: int = 4,
        mlp_ratio: int = 1,
        use_geo: bool = True,
        use_feat: bool = True,
        attention_type: str = "sc-msa",
        scale_qk: bool = False,
    ):
        super().__init__()
        self.posfus = PositionalFusion(in_ch, out_ch, use_geo, use_feat)
        self.norm1 = nn.LayerNorm(out_ch)
        self.attn = SCMSA(out_ch, psi, attention_type, scale_qk)
        self.norm2 = nn.LayerNorm(out_ch)
        hidden = out_ch * mlp_ratio
        self.mlp = nn.Sequential(
            PointMLP(out_ch, hidden, bn=False, act=True),
            nn.Linear(hidden, out_ch),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        g = self.posfus(xyz, feat, knn_idx)
        g = self.attn(self.norm1(g)) + g
        g = self.mlp(self.norm2(g)) + g
        return g


# =============================================================================
# PU-Transformer 主干
# =============================================================================

class PUTransformer(nn.Module):
    """PU-Transformer 生成器。

    Parameters
    ----------
    up_ratio : 上采样倍率 r
    dims : 各 Encoder 输出通道，默认 (32,64,128,256,256) —— spec §4.1 原文值
    k : PosFus 邻居数，默认 20 —— spec §4.1 原文值
    psi : SC-MSA reduction ratio，默认 4 —— spec §4.2 原文值
    head_dim : Head MLP 输出通道。spec §9 待验证项，默认与 dims[0] 一致
    mlp_ratio : Encoder 内 MLP 扩张比。spec §9 待验证项
    use_geo / use_feat / attention_type / tail_mode : 消融开关
    tail_mode : 'shuffle' —— PixelShuffle（论文采用）
                'mlp'     —— 直接 MLP 扩张（论文 Tab.3 C1，最差）
    """

    def __init__(
        self,
        up_ratio: int = 4,
        dims: tuple[int, ...] = (32, 64, 128, 256, 256),
        k: int = 20,
        psi: int = 4,
        head_dim: int | None = None,
        mlp_ratio: int = 1,
        use_geo: bool = True,
        use_feat: bool = True,
        attention_type: str = "sc-msa",
        tail_mode: str = "shuffle",
        scale_qk: bool = False,
    ):
        super().__init__()
        if tail_mode not in ("shuffle", "mlp"):
            raise ValueError(f"tail_mode 只能是 shuffle|mlp，收到 {tail_mode}")
        if tail_mode == "shuffle" and dims[-1] % up_ratio != 0:
            raise ValueError(
                f"shuffle 模式要求最后通道 {dims[-1]} 能被 up_ratio {up_ratio} 整除"
            )

        self.up_ratio = up_ratio
        self.dims = tuple(dims)
        self.k = k
        self.tail_mode = tail_mode

        head_dim = dims[0] if head_dim is None else head_dim
        # --- Head ---
        self.head = PointMLP(3, head_dim)

        # --- Body ---
        self.encoders = nn.ModuleList()
        in_ch = head_dim
        for out_ch in dims:
            self.encoders.append(
                TransformerEncoder(
                    in_ch, out_ch, psi=psi, mlp_ratio=mlp_ratio,
                    use_geo=use_geo, use_feat=use_feat,
                    attention_type=attention_type, scale_qk=scale_qk,
                )
            )
            in_ch = out_ch

        # --- Tail ---
        if tail_mode == "shuffle":
            # Shuffle: (N, C) -> (rN, C/r)，然后 MLP -> 3
            self.tail = nn.Sequential(
                PointMLP(dims[-1] // up_ratio, 64, bn=False),
                nn.Linear(64, 3),
            )
        else:
            self.tail = nn.Sequential(
                PointMLP(dims[-1], 64, bn=False),
                nn.Linear(64, 3 * up_ratio),
            )

    @staticmethod
    def _pixel_shuffle_1d(x: torch.Tensor, r: int) -> torch.Tensor:
        """点云版周期性重排：(B, N, C) -> (B, r*N, C/r)。

        对应论文的 Shuffle 操作（Shi et al. 2016 PixelShuffle 的 1D 类比），
        不引入任何参数。
        """
        b, n, c = x.shape
        if c % r != 0:
            raise ValueError(f"通道 {c} 不能被 r={r} 整除")
        return x.reshape(b, n, r, c // r).reshape(b, n * r, c // r)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : (B, N, 3) 稀疏输入点云

        Returns
        -------
        (B, r*N, 3) 密集输出点云
        """
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"期望 (B,N,3)，收到 {tuple(xyz.shape)}")

        # knn 只算一次，全程复用（spec §2.2）
        knn_idx = knn_indices(xyz, self.k)

        feat = self.head(xyz)
        for enc in self.encoders:
            feat = enc(xyz, feat, knn_idx)

        if self.tail_mode == "shuffle":
            dense = self._pixel_shuffle_1d(feat, self.up_ratio)   # (B, rN, C/r)
            return self.tail(dense)                                # (B, rN, 3)
        else:
            out = self.tail(feat)                                  # (B, N, 3r)
            b, n, _ = out.shape
            return out.reshape(b, n * self.up_ratio, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 自检
# =============================================================================

def self_check() -> bool:
    """结构自检 + 参数量对齐论文 969.9k。"""
    ok = True
    torch.manual_seed(0)

    print("=" * 68)
    print("models/pu_transformer.py 自检")
    print("=" * 68)

    b, n = 2, 64

    # --- 1. 默认配置前向 ---
    net = PUTransformer(up_ratio=4)
    x = torch.randn(b, n, 3)
    y = net(x)
    p1 = y.shape == (b, n * 4, 3)
    ok &= p1
    print(f"[{'PASS' if p1 else 'FAIL'}] 4x 前向: {tuple(x.shape)} -> {tuple(y.shape)} "
          f"(期望 {(b, n*4, 3)})")

    # --- 2. 参数量对齐论文 ---
    n_param = net.count_parameters()
    target = 969_900
    ratio = n_param / target
    p2 = 0.5 < ratio < 2.0   # 宽松门槛，先看量级
    ok &= p2
    print(f"[{'PASS' if p2 else 'FAIL'}] 参数量 {n_param:,} vs 论文 969,900 "
          f"(比例 {ratio:.3f})")

    # --- 3. SC-MSA 窗口铺满校验 ---
    try:
        for dim in (32, 64, 128, 256):
            m = SCMSA(dim, psi=4)
            last = (m.n_heads - 1) * m.d + m.w
            assert last == dim, f"dim={dim} 未铺满: {last}"
            assert m.n_heads == 7, f"dim={dim} head 数应为 7，实为 {m.n_heads}"
        print("[PASS] SC-MSA 窗口铺满且 M=2*psi-1=7（dim=32/64/128/256 全过）")
    except AssertionError as e:
        ok = False
        print(f"[FAIL] SC-MSA 窗口校验: {e}")

    # --- 4. 消融开关可用 ---
    variants = {
        "A1 无PosFus上下文(仅geo)": dict(use_geo=True, use_feat=False),
        "A3 仅feat": dict(use_geo=False, use_feat=True),
        "B3 常规MSA": dict(attention_type="msa"),
        "C1 MLP tail": dict(tail_mode="mlp"),
    }
    for name, kw in variants.items():
        try:
            m = PUTransformer(up_ratio=4, **kw)
            out = m(x)
            assert out.shape == (b, n * 4, 3), f"输出形状 {tuple(out.shape)}"
            print(f"[PASS] 消融 {name}: 参数 {m.count_parameters():,}, "
                  f"输出 {tuple(out.shape)}")
        except Exception as e:
            ok = False
            print(f"[FAIL] 消融 {name}: {type(e).__name__}: {e}")

    # --- 5. 不同倍率 ---
    for r in (4, 8, 16):
        try:
            m = PUTransformer(up_ratio=r)
            out = m(x)
            assert out.shape == (b, n * r, 3)
            print(f"[PASS] r={r}: 输出 {tuple(out.shape)}, 参数 {m.count_parameters():,}")
        except Exception as e:
            ok = False
            print(f"[FAIL] r={r}: {type(e).__name__}: {e}")

    # --- 6. 梯度回传 ---
    net.zero_grad()
    net(x).sum().backward()
    n_grad = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in net.parameters())
    p6 = n_grad > n_total * 0.9
    ok &= p6
    print(f"[{'PASS' if p6 else 'FAIL'}] 梯度回传: {n_grad}/{n_total} 参数收到非零梯度")

    # --- 7. knn 索引复用验证（性能相关） ---
    idx = knn_indices(x, 20)
    p7 = idx.shape == (b, n, 20) and (idx[:, :, 0] == torch.arange(n)).all()
    ok &= p7
    print(f"[{'PASS' if p7 else 'FAIL'}] knn 形状 {tuple(idx.shape)}，"
          f"第一近邻为自身: {bool((idx[:, :, 0] == torch.arange(n)).all())}")

    print("-" * 68)
    print(f"总体: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
