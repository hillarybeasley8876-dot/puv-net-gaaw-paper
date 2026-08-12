"""
PUV-Net 网络骨架 —— 体积点云上采样。

设计原则
--------
每个待验证假设都做成**独立可开关的模块**，消融实验只改配置不改代码。
这样每个消融数字都能追溯到具体的开关组合，审稿人可以逐条核验。

对应论文假设：
  H2  attention        Transformer 注意力 vs 纯 MLP 特征提取
  H3a positional_enc   正余弦位置编码
  H3b use_normals      法线特征融合
  H5  分离训练          由 train.py 的 stage 控制，不在模型内
  H6  volumetric_head  体积感知的内部点生成头

关于「升维」的实现选择
----------------------
原稿描述的是「通道扩展 → 重排列」（PU-Net 的 shuffle 路线）。
本实现同时提供两条路径：
  - shuffle:  特征 C → r*C 后 reshape，参数少但输出点易共线
  - offset:   为每个输入点预测 r 个坐标偏移，几何可解释性更强
这本身就是一组值得报告的对比（原稿没做）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class SharedMLP(nn.Sequential):
    """逐点共享 MLP，作用在 (B, C, N) 上。"""

    def __init__(self, dims: list[int], bn: bool = True, last_act: bool = True):
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], 1))
            is_last = (i == len(dims) - 2)
            if not is_last or last_act:
                if bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
        super().__init__(*layers)


class PositionalEncoding(nn.Module):
    """正余弦位置编码（H3a）。

    对每个坐标分量施加 n_freq 个频率的 sin/cos，输出维度 3 + 3*2*n_freq。
    include_input=True 保留原始坐标，避免高频编码淹没低频几何信息。
    """

    def __init__(self, n_freq: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freq = n_freq
        self.include_input = include_input
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(n_freq).float() * torch.pi)

    @property
    def out_dim(self) -> int:
        return (3 if self.include_input else 0) + 3 * 2 * self.n_freq

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, 3, N)
        out = [xyz] if self.include_input else []
        for f in self.freqs:
            out.append(torch.sin(xyz * f))
            out.append(torch.cos(xyz * f))
        return torch.cat(out, dim=1)


def knn_graph(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """返回每点的 k 近邻索引 (B, N, k)，不含自身。

    xyz: (B, 3, N)。用矩阵形式算平方距离，N 在几千量级时显存可接受。
    """
    b, _, n = xyz.shape
    x = xyz.transpose(1, 2)                              # (B, N, 3)
    d = torch.cdist(x, x)                                # (B, N, N)
    d = d + torch.eye(n, device=xyz.device).unsqueeze(0) * 1e10
    return d.topk(k, dim=-1, largest=False).indices


def gather_neighbors(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引取邻居特征。feat (B,C,N), idx (B,N,k) -> (B,C,N,k)"""
    b, c, n = feat.shape
    k = idx.shape[-1]
    flat = idx.reshape(b, -1)                            # (B, N*k)
    g = torch.gather(feat, 2, flat.unsqueeze(1).expand(b, c, n * k))
    return g.reshape(b, c, n, k)


class LocalGeometricBlock(nn.Module):
    """局部几何特征聚合（EdgeConv 风格），作为 H2 的对照基线。

    用相对坐标差 (neighbor - center) 编码局部几何，这是点云网络的
    标准做法，也是「无注意力」对照组的主体。
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = SharedMLP([in_dim * 2, out_dim, out_dim])

    def forward(self, feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        nb = gather_neighbors(feat, idx)                 # (B,C,N,k)
        center = feat.unsqueeze(-1).expand_as(nb)
        edge = torch.cat([center, nb - center], dim=1)   # (B,2C,N,k)
        b, c2, n, k = edge.shape
        out = self.mlp(edge.permute(0, 1, 3, 2).reshape(b, c2, k * n))
        out = out.reshape(b, -1, k, n).permute(0, 1, 3, 2)
        return out.max(dim=-1).values                    # (B,out,N)


class PointTransformerBlock(nn.Module):
    """向量注意力块（H2）。

    采用 Point Transformer 的向量注意力而非标准 scalar dot-product：
    注意力权重是逐通道的向量，且显式融入相对位置编码 delta。
    对点云而言这比 scalar attention 更契合几何结构。

    残差连接保证 attention 关掉时退化行为可控。
    """

    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.to_q = nn.Conv1d(dim, dim, 1)
        self.to_k = nn.Conv1d(dim, dim, 1)
        self.to_v = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, feat: torch.Tensor, xyz: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
        q = self.to_q(feat)
        k = gather_neighbors(self.to_k(feat), idx)       # (B,C,N,k)
        v = gather_neighbors(self.to_v(feat), idx)

        nb_xyz = gather_neighbors(xyz, idx)
        delta = self.pos_mlp(nb_xyz - xyz.unsqueeze(-1))  # (B,C,N,k)

        attn = self.attn_mlp(q.unsqueeze(-1) - k + delta)
        attn = torch.softmax(attn, dim=-1)
        out = (attn * (v + delta)).sum(dim=-1)           # (B,C,N)
        return F.gelu(self.norm(out) + feat)


# ---------------------------------------------------------------------------
# 主干
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """多尺度特征提取。attention 开关即 H2 的消融点。"""

    def __init__(self, in_dim: int, dim: int = 128, n_blocks: int = 3,
                 k: int = 16, attention: bool = True):
        super().__init__()
        self.k = k
        self.attention = attention
        self.stem = SharedMLP([in_dim, dim, dim])
        self.local = nn.ModuleList(
            [LocalGeometricBlock(dim, dim, k) for _ in range(n_blocks)])
        self.attn = nn.ModuleList(
            [PointTransformerBlock(dim, k) for _ in range(n_blocks)]
            if attention else [])
        # 拼接各层输出后压回 dim，保持多尺度信息
        self.fuse = SharedMLP([dim * (n_blocks + 1), dim * 2, dim])

    def forward(self, feat: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        idx = knn_graph(xyz, self.k)
        h = self.stem(feat)
        scales = [h]
        for i, blk in enumerate(self.local):
            h = blk(h, idx)
            if self.attention:
                h = self.attn[i](h, xyz, idx)
            scales.append(h)
        return self.fuse(torch.cat(scales, dim=1))


class UpsampleHead(nn.Module):
    """表面上采样头：把 N 点扩到 r*N 点。

    mode='offset'  为每个点预测 r 个偏移，几何可解释
    mode='shuffle' 通道扩展后 reshape（PU-Net 路线）
    """

    def __init__(self, dim: int, ratio: int = 4, mode: str = "offset"):
        super().__init__()
        assert mode in ("offset", "shuffle")
        self.ratio = ratio
        self.mode = mode
        if mode == "offset":
            self.head = SharedMLP([dim, dim, 3 * ratio], last_act=False)
        else:
            self.expand = SharedMLP([dim, dim * ratio])
            self.head = SharedMLP([dim, dim // 2, 3], last_act=False)

    def forward(self, feat: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        b, _, n = xyz.shape
        r = self.ratio
        if self.mode == "offset":
            off = self.head(feat).reshape(b, 3, r, n)
            return (xyz.unsqueeze(2) + off).reshape(b, 3, r * n)
        f = self.expand(feat).reshape(b, -1, r, n).reshape(b, -1, r * n)
        base = xyz.unsqueeze(2).expand(b, 3, r, n).reshape(b, 3, r * n)
        return base + self.head(f)


class VolumetricHead(nn.Module):
    """体积内部点生成头（H6）—— 本文的核心主张所在。

    关键设计：内部点不是「表面点的偏移」，而是由表面特征聚合出的
    全局形状码 + 可学习的内部查询向量共同解码得到。

    为什么这样设计：内部点在几何上不属于任何单个表面点的邻域，
    如果用表面点偏移的方式生成，网络只能学出「向内挤压」的浅层规律，
    无法表达空腔、管道这类真正的内部拓扑。全局形状码 + 查询机制
    才能让不同查询关注不同的内部区域。

    depth_scale 限制内部点的初始分布范围，避免训练初期大量点跑到网格外，
    否则 interior_ratio 一开始就是 0，梯度信号很弱。
    """

    def __init__(self, dim: int, n_queries: int = 4096,
                 depth_scale: float = 0.5):
        super().__init__()
        self.n_queries = n_queries
        self.depth_scale = depth_scale
        self.queries = nn.Parameter(torch.randn(1, dim, n_queries) * 0.02)
        self.global_mlp = SharedMLP([dim, dim * 2, dim * 2])
        # 输入 = 查询向量 dim + 全局码 (max|mean 各 2*dim) = dim + 4*dim
        self.decode = SharedMLP([dim * 5, dim * 2, dim, 3], last_act=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        g = self.global_mlp(feat)                        # (B, 2C, N)
        # max + mean 双池化：max 抓显著结构，mean 抓整体分布，二者互补。
        # 注意 max(dim) 返回 namedtuple 需取 .values，mean(dim) 直接返回 tensor。
        g = torch.cat([g.max(dim=2).values, g.mean(dim=2)], dim=1)  # (B, 4C)
        g = g.unsqueeze(-1).expand(-1, -1, self.n_queries)          # (B, 4C, Q)
        q = self.queries.expand(b, -1, -1)                          # (B, C, Q)
        return torch.tanh(self.decode(torch.cat([q, g], dim=1))) * self.depth_scale


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------

class PUVNet(nn.Module):
    """体积点云上采样网络。

    forward 返回 dict，便于分阶段训练（H5）时只取需要的分支：
      surface  (B, 3, r*N)  表面上采样结果
      interior (B, 3, Q)    内部点，volumetric=False 时为 None
    """

    def __init__(self,
                 ratio: int = 4,
                 dim: int = 128,
                 n_blocks: int = 3,
                 k: int = 16,
                 attention: bool = True,          # H2
                 positional_enc: bool = True,     # H3a
                 n_freq: int = 6,
                 use_normals: bool = False,       # H3b
                 volumetric: bool = True,         # H6
                 n_interior: int = 4096,
                 upsample_mode: str = "offset"):
        super().__init__()
        self.cfg = dict(ratio=ratio, dim=dim, n_blocks=n_blocks, k=k,
                        attention=attention, positional_enc=positional_enc,
                        n_freq=n_freq, use_normals=use_normals,
                        volumetric=volumetric, n_interior=n_interior,
                        upsample_mode=upsample_mode)

        self.pe = PositionalEncoding(n_freq) if positional_enc else None
        in_dim = self.pe.out_dim if positional_enc else 3
        if use_normals:
            in_dim += 3
        self.use_normals = use_normals

        self.encoder = FeatureExtractor(in_dim, dim, n_blocks, k, attention)
        self.up_head = UpsampleHead(dim, ratio, upsample_mode)
        self.vol_head = VolumetricHead(dim, n_interior) if volumetric else None

    def forward(self, xyz: torch.Tensor,
                normals: torch.Tensor | None = None) -> dict:
        """xyz: (B, 3, N)，normals: (B, 3, N) 或 None"""
        feat_in = self.pe(xyz) if self.pe is not None else xyz
        if self.use_normals:
            if normals is None:
                raise ValueError("use_normals=True 但未提供 normals")
            feat_in = torch.cat([feat_in, normals], dim=1)

        feat = self.encoder(feat_in, xyz)
        return {
            "surface": self.up_head(feat, xyz),
            "interior": self.vol_head(feat) if self.vol_head else None,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def self_check(verbose: bool = True) -> bool:
    """验证所有开关组合都能前反向跑通、形状正确、梯度非零。

    这一步能挡掉绝大多数"配置能写但跑不起来"的问题。
    """
    ok = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def log(*a):
        if verbose:
            print(*a)

    log(f"device = {dev}")
    B, N, R, Q = 2, 256, 4, 512

    combos = [
        dict(attention=True,  positional_enc=True,  use_normals=False, volumetric=True),
        dict(attention=False, positional_enc=True,  use_normals=False, volumetric=True),
        dict(attention=True,  positional_enc=False, use_normals=False, volumetric=True),
        dict(attention=True,  positional_enc=True,  use_normals=True,  volumetric=True),
        dict(attention=True,  positional_enc=True,  use_normals=False, volumetric=False),
        dict(attention=True,  positional_enc=True,  use_normals=False, volumetric=True,
             upsample_mode="shuffle"),
    ]

    for i, c in enumerate(combos):
        try:
            net = PUVNet(ratio=R, dim=64, n_blocks=2, k=8,
                         n_interior=Q, **c).to(dev)
            xyz = torch.randn(B, 3, N, device=dev) * 0.3
            nrm = F.normalize(torch.randn(B, 3, N, device=dev), dim=1)
            out = net(xyz, nrm if c.get("use_normals") else None)

            assert out["surface"].shape == (B, 3, R * N), \
                f"surface shape {out['surface'].shape}"
            if c["volumetric"]:
                assert out["interior"].shape == (B, 3, Q), \
                    f"interior shape {out['interior'].shape}"
            else:
                assert out["interior"] is None

            loss = out["surface"].square().mean()
            if out["interior"] is not None:
                loss = loss + out["interior"].square().mean()
            loss.backward()

            gnorm = sum(p.grad.norm().item() for p in net.parameters()
                        if p.grad is not None)
            assert gnorm > 0, "梯度全为零"

            tag = ",".join(f"{k}={v}" for k, v in c.items())
            log(f"  [{i}] OK  params={net.n_params()/1e6:.2f}M "
                f"gnorm={gnorm:.2f}  ({tag})")
        except Exception as e:
            log(f"  [{i}] FAIL  {type(e).__name__}: {e}")
            ok = False

    log(f"\nmodel self_check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
