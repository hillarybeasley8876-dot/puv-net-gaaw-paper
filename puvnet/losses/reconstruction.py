"""可微损失函数 —— 训练用（求快），与 metrics/ 的 numpy 参考实现（求准）分离。

分离的理由：训练时 CD 要在 GPU 上跑几万次，必须用批量矩阵实现；
最终报告的指标必须用 KD-tree 精确实现。两者在单测中交叉验证，
防止"训练 loss 降了但真实指标没降"这类静默失效。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor,
                 squared: bool = True) -> torch.Tensor:
    """批量双向 Chamfer。pred (B,3,N), gt (B,3,M)。

    squared=True 用平方距离，梯度更平滑，训练更稳；
    评测时用 metrics.chamfer_distance 的非平方版本报告。
    """
    p = pred.transpose(1, 2)                     # (B,N,3)
    g = gt.transpose(1, 2)                       # (B,M,3)
    d = torch.cdist(p, g)                        # (B,N,M)
    if squared:
        d = d ** 2
    return d.min(dim=2).values.mean(dim=1) .mean() + \
           d.min(dim=1).values.mean(dim=1).mean()


def repulsion_loss(pred: torch.Tensor, k: int = 8,
                   radius: float = 0.05) -> torch.Tensor:
    """排斥损失（PU-Net/PU-GAN 标准做法）—— 防止生成点堆叠。

    只惩罚距离小于 radius 的邻居对，用衰减权重避免过度推开。
    这是保证 Uniformity 指标的关键项；缺了它上采样点会成簇。
    """
    p = pred.transpose(1, 2)                     # (B,N,3)
    d = torch.cdist(p, p)
    n = p.shape[1]
    d = d + torch.eye(n, device=p.device).unsqueeze(0) * 1e10
    knn = d.topk(k, dim=-1, largest=False).values      # (B,N,k)
    # 只对过近的点施加排斥
    w = torch.exp(-(knn ** 2) / (radius ** 2))
    return (F.relu(radius - knn) * w).mean()


def interior_containment_loss(interior: torch.Tensor,
                              gt_interior: torch.Tensor) -> torch.Tensor:
    """内部点分布损失：让预测内部点匹配真值内部点分布。

    直接用 CD 而非逐点回归，因为内部点无天然对应关系（无序集合）。
    """
    return chamfer_loss(interior, gt_interior)


def total_loss(out: dict, batch: dict, w: dict) -> tuple[torch.Tensor, dict]:
    """组合损失。返回 (loss, 各项标量字典) 便于 TensorBoard 逐项监控。

    逐项记录是必须的：只看总 loss 无法发现"内部项不降但表面项在降"
    这类问题，而这恰好是 H6 是否真的生效的关键证据。
    """
    logs = {}
    loss = torch.zeros((), device=out["surface"].device)

    l_cd = chamfer_loss(out["surface"], batch["surface_gt"])
    loss = loss + w.get("cd", 1.0) * l_cd
    logs["cd"] = l_cd.item()

    if w.get("repulsion", 0) > 0:
        l_rep = repulsion_loss(out["surface"])
        loss = loss + w["repulsion"] * l_rep
        logs["repulsion"] = l_rep.item()

    if out.get("interior") is not None and "interior_gt" in batch:
        l_in = interior_containment_loss(out["interior"], batch["interior_gt"])
        loss = loss + w.get("interior", 1.0) * l_in
        logs["interior"] = l_in.item()

    logs["total"] = loss.item()
    return loss, logs


def self_check(verbose: bool = True) -> bool:
    """与 numpy 参考实现交叉验证，并检查损失的基本性质。"""
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from puvnet.metrics.pointcloud import chamfer_distance

    ok = True

    def log(*a):
        if verbose:
            print(*a)

    torch.manual_seed(0)
    p = torch.randn(1, 3, 200)
    g = torch.randn(1, 3, 300)

    # --- 与 numpy 参考实现对齐（非平方形式）---
    mine = chamfer_loss(p, g, squared=False).item()
    ref = chamfer_distance(p[0].T.numpy().astype(np.float64),
                           g[0].T.numpy().astype(np.float64), squared=False)
    log(f"[cd] torch={mine:.6f} numpy={ref:.6f} diff={abs(mine-ref):.2e}")
    if abs(mine - ref) > 1e-4:
        log("  !! 可微实现与参考实现不一致"); ok = False

    # --- 相同点集 CD 应为 0（float32 下 cdist 有 ~1e-7 量级数值误差）---
    z = chamfer_loss(p, p).item()
    log(f"[cd self] {z:.3e} (应 ≈0，float32 容差 1e-5)")
    if z > 1e-5:
        log("  !! 自身 CD 显著不为 0"); ok = False

    # --- 排斥损失：同一尺度下，堆叠点应显著大于均匀点 ---
    # 两者都归一到单位立方体量级，否则"散开"样本邻域内无点，
    # 得到 0 只是因为尺度不可比，验证不到方向性。
    n_side = 5
    lin = torch.linspace(-0.5, 0.5, n_side)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), 0)
    uniform = grid.reshape(1, 3, -1)                       # 125 点规则栅格
    clumped = torch.randn(1, 3, n_side ** 3) * 0.01        # 同点数但成簇
    r_c = repulsion_loss(clumped).item()
    r_u = repulsion_loss(uniform).item()
    log(f"[repulsion] clumped={r_c:.6f} > uniform={r_u:.6f} ?")
    if not (r_c > r_u):
        log("  !! 排斥损失方向错误"); ok = False

    # --- 梯度可回传 ---
    q = torch.randn(1, 3, 50, requires_grad=True)
    chamfer_loss(q, g).backward()
    log(f"[grad] norm={q.grad.norm().item():.4f} (应 > 0)")
    if q.grad.norm().item() <= 0:
        log("  !! 无梯度"); ok = False

    log(f"\nlosses self_check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
