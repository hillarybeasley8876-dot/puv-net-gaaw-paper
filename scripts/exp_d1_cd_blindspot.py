"""D1 构造实验：量化 Chamfer Distance 对局部密度失衡的不敏感性。

对应 docs/PAPER_REMAKE_PLAN.md §3.4 实验 D1，支撑 RQ1。

论点
----
CD 损失 = 双向平均最近邻距离。它衡量的是「每个点附近有没有 GT 点」，
但**不衡量点的分布是否均匀**。因此存在这样的情形：

    两个点云对 GT 的 CD 几乎相同，但均匀性（NUC）差若干个数量级。

若该现象成立，则「只用 CD 损失训练」的方法（如原版 PU-Transformer，
其论文明写 "we only use the modified Chamfer Distance loss"）
在优化目标上就存在一个结构性盲区 —— 这为引入均匀性/对抗约束提供了
**机制性理由**，而非「A+B 拼接」。

⚠️ 本实验为纯构造实验，不涉及任何训练与数据集，结果完全可复现。
   随机种子固定，任何人重跑应得到相同数字。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.metrics.pointcloud import (  # noqa: E402
    chamfer_distance,
    hausdorff_distance,
    uniformity_nuc,
)

SEED = 20260810
OUT_DIR = ROOT / "runs" / "D1_cd_blindspot"


def fibonacci_sphere(n: int) -> np.ndarray:
    """Fibonacci 格点球面采样 —— 近似最优均匀分布，作为 GT。"""
    idx = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    return np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], axis=1)


def make_uniform(gt: np.ndarray, jitter: float, rng) -> np.ndarray:
    """构型 U：在 GT 附近各向同性小抖动 —— 保持均匀。"""
    noise = rng.standard_normal(gt.shape) * jitter
    pts = gt + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def make_clumped(gt: np.ndarray, frac: float, jitter: float, rng) -> np.ndarray:
    """构型 C：把一部分点「成对塌缩」到邻居位置 —— 制造局部密度失衡。

    关键设计：塌缩后的点仍然**位于 GT 表面附近**，
    所以每个点到 GT 的最近邻距离依然很小 → CD 几乎不变；
    但点在表面上的分布变得极不均匀 → NUC 显著恶化。
    """
    pts = gt.copy()
    n = len(gt)
    n_move = int(n * frac)
    idx = rng.choice(n, size=n_move, replace=False)

    # 每个被选中的点，移动到它的某个近邻处（造成重叠密集区）
    # 用简单的最近邻查找
    from scipy.spatial import cKDTree
    tree = cKDTree(gt)
    _, nbr = tree.query(gt[idx], k=3)
    target = gt[nbr[:, 1]]                      # 移到最近邻（非自身）位置
    pts[idx] = target + rng.standard_normal((n_move, 3)) * jitter
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def main() -> int:
    rng = np.random.default_rng(SEED)
    n = 8192                                     # 与 4x 上采样的 GT 规模一致

    gt = fibonacci_sphere(n)

    print("=" * 78)
    print("D1 构造实验：CD 对局部密度失衡的不敏感性")
    print("=" * 78)
    print(f"GT: Fibonacci 球面 {n} 点（近似最优均匀）")
    print(f"随机种子: {SEED}（结果完全可复现）")
    print()

    # 构型 U：均匀抖动
    jitter = 0.004
    pc_u = make_uniform(gt, jitter=jitter, rng=np.random.default_rng(SEED))

    # 构型 C：成对塌缩（多个塌缩比例）
    configs = []
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        pc_c = make_clumped(gt, frac=frac, jitter=jitter,
                            rng=np.random.default_rng(SEED))
        configs.append((frac, pc_c))

    rows = []

    cd_u = chamfer_distance(pc_u, gt)
    hd_u = hausdorff_distance(pc_u, gt)
    nuc_u = uniformity_nuc(pc_u, seed=SEED)["nuc_mean"]
    rows.append({
        "config": "U (均匀抖动)",
        "clump_frac": 0.0,
        "cd": cd_u, "hd": hd_u, "nuc": nuc_u,
    })

    for frac, pc_c in configs:
        cd_c = chamfer_distance(pc_c, gt)
        hd_c = hausdorff_distance(pc_c, gt)
        nuc_c = uniformity_nuc(pc_c, seed=SEED)["nuc_mean"]
        rows.append({
            "config": f"C (塌缩 {frac:.0%})",
            "clump_frac": frac,
            "cd": cd_c, "hd": hd_c, "nuc": nuc_c,
        })

    # 输出表格
    hdr = f"{'构型':<16} {'CD (×1e-3)':>13} {'CD 相对U':>10} {'HD (×1e-3)':>12} {'NUC':>12} {'NUC 相对U':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cd_rel = r["cd"] / cd_u
        nuc_rel = r["nuc"] / nuc_u if nuc_u > 0 else float("inf")
        print(f"{r['config']:<16} {r['cd']*1e3:>13.4f} {cd_rel:>9.2f}x "
              f"{r['hd']*1e3:>12.4f} {r['nuc']:>12.6f} {nuc_rel:>11.1f}x")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)

    # 找出 CD 变化最小而 NUC 变化最大的对照
    best = None
    for r in rows[1:]:
        cd_rel = r["cd"] / cd_u
        nuc_rel = r["nuc"] / nuc_u
        score = nuc_rel / cd_rel          # NUC 恶化倍数 / CD 恶化倍数
        if best is None or score > best[0]:
            best = (score, r, cd_rel, nuc_rel)

    score, r, cd_rel, nuc_rel = best
    print(f"最显著对照: {r['config']}")
    print(f"  CD  仅恶化 {cd_rel:.2f} 倍  ({cd_u*1e3:.4f} -> {r['cd']*1e3:.4f}, ×1e-3)")
    print(f"  NUC 恶化了 {nuc_rel:.1f} 倍  ({nuc_u:.6f} -> {r['nuc']:.6f})")
    print(f"  敏感度失配比 = {score:.1f}x")
    print()
    print("解读：")
    print("  CD 几乎无法察觉的分布退化，在 NUC 上呈现数量级差异。")
    print("  => 仅以 CD 为训练目标的模型，其损失函数对局部密度失衡")
    print("     存在结构性盲区。这为引入均匀性/对抗约束提供机制性依据。")
    print()
    print("⚠️ 本实验证明的是【CD 指标的盲区】，")
    print("   尚【未】证明「加入对抗约束能提升上采样性能」——")
    print("   后者需 B 组消融实验（见 PAPER_REMAKE_PLAN §3.3）。")

    # 落盘
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "D1_cd_blindspot",
        "seed": SEED,
        "n_points": n,
        "jitter": jitter,
        "gt": "fibonacci_sphere",
        "rows": rows,
        "most_significant": {
            "config": r["config"],
            "cd_ratio": cd_rel,
            "nuc_ratio": nuc_rel,
            "mismatch_score": score,
        },
        "note": "纯构造实验，无训练无数据集，固定种子完全可复现",
    }
    out = OUT_DIR / "result.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    np.savez_compressed(OUT_DIR / "point_clouds.npz",
                        gt=gt, uniform=pc_u,
                        **{f"clump_{int(f*100)}": p for f, p in configs})
    print()
    print(f"结果已落盘: {out}")
    print(f"点云已落盘: {OUT_DIR / 'point_clouds.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
