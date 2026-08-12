"""D1v2 构造实验：CD 对局部密度失衡是否敏感（对齐误差预算版）。

对应 docs/PAPER_REMAKE_PLAN.md §3.4 实验 D1，支撑（或否证）RQ1。

v1 为什么作废
-------------
v1 (`exp_d1_cd_blindspot.py`) 有设计 bug：
  - make_uniform  扰动【全部】点  -> 每点都带 jitter 误差，CD 地板 ~10e-3
  - make_clumped  只扰动 frac 比例的点，其余点【精确压在 GT 上】-> 对 CD 贡献 ~0
两个构型误差预算不对等，实际测的是「扰动了多少比例的点」，不是「CD 能否看见密度失衡」。
后果：塌缩 10% 的 CD (3.85e-3) 反而比均匀 (10.03e-3) 好 2.6 倍，对照组逻辑上不成立。
v1 结果全部作废，不得进入论文。

v2 的修法
---------
两个构型严格对齐：
  - 扰动【相同数量】的点（同一组 idx，同一随机种子）
  - 位移【相同幅度】 step（同一组 |d| 采样）
  - 唯一差别是位移【方向】：
      U (均匀/散开)：沿切平面随机方向散开 -> 点仍在表面附近，分布保持均匀
      C (塌缩)      ：沿指向最近邻的方向移动 step -> 点仍在表面附近，但产生局部聚集
两者到 GT 表面的距离量级相同 => CD 被构造性地held住；
差别只在表面上的分布 => NUC 是唯一自由变量。

预注册判据（跑之前定死，不得事后修改）
--------------------------------------
对至少一个 step 档位，同时满足：
  (P1) CD 比值 |cd_C / cd_U - 1| <= 0.10      即 CD 几乎看不出差别
  (P2) NUC 比值 nuc_C / nuc_U   >= 3.0        即均匀性显著恶化
=> PASS：D1 支持 RQ1（CD 对局部密度失衡不敏感）
=> FAIL：D1 不支持 RQ1，论文必须删掉「CD 盲区」的机制性叙事，
         改为仅以 B 组消融的经验结果论述，或放弃该创新点定位。

诚信约束
--------
- 本脚本不打印任何超出判据的解释性结论。
- 判据不通过就如实输出 FAIL，不调参重试凑结果。
- NUC 为简化实现，仅用于同一实验内的相对比较，不与文献绝对值对表。
- 固定种子，任何人重跑应得到相同数字。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.metrics.pointcloud import (  # noqa: E402
    chamfer_distance,
    hausdorff_distance,
    uniformity_nuc,
)

SEED = 20260810
OUT_DIR = ROOT / "runs" / "D1v2_cd_blindspot"

# ---- 预注册判据（跑前定死） ----
CD_TOL = 0.10        # (P1) CD 相对差异容许上限
NUC_MIN_RATIO = 3.0  # (P2) NUC 恶化倍数下限

N_POINTS = 8192      # 与 4x 上采样 GT 规模一致 (2048 x 4)
PERTURB_FRAC = 0.50  # 两构型都扰动这个比例的点
STEPS = (0.005, 0.010, 0.015, 0.020, 0.030)  # 位移幅度档位（球半径=1）


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


def _tangent_dirs(pts: np.ndarray, rng) -> np.ndarray:
    """在每个点的切平面内取随机单位方向（球面上 normal = 点本身）。"""
    normal = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    rand = rng.standard_normal(pts.shape)
    # 去掉法向分量 -> 落在切平面
    tang = rand - (np.sum(rand * normal, axis=1, keepdims=True)) * normal
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    return tang / norm


def _neighbor_dirs(gt: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """每个被选中点 -> 指向其最近邻（非自身）的单位方向。"""
    tree = cKDTree(gt)
    _, nbr = tree.query(gt[idx], k=2)
    target = gt[nbr[:, 1]]
    d = target - gt[idx]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    return d / norm


def build_pair(gt: np.ndarray, step: float, frac: float, seed: int):
    """构造误差预算严格对齐的一对点云 (U, C)。

    共享：同一组 idx、同一 step。差别仅在位移方向。
    """
    rng = np.random.default_rng(seed)
    n = len(gt)
    n_move = int(n * frac)
    idx = rng.choice(n, size=n_move, replace=False)

    # 方向 A：切平面随机方向（散开，保持均匀）
    dir_u = _tangent_dirs(gt[idx], rng)
    # 方向 B：指向最近邻（塌缩，制造聚集）
    dir_c = _neighbor_dirs(gt, idx)

    pc_u = gt.copy()
    pc_u[idx] = gt[idx] + dir_u * step
    pc_c = gt.copy()
    pc_c[idx] = gt[idx] + dir_c * step

    # 都投回单位球面，保证「仍在表面附近」这一前提对两者同等成立
    pc_u = pc_u / np.linalg.norm(pc_u, axis=1, keepdims=True)
    pc_c = pc_c / np.linalg.norm(pc_c, axis=1, keepdims=True)
    return pc_u, pc_c, idx


def main() -> int:
    gt = fibonacci_sphere(N_POINTS)

    print("=" * 84)
    print("D1v2 构造实验：CD 对局部密度失衡是否敏感（对齐误差预算）")
    print("=" * 84)
    print(f"GT           : Fibonacci 球面 {N_POINTS} 点（近似最优均匀）")
    print(f"随机种子     : {SEED}（完全可复现）")
    print(f"扰动比例     : {PERTURB_FRAC:.0%}（两构型相同）")
    print("位移幅度     : 两构型相同，仅方向不同（U=切平面散开 / C=指向最近邻塌缩）")
    print(f"预注册判据   : |CD比值-1| <= {CD_TOL:.2f}  且  NUC比值 >= {NUC_MIN_RATIO:.1f}")
    print()

    rows = []
    hdr = (f"{'step':>7} | {'CD_U':>9} {'CD_C':>9} {'CD比值':>8} | "
           f"{'NUC_U':>10} {'NUC_C':>10} {'NUC比值':>8} | {'P1':>4} {'P2':>4}")
    print(hdr)
    print("-" * len(hdr))

    any_pass = False
    for step in STEPS:
        pc_u, pc_c, _ = build_pair(gt, step=step, frac=PERTURB_FRAC, seed=SEED)

        cd_u = chamfer_distance(pc_u, gt)
        cd_c = chamfer_distance(pc_c, gt)
        hd_u = hausdorff_distance(pc_u, gt)
        hd_c = hausdorff_distance(pc_c, gt)
        nuc_u = uniformity_nuc(pc_u, seed=SEED)["nuc_mean"]
        nuc_c = uniformity_nuc(pc_c, seed=SEED)["nuc_mean"]

        cd_ratio = cd_c / cd_u if cd_u > 0 else float("inf")
        nuc_ratio = nuc_c / nuc_u if nuc_u > 0 else float("inf")

        p1 = abs(cd_ratio - 1.0) <= CD_TOL
        p2 = nuc_ratio >= NUC_MIN_RATIO
        if p1 and p2:
            any_pass = True

        print(f"{step:>7.3f} | {cd_u*1e3:>9.4f} {cd_c*1e3:>9.4f} {cd_ratio:>7.3f}x | "
              f"{nuc_u:>10.6f} {nuc_c:>10.6f} {nuc_ratio:>7.2f}x | "
              f"{'PASS' if p1 else 'fail':>4} {'PASS' if p2 else 'fail':>4}")

        rows.append({
            "step": step,
            "cd_u": cd_u, "cd_c": cd_c, "cd_ratio": cd_ratio,
            "hd_u": hd_u, "hd_c": hd_c,
            "nuc_u": nuc_u, "nuc_c": nuc_c, "nuc_ratio": nuc_ratio,
            "p1_cd_within_tol": bool(p1),
            "p2_nuc_ratio_ok": bool(p2),
            "verdict": "PASS" if (p1 and p2) else "FAIL",
        })

    print()
    print("=" * 84)
    verdict = "PASS" if any_pass else "FAIL"
    print(f"预注册判据总判定: {verdict}")
    print("=" * 84)
    if any_pass:
        print("至少一个 step 档位同时满足 P1 与 P2：")
        print("  => D1 支持 RQ1：存在 CD 几乎无差别而均匀性显著恶化的情形。")
        print("  => 仅此一条。本实验【未】证明加入对抗/均匀性约束能提升上采样性能，")
        print("     后者须由 B 组消融给出（PAPER_REMAKE_PLAN §3.3）。")
    else:
        print("无任何 step 档位同时满足 P1 与 P2：")
        print("  => D1 不支持 RQ1。按预注册约定，论文必须删除「CD 盲区」机制性叙事，")
        print("     不得改判据、不得调参重试。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "D1v2_cd_blindspot",
        "supersedes": "D1_cd_blindspot (v1, 作废：两构型误差预算不对等)",
        "seed": SEED,
        "n_points": N_POINTS,
        "perturb_frac": PERTURB_FRAC,
        "steps": list(STEPS),
        "preregistered_criteria": {
            "cd_tol": CD_TOL,
            "nuc_min_ratio": NUC_MIN_RATIO,
            "note": "跑前定死，不得事后修改",
        },
        "rows": rows,
        "verdict": verdict,
        "caveats": [
            "NUC 为简化实现，仅用于本实验内相对比较，不与文献绝对值对表",
            "纯构造实验，不涉及训练与数据集",
            "本实验只关乎 CD 指标的敏感性，不构成对抗约束有效性的证据",
        ],
    }
    out = OUT_DIR / "result.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"结果已落盘: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
