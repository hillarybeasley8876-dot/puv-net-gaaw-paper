"""排查 CD 量纲问题：为什么作弊上界 (11.39e-3) 比文献 SOTA (0.451e-3) 差 25 倍。

只做测量，不改任何评测代码，不下结论前先拿到数字。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import PU1KTestSet  # noqa: E402
from puvnet.metrics.pointcloud import (  # noqa: E402
    chamfer_distance, normalize_point_cloud,
)

ts = PU1KTestSet(input_n=2048, up_ratio=4)
name = ts.names[0]
inp, gt = ts.load(name)

print("=" * 78)
print("CD 量纲排查")
print("=" * 78)
print(f"样本: {name}")
print(f"input {inp.shape}  gt {gt.shape}")
print()

# 1. 原始坐标范围
for tag, pc in (("input", inp), ("gt", gt)):
    c = pc.mean(0)
    r = np.linalg.norm(pc - c, axis=1)
    print(f"{tag:<6} 质心={np.round(c,4)}  最大半径={r.max():.6f}  "
          f"包围盒={np.round(pc.max(0)-pc.min(0),4)}")
print()

# 2. GT 自身的平均最近邻间距 —— CD 的天然下界参考
from scipy.spatial import cKDTree
tree = cKDTree(gt)
d, _ = tree.query(gt, k=2)
nn = d[:, 1]
print(f"GT 平均最近邻间距 = {nn.mean():.6f}  中位数={np.median(nn):.6f}")
print(f"  归一化后(除以最大半径 {np.linalg.norm(gt-gt.mean(0),axis=1).max():.4f}): "
      f"{nn.mean()/np.linalg.norm(gt-gt.mean(0),axis=1).max():.6f}")
print()

# 3. 完美预测（gt 自己）的 CD —— 必须为 0
gt_norm, c0, s0 = normalize_point_cloud(gt)
print(f"[A] CD(gt, gt) 归一化后        = {chamfer_distance(gt_norm, gt_norm):.8f}  (应为0)")

# 4. GT 抽一半复制 —— 我在 E-000 里用的作弊构造
from puvnet.data.pu_dataset import farthest_point_sample
idx = farthest_point_sample(gt, len(gt)//2, seed=0)
half = gt[idx]
dup = np.concatenate([half, half], axis=0)[:len(gt)]
dup_norm = (dup - c0) / s0
print(f"[B] CD(gt_half_dup, gt) 归一化 = {chamfer_distance(dup_norm, gt_norm)*1e3:.4f} e-3")

# 5. 只抽一半、不复制（点数只有一半）
half_norm = (half - c0) / s0
print(f"[C] CD(gt_half, gt) 归一化     = {chamfer_distance(half_norm, gt_norm)*1e3:.4f} e-3")

# 6. 关键对照：CD 在【原始尺度】下算（不归一化）
print(f"[D] CD(gt_half_dup, gt) 原尺度 = {chamfer_distance(dup, gt)*1e3:.4f} e-3")
print()

# 7. 单向 vs 双向
def cd_oneway(a, b):
    t = cKDTree(b)
    dd, _ = t.query(a, k=1)
    return dd.mean()

print("单向分解（归一化尺度）:")
print(f"  pred->gt = {cd_oneway(dup_norm, gt_norm)*1e3:.4f} e-3")
print(f"  gt->pred = {cd_oneway(gt_norm, dup_norm)*1e3:.4f} e-3")
print(f"  双向和   = {(cd_oneway(dup_norm,gt_norm)+cd_oneway(gt_norm,dup_norm))*1e3:.4f} e-3")
print(f"  双向均值 = {(cd_oneway(dup_norm,gt_norm)+cd_oneway(gt_norm,dup_norm))/2*1e3:.4f} e-3")
print()

# 8. 平方距离版本
print("平方距离版本（文献部分用 squared CD）:")
print(f"  CD_squared(gt_half_dup, gt) 归一化 = "
      f"{chamfer_distance(dup_norm, gt_norm, squared=True)*1e3:.4f} e-3")
print()

# 9. patch 尺度参考：256 点邻域的半径
tree_in = cKDTree(inp)
_, nbr = tree_in.query(inp[0], k=256)
patch = inp[nbr]
pr = np.linalg.norm(patch - patch.mean(0), axis=1).max()
print(f"典型 256 点 patch 最大半径 = {pr:.6f}")
print(f"整模型最大半径             = {np.linalg.norm(gt-gt.mean(0),axis=1).max():.6f}")
print(f"比值 (整模型/patch)        = "
      f"{np.linalg.norm(gt-gt.mean(0),axis=1).max()/pr:.2f}")
