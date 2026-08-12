"""点云上采样标准评测指标。

本模块实现点云上采样领域的四项标准指标，协议依据见 docs/SOTA_SURVEY.md §3。

设计原则
--------
1. **numpy 精确版求准，torch 版求快**：评测出论文数字一律走 numpy 精确版；
   torch 版仅用于训练时的可微损失。两者必须交叉验证（见 self_check）。
2. **不依赖 CGAL**：官方 P2F 依赖 CGAL C++ 库（Windows 上是硬坑），
   本实现用 trimesh 的最近点查询替代，需与文献值校准后方可信任。
3. **单位约定**：论文报告值单位为 1e-3。本模块返回**原始值**，
   由调用方决定是否乘 1e3。绝不在模块内部偷偷缩放。

指标定义（均为越小越好）
----------------------
- CD   : Chamfer Distance，双向最近邻**平方**距离的均值之和
- HD   : Hausdorff Distance，双向最近邻**平方**距离最大值的较大者
- P2F  : Point-to-Surface，预测点到原始 mesh 表面的 L2 距离（均值/标准差）
- NUC  : Normalized Uniformity Coefficient，多半径 disk 均匀性（**简化实现**）

⚠️ 协议权威依据
--------------
CD / HD / P2F 的精确定义以 PU-GCN 官方一手代码为准，已逐行核对并落盘存证：

    refs/pu_gcn/CD_PROTOCOL_SOURCE.md          （分析与结论）
    refs/pu_gcn/snapshot/*.py|*.cpp            （官方源码原文 + sha256）

关键点（曾经写错过，务必先读上述文件再改任何默认值）：
1. `tf_nndistance` 返回 `reduce_sum(diff**2)` = **平方距离，不开方**
2. `evaluate.py` 中 `CD = mean(forward) + mean(backward)`
3. `Common/metrics.py` 中 `HD = max(max(forward), max(backward))`，同为平方量纲
4. 评测阶段 pred 与 gt **各自独立**归一化（训练阶段 input/gt 必须共享，勿混）
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# =============================================================================
# 归一化
# =============================================================================

def normalize_point_cloud(pc: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """按最大半径归一化到单位球。

    协议说明：点云上采样文献普遍使用「中心化 + 最大半径缩放」而非包围盒对角线，
    这样 CD 值在不同形状间可比。返回 center/scale 以便逆变换与复现。

    Parameters
    ----------
    pc : (N, 3) 点云

    Returns
    -------
    normalized : (N, 3) 归一化后点云
    center : (1, 3) 质心
    scale : float 最大半径
    """
    pc = np.asarray(pc, dtype=np.float64)
    center = pc.mean(axis=0, keepdims=True)
    centered = pc - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-12:
        scale = 1.0
    return centered / scale, center, scale


# =============================================================================
# CD / HD —— numpy 精确版
# =============================================================================

def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 中每个点到 b 的最近邻欧氏距离。"""
    if _HAS_SCIPY:
        tree = cKDTree(b)
        d, _ = tree.query(a, k=1)
        return np.asarray(d, dtype=np.float64)
    # 无 scipy 时退化为暴力计算（小规模可用）
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)


def chamfer_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    squared: bool = True,
) -> float:
    """Chamfer Distance（对称，双向平均）—— 默认对齐 PU-GCN 官方协议。

    官方定义（squared=True，本项目主表口径）::

        CD = mean_{x in pred} min_{y in gt} ||x-y||^2
           + mean_{y in gt} min_{x in pred} ||y-x||^2

    Parameters
    ----------
    squared : 距离本体是否用平方。**默认 True。**

        依据（一手代码，见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md §1）：
        - `tf_ops/nn_distance/tf_nndistance_cpu.py`:
          `pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1)` —— 无 sqrt
        - `evaluate.py`: `row["CD"] = cd_forward_value + cd_backward_value`
          其中两者均为上述平方距离的 `np.mean`

        历史教训：本函数曾默认 `squared=False` 并注明「与 PU-GCN evaluate.py 一致」，
        该注释**未经核实且是错的**，直接导致 E-000 校准量纲偏离文献 25 倍。
        切换此开关会使数值差一个量级，绝不可在同一张表里混用。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    d1 = _nn_dist(pred, gt)
    d2 = _nn_dist(gt, pred)
    if squared:
        d1, d2 = d1 ** 2, d2 ** 2
    return float(d1.mean() + d2.mean())


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    squared: bool = True,
) -> float:
    """双向 Hausdorff 距离（对最坏离群点敏感）—— 默认对齐 PU-GCN 官方协议。

    官方定义（squared=True，本项目主表口径）::

        HD = max( max_x min_y ||x-y||^2 ,  max_y min_x ||y-x||^2 )

    依据（见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md §2）：
    `Common/metrics.py::hausdorff_from_nn_distances` 取两方向最大值的较大者
    （其 docstring 明确指出「不是两个 max 相加」），而输入数组正是
    `tf_nndistance` 输出的**平方距离** —— 故 HD 与 CD 同处平方量纲。

    Parameters
    ----------
    squared : 距离本体是否用平方。**默认 True**，须与 CD 保持一致。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    d1 = _nn_dist(pred, gt)
    d2 = _nn_dist(gt, pred)
    if squared:
        d1, d2 = d1 ** 2, d2 ** 2
    return float(max(d1.max(), d2.max()))


# =============================================================================
# 官方协议评测入口
# =============================================================================

def eval_cd_hd_official(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, float]:
    """按 PU-GCN 官方协议计算 CD / HD，返回论文口径（已乘 1e3）与原始值。

    本函数是**论文主表 CD / HD 数字的唯一出口**。它把三条易错约束固化在代码里，
    不依赖调用方记得：

    1. pred 与 gt **各自独立**归一化（官方 `evaluate.py` 两次调用
       `normalize_point_cloud`，返回的 centroid/scale 均未复用）。
       —— 注意与训练阶段区分：训练时 input/gt 必须共享 gt 的 center/scale。
    2. 距离本体用**平方距离**。
    3. CD 双向 mean 求和；HD 取双向 max 的较大者。

    依据：refs/pu_gcn/CD_PROTOCOL_SOURCE.md §1-§3（含官方源码 sha256 存证）。

    Parameters
    ----------
    pred : (N, 3) 预测点云（**原始坐标尺度**，本函数内部归一化）
    gt   : (M, 3) 真值点云（**原始坐标尺度**，本函数内部归一化）

    Returns
    -------
    dict 含以下键：
        cd            原始值（归一化坐标系下）
        hd            原始值
        cd_1e3        cd * 1000，论文报告口径
        hd_1e3        hd * 1000，论文报告口径
        pred_scale    pred 自身归一化用的最大半径
        gt_scale      gt 自身归一化用的最大半径
        scale_ratio   pred_scale / gt_scale，健全性检查用（应接近 1）
    """
    pred_n, _, pred_scale = normalize_point_cloud(pred)
    gt_n, _, gt_scale = normalize_point_cloud(gt)

    cd = chamfer_distance(pred_n, gt_n, squared=True)
    hd = hausdorff_distance(pred_n, gt_n, squared=True)

    return {
        "cd": cd,
        "hd": hd,
        "cd_1e3": cd * 1e3,
        "hd_1e3": hd * 1e3,
        "pred_scale": pred_scale,
        "gt_scale": gt_scale,
        "scale_ratio": pred_scale / gt_scale if gt_scale > 1e-12 else float("nan"),
    }


# =============================================================================
# P2F —— 点到 mesh 表面距离
# =============================================================================

def point_to_surface(
    pred: np.ndarray,
    mesh,
) -> tuple[float, float]:
    """预测点到原始 mesh 表面的距离（均值, 标准差）。

    官方实现依赖 CGAL（`evaluation_code/evaluation.cpp`）；此处用 trimesh 的
    精确最近点查询替代。

    与官方一致的两点（见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md §4）：
    - 距离是 **L2 距离（开方）**，注意与 CD/HD 的平方量纲不同；
    - 计算发生在**原始坐标尺度**上 —— `evaluation.cpp` 直接读 mesh 的 `.off`
      与预测 `.xyz`，中间无任何归一化。**故传入 pred 必须是未归一化的原始坐标。**

    ⚠️ 已实测的指标盲区（E-000 检验项 E4 成立）：P2F 对「点是否贴在表面上」敏感，
    对「点分布是否均匀」几乎不敏感 —— 把输入点原样复制 4 份（零上采样价值）
    也能拿到近乎完美的 P2F。**绝不可单独用 P2F 论证方法优劣**，须写入 Limitations。

    Parameters
    ----------
    pred : (N, 3) 预测点云（**原始坐标尺度，不要预先归一化**）
    mesh : trimesh.Trimesh 原始三角网格

    Returns
    -------
    (mean, std) 距离统计量
    """
    import trimesh  # 延迟导入，避免无 mesh 场景下的硬依赖

    pred = np.asarray(pred, dtype=np.float64)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh 必须是 trimesh.Trimesh，收到 {type(mesh)}")

    # closest_point 返回 (closest_points, distances, triangle_ids)
    _, dist, _ = trimesh.proximity.closest_point(mesh, pred)
    dist = np.abs(np.asarray(dist, dtype=np.float64))
    return float(dist.mean()), float(dist.std())


# =============================================================================
# 均匀性 NUC
# =============================================================================

def uniformity_nuc(
    pred: np.ndarray,
    radii: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2),
    n_seeds: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """归一化均匀性系数（Normalized Uniformity Coefficient）。

    协议依据 PU-GAN：在归一化点云表面撒 D 个 disk（半径为 r*sqrt(p)，p 为面积占比），
    统计每个 disk 内的点数，理想情况下点数应等于 n_total * p。
    NUC 为实际点数与期望点数比值的方差 —— 越小越均匀。

    本实现为简化版：以随机选中的已有点为 disk 中心（避免依赖 mesh 表面采样），
    用欧氏球邻域计数，指标取比值方差。

    ⚠️ 与官方实现的结构性差异（一手核对，见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md §5）：

    | 项 | 官方 PU-GCN | 本实现 |
    |---|---|---|
    | disk 成员判定 | **测地距离**（CGAL Surface_mesh_shortest_path） | 欧氏距离 |
    | disk 内点 | 投影到 mesh 表面后的点 | 原始预测点 |
    | disk 中心 | mesh 表面按面积加权随机采样 | 随机选已有点 |
    | 档位 | **2 档**（p=0.8%, 1.2%） | 5 档（0.4%~1.2%） |
    | 公式 | coverage × spacing 偏差，二者**相乘** | 仅比值方差 |

    官方公式为::

        coverage = (density - p*N)^2 / (p*N)
        expect_d = sqrt(2 * (pi*r^2/n_in) / sqrt(3))
        spacing  = mean( (nn_dist - expect_d)^2 / expect_d )
        uniform  = mean over 1000 disks of ( coverage * spacing )

    → **本实现只可用于本文内部方法间相对比较，绝不可与文献 uniformity 绝对值对表。**
    这一限制必须写入论文 Limitations。若必须对表，需在 Linux 上编译 CGAL 评测程序。

    Parameters
    ----------
    pred : (N, 3) 点云（内部会归一化）
    radii : 各档 disk 半径比例 p 的开方前系数（PU-GAN 用 p = 0.4%~1.2%）
    n_seeds : disk 数量

    Returns
    -------
    dict: 每档半径的 NUC 值 + 平均值
    """
    pts, _, _ = normalize_point_cloud(pred)
    n_total = len(pts)
    rng = np.random.default_rng(seed)

    if not _HAS_SCIPY:
        raise RuntimeError("uniformity_nuc 需要 scipy")

    tree = cKDTree(pts)
    n_seeds = min(n_seeds, n_total)
    seed_idx = rng.choice(n_total, size=n_seeds, replace=False)
    seeds = pts[seed_idx]

    out: dict[str, float] = {}
    values = []
    for p in radii:
        # p 为面积占比（百分数形式，如 0.4 表示 0.4%）
        frac = p / 100.0
        # 单位球表面积 4*pi；disk 半径 r 使 disk 面积 / 总面积 = frac
        r = float(np.sqrt(frac * 4.0))
        counts = np.array([len(tree.query_ball_point(s, r)) for s in seeds],
                          dtype=np.float64)
        expected = n_total * frac
        if expected < 1e-12:
            continue
        ratio = counts / expected
        # NUC = 比值的方差（PU-GAN 定义）
        nuc = float(np.var(ratio))
        out[f"nuc_p{p}"] = nuc
        values.append(nuc)

    out["nuc_mean"] = float(np.mean(values)) if values else float("nan")
    return out


# =============================================================================
# torch 可微版（训练用）
# =============================================================================

if _HAS_TORCH:

    def chamfer_distance_torch(
        pred: "torch.Tensor",
        gt: "torch.Tensor",
        squared: bool = True,
    ) -> "torch.Tensor":
        """可微 Chamfer Distance。

        Parameters
        ----------
        pred : (B, N, 3)
        gt   : (B, M, 3)
        squared : 默认 True，与 numpy 版及 PU-GCN 官方协议一致。

        Returns
        -------
        (B,) 每个样本的 CD

        ⚠️ 使用 O(N*M) 显存的朴素实现。N*M 过大时需分块或换 CUDA 扩展。

        注：训练损失用哪个口径是**独立决策**（PU-Transformer 原文用 modified CD），
        与本函数默认值无关；训练侧显式传参，不要依赖默认。
        """
        if pred.dim() != 3 or gt.dim() != 3:
            raise ValueError(f"期望 (B,N,3)，收到 pred={pred.shape} gt={gt.shape}")
        # (B, N, M)
        d2 = torch.cdist(pred, gt, p=2) ** 2
        min_pred = d2.min(dim=2).values  # (B, N)
        min_gt = d2.min(dim=1).values    # (B, M)
        if not squared:
            min_pred = torch.sqrt(min_pred.clamp_min(1e-12))
            min_gt = torch.sqrt(min_gt.clamp_min(1e-12))
        return min_pred.mean(dim=1) + min_gt.mean(dim=1)

    def repulsion_loss_torch(
        pred: "torch.Tensor",
        k: int = 5,
        radius: float = 0.07,
        eps: float = 1e-12,
    ) -> "torch.Tensor":
        """PU-Net/PU-GAN 的排斥损失，抑制点聚集。

        对每个点的 k 近邻（不含自身），距离越近惩罚越大。

        Parameters
        ----------
        pred : (B, N, 3)
        k : 近邻数
        radius : 影响半径，超出则不惩罚

        Returns
        -------
        (B,) 每个样本的 repulsion loss
        """
        d2 = torch.cdist(pred, pred, p=2) ** 2          # (B, N, N)
        # 取 k+1 个最近（含自身距离 0），丢掉自身
        knn_d2, _ = torch.topk(d2, k=k + 1, dim=2, largest=False)
        knn_d2 = knn_d2[:, :, 1:]                        # (B, N, k)
        dist = torch.sqrt(knn_d2.clamp_min(eps))
        # 权重：距离小于 radius 时才惩罚
        weight = torch.exp(-(dist ** 2) / (radius ** 2))
        loss = torch.clamp(radius - dist, min=0.0) * weight
        return loss.mean(dim=(1, 2))


# =============================================================================
# 自检
# =============================================================================

def _reference_nn_distance(pc1: np.ndarray, pc2: np.ndarray):
    """PU-GCN `tf_nndistance_cpu.nn_distance_cpu` 的 numpy 逐行等价实现。

    仅用于自检时做独立参照，**不用于生产路径**（O(N*M) 显存）。
    源码见 refs/pu_gcn/snapshot/tf_ops__nn_distance__tf_nndistance_cpu.py::

        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1)
        dist1 = tf.reduce_min(pc_dist, axis=2)
        dist2 = tf.reduce_min(pc_dist, axis=1)

    Returns
    -------
    (dist1, dist2) 均为**平方距离**数组
    """
    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)
    diff = pc1[:, None, :] - pc2[None, :, :]
    d = (diff ** 2).sum(axis=-1)
    return d.min(axis=1), d.min(axis=0)


def self_check() -> bool:
    """模块自检：用解析可算的构造用例 + 官方源码等价参照验证每个指标。

    验证项
    ------
    1. 完全相同的点云 → CD = 0, HD = 0
    2. 单点平移 t（squared 口径）→ CD = 2t², HD = t²；解析解
    3. numpy CD 与 torch CD 数值一致（交叉验证，squared 口径）
    4. 均匀网格的 NUC 应显著低于聚集点云
    5. 归一化后最大半径恰为 1
    6. **CD/HD 与 PU-GCN 官方 nn_distance 语义逐行参照实现一致**
    7. **eval_cd_hd_official 对输入做各自独立归一化**（缩放不变性）
    8. squared=False 与 squared=True 关系自洽（同一构造下手工核算）
    """
    ok = True
    rng = np.random.default_rng(42)

    print("=" * 62)
    print("metrics/pointcloud.py 自检（协议：PU-GCN 官方，squared 口径）")
    print("=" * 62)

    # --- 1. 相同点云 ---
    pc = rng.standard_normal((500, 3))
    cd0 = chamfer_distance(pc, pc)
    hd0 = hausdorff_distance(pc, pc)
    p1 = abs(cd0) < 1e-12 and abs(hd0) < 1e-12
    ok &= p1
    print(f"[{'PASS' if p1 else 'FAIL'}] 相同点云: CD={cd0:.3e} HD={hd0:.3e} (期望 0)")

    # --- 2. 单点平移，解析解（squared 口径） ---
    t = 0.1
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[t, 0.0, 0.0]])
    cd_shift = chamfer_distance(a, b)
    hd_shift = hausdorff_distance(a, b)
    exp_cd, exp_hd = 2 * t ** 2, t ** 2
    p2 = abs(cd_shift - exp_cd) < 1e-12 and abs(hd_shift - exp_hd) < 1e-12
    ok &= p2
    print(f"[{'PASS' if p2 else 'FAIL'}] 单点平移 t={t}: "
          f"CD={cd_shift:.8f} (期望 2t^2={exp_cd}) "
          f"HD={hd_shift:.8f} (期望 t^2={exp_hd})")

    # --- 3. numpy vs torch 交叉验证 ---
    if _HAS_TORCH:
        p = rng.standard_normal((256, 3))
        g = rng.standard_normal((512, 3))
        cd_np = chamfer_distance(p, g)
        cd_t = chamfer_distance_torch(
            torch.from_numpy(p).unsqueeze(0),
            torch.from_numpy(g).unsqueeze(0),
        ).item()
        diff = abs(cd_np - cd_t)
        p3 = diff < 1e-9
        ok &= p3
        print(f"[{'PASS' if p3 else 'FAIL'}] numpy vs torch CD: "
              f"{cd_np:.9f} vs {cd_t:.9f} (diff={diff:.2e})")
    else:
        print("[SKIP] torch 未安装，跳过交叉验证")

    # --- 4. 均匀 vs 聚集 的 NUC ---
    if _HAS_SCIPY:
        # 球面近似均匀：Fibonacci 球
        n = 4096
        idx = np.arange(n, dtype=np.float64) + 0.5
        phi = np.arccos(1 - 2 * idx / n)
        theta = np.pi * (1 + 5 ** 0.5) * idx
        uniform = np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=1)
        # 聚集：一半点挤在小区域
        clustered = uniform.copy()
        clustered[: n // 2] = uniform[: n // 2] * 0.05 + np.array([0.9, 0.0, 0.0])

        nuc_u = uniformity_nuc(uniform)["nuc_mean"]
        nuc_c = uniformity_nuc(clustered)["nuc_mean"]
        p4 = nuc_u < nuc_c
        ok &= p4
        print(f"[{'PASS' if p4 else 'FAIL'}] NUC 均匀={nuc_u:.6f} < 聚集={nuc_c:.6f}")
    else:
        print("[SKIP] scipy 未安装，跳过 NUC 检查")

    # --- 5. 归一化 ---
    raw = rng.standard_normal((300, 3)) * 7.0 + 3.0
    norm, center, scale = normalize_point_cloud(raw)
    max_r = float(np.max(np.linalg.norm(norm, axis=1)))
    p5 = abs(max_r - 1.0) < 1e-12
    ok &= p5
    print(f"[{'PASS' if p5 else 'FAIL'}] 归一化最大半径={max_r:.12f} (期望 1.0), "
          f"scale={scale:.4f}")

    # --- 6. 与官方 nn_distance 语义逐行参照对比 ---
    pa = rng.standard_normal((120, 3))
    pb = rng.standard_normal((200, 3)) * 1.3
    ref_d1, ref_d2 = _reference_nn_distance(pa, pb)
    ref_cd = float(ref_d1.mean() + ref_d2.mean())
    ref_hd = float(max(ref_d1.max(), ref_d2.max()))
    got_cd = chamfer_distance(pa, pb, squared=True)
    got_hd = hausdorff_distance(pa, pb, squared=True)
    dcd, dhd = abs(ref_cd - got_cd), abs(ref_hd - got_hd)
    p6 = dcd < 1e-12 and dhd < 1e-12
    ok &= p6
    print(f"[{'PASS' if p6 else 'FAIL'}] 官方语义参照: "
          f"CD {got_cd:.12f} vs {ref_cd:.12f} (d={dcd:.2e}) | "
          f"HD {got_hd:.12f} vs {ref_hd:.12f} (d={dhd:.2e})")

    # --- 7. eval_cd_hd_official 各自独立归一化 → 缩放不变 ---
    src_pred = rng.standard_normal((300, 3))
    src_gt = src_pred + rng.standard_normal((300, 3)) * 0.02
    r_base = eval_cd_hd_official(src_pred, src_gt)
    # pred 单独放大 100 倍并平移：若各自独立归一化，CD 应几乎不变
    r_scaled = eval_cd_hd_official(src_pred * 100.0 + 7.0, src_gt)
    rel = abs(r_scaled["cd"] - r_base["cd"]) / max(r_base["cd"], 1e-12)
    p7 = rel < 1e-9
    ok &= p7
    print(f"[{'PASS' if p7 else 'FAIL'}] 独立归一化(pred x100+7): "
          f"CD {r_base['cd_1e3']:.6f}e-3 -> {r_scaled['cd_1e3']:.6f}e-3 "
          f"(相对变化={rel:.2e}) scale_ratio={r_scaled['scale_ratio']:.4f}")

    # --- 8. squared 开关自洽性 ---
    d1 = _nn_dist(pa, pb)
    d2 = _nn_dist(pb, pa)
    manual_l2 = float(d1.mean() + d2.mean())
    manual_sq = float((d1 ** 2).mean() + (d2 ** 2).mean())
    got_l2 = chamfer_distance(pa, pb, squared=False)
    p8 = (abs(got_l2 - manual_l2) < 1e-12
          and abs(got_cd - manual_sq) < 1e-12)
    ok &= p8
    print(f"[{'PASS' if p8 else 'FAIL'}] squared 开关自洽: "
          f"L2={got_l2:.9f} sq={got_cd:.9f} (比值={got_cd/max(got_l2,1e-12):.4f})")

    print("-" * 62)
    print(f"总体: {'PASS' if ok else 'FAIL'}")
    print("协议依据: refs/pu_gcn/CD_PROTOCOL_SOURCE.md")
    print("=" * 62)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
