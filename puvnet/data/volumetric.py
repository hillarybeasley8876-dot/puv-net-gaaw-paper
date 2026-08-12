"""
体积点云数据生成 —— 从水密网格合成「表面稀疏输入 + 体积内部真值」。

这是方向 2 的核心实现。命题成立的前提是内部真值可信，因此本模块的
每一步都必须可验证、可复现，不允许出现"看起来对"的静默失败。

关键设计决策（写进论文 Methods 的内容）
--------------------------------------
1. 水密性是硬门禁。非水密网格的内外判定无意义，必须显式拒绝而非静默跳过。
2. 归一化在采样之前完成，且使用「表面点云」的包围盒中心与最大半径，
   保证输入与真值处于同一坐标系，且与测试时的真实输入分布一致。
3. 内部点采样使用「拒绝采样 + contains 判定」，不使用体素中心近似，
   因为体素化会引入分辨率相关的伪结构，消融时无法区分是网络效果还是体素伪影。
4. 表面点与内部点分别记录，训练时可按比例混合，便于 H1/H6 的消融。
5. 每个样本落盘时附带 provenance（源文件、随机种子、点数、水密性检查结果），
   审稿人问"数据怎么造的"时可逐样本回溯。
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize_to_unit_sphere(points: np.ndarray, center: np.ndarray | None = None,
                             scale: float | None = None):
    """把点云归一化到单位球内。

    返回 (normalized_points, center, scale)，center/scale 可复用于同一样本的
    其他点集，保证输入与真值共享坐标系。

    注意：使用最大半径而非包围盒对角线，因为前者对旋转更稳定，
    这对 H1 的旋转鲁棒性实验很重要。
    """
    if center is None:
        center = points.mean(axis=0)
    centered = points - center
    if scale is None:
        radii = np.linalg.norm(centered, axis=1)
        scale = float(radii.max())
        if scale < 1e-12:
            raise ValueError("退化点云：所有点重合，无法归一化")
    return centered / scale, center, scale


# ---------------------------------------------------------------------------
# 水密性检查
# ---------------------------------------------------------------------------

@dataclass
class WatertightReport:
    """水密性检查报告。任何一项为 False 都意味着内外判定不可信。"""
    is_watertight: bool
    is_winding_consistent: bool
    euler_number: int
    volume: float
    n_vertices: int
    n_faces: int

    @property
    def usable(self) -> bool:
        """是否可用于生成内部真值。

        要求水密 + 绕序一致 + 体积为正。体积为负说明法向朝内，
        contains 判定会整体反转，这是最隐蔽的一类错误。
        """
        return (self.is_watertight
                and self.is_winding_consistent
                and self.volume > 1e-9)

    def reason(self) -> str:
        if not self.is_watertight:
            return "网格非水密（存在边界洞），内外判定无意义"
        if not self.is_winding_consistent:
            return "面绕序不一致，法向朝向混乱，contains 判定不可信"
        if self.volume <= 1e-9:
            return f"体积非正 (volume={self.volume:.3e})，法向可能整体朝内"
        return "ok"


def check_watertight(mesh) -> WatertightReport:
    """检查 trimesh 网格能否用于生成内部真值。"""
    return WatertightReport(
        is_watertight=bool(mesh.is_watertight),
        is_winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        volume=float(mesh.volume),
        n_vertices=int(len(mesh.vertices)),
        n_faces=int(len(mesh.faces)),
    )


def try_repair(mesh):
    """尝试修复轻度破损的网格。

    只做保守修复（去重、去退化面、修绕序、填小洞）。
    修复后仍需重新走 check_watertight —— 不假设修复一定成功。
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_infinite_values()
    mesh.fix_normals()
    try:
        mesh.fill_holes()
    except Exception:
        pass
    return mesh


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def sample_surface(mesh, n: int, rng: np.random.Generator):
    """按面积均匀采样表面点，同时返回对应法线。

    法线来自所在三角面，而非顶点插值 —— 顶点法线在稀疏输入下会被平滑，
    削弱 H3（法线融合）的信号强度。
    """
    import trimesh
    points, face_idx = trimesh.sample.sample_surface(
        mesh, n, seed=int(rng.integers(0, 2**31 - 1)))
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    return np.asarray(points, dtype=np.float64), normals


def sample_interior(mesh, n: int, rng: np.random.Generator,
                    max_rounds: int = 60, oversample: float = 4.0):
    """拒绝采样生成严格位于网格内部的点。

    在包围盒内批量撒点，用 mesh.contains 判定，保留内部点，直到收集够 n 个。

    为什么不用体素中心：体素化的点会落在规则栅格上，网络可以直接学到
    这个栅格先验从而"作弊"，消融实验会得到虚高的收益。随机内部点没有
    这个问题。

    为什么限制 max_rounds：薄壳物体的内部体积占包围盒比例极小，
    可能永远采不够。这种样本必须显式失败而不是死循环。
    """
    lo, hi = mesh.bounds
    collected = []
    total = 0
    batch = max(int(n * oversample), 1024)

    for _ in range(max_rounds):
        cand = rng.uniform(lo, hi, size=(batch, 3))
        inside = mesh.contains(cand)
        hits = cand[inside]
        total += batch
        if len(hits):
            collected.append(hits)
            if sum(len(c) for c in collected) >= n:
                break

    if not collected:
        raise RuntimeError("拒绝采样未命中任何内部点：网格可能是零体积薄壳")

    pts = np.concatenate(collected, axis=0)
    if len(pts) < n:
        raise RuntimeError(
            f"内部点不足：需要 {n}，实得 {len(pts)}，"
            f"命中率 {len(pts)/total:.4%}。该样本内部体积过小，应剔除")

    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx].astype(np.float64), float(len(pts) / total)


# ---------------------------------------------------------------------------
# 单样本生成
# ---------------------------------------------------------------------------

@dataclass
class SampleSpec:
    """一个样本的生成规格。全部参数显式化，便于写进论文和复现。"""
    n_input: int = 1024          # 稀疏表面输入点数
    n_surface_gt: int = 4096     # 表面真值点数
    n_interior_gt: int = 4096    # 内部真值点数
    seed: int = 0


def build_sample(mesh_path: str | Path, spec: SampleSpec,
                 allow_repair: bool = True) -> dict:
    """从一个网格文件生成一个完整训练样本。

    返回 dict，包含点集与完整 provenance。任何一步不可信都直接抛异常，
    绝不返回"部分可用"的样本。
    """
    import trimesh

    mesh_path = Path(mesh_path)
    rng = np.random.default_rng(spec.seed)

    loaded = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{mesh_path.name}: 不是单一三角网格")

    report = check_watertight(loaded)
    repaired = False
    if not report.usable and allow_repair:
        loaded = try_repair(loaded)
        report = check_watertight(loaded)
        repaired = True

    if not report.usable:
        raise ValueError(f"{mesh_path.name}: {report.reason()}")

    # 采样在原始尺度下进行（contains 依赖真实几何），归一化在最后统一做
    surf_gt, surf_normals = sample_surface(loaded, spec.n_surface_gt, rng)
    inter_gt, hit_rate = sample_interior(loaded, spec.n_interior_gt, rng)
    inp, inp_normals = sample_surface(loaded, spec.n_input, rng)

    # 关键：用表面真值确定坐标系，输入与内部真值共享同一 center/scale
    surf_gt_n, center, scale = normalize_to_unit_sphere(surf_gt)
    inter_gt_n, _, _ = normalize_to_unit_sphere(inter_gt, center, scale)
    inp_n, _, _ = normalize_to_unit_sphere(inp, center, scale)

    return {
        "input": inp_n.astype(np.float32),
        "input_normals": inp_normals.astype(np.float32),
        "surface_gt": surf_gt_n.astype(np.float32),
        "surface_normals": surf_normals.astype(np.float32),
        "interior_gt": inter_gt_n.astype(np.float32),
        "provenance": {
            "source": str(mesh_path),
            "source_sha1": _file_sha1(mesh_path),
            "spec": asdict(spec),
            "watertight": asdict(report),
            "repaired": repaired,
            "interior_hit_rate": hit_rate,
            "center": center.tolist(),
            "scale": scale,
        },
    }


def _file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_sample(sample: dict, out_path: str | Path) -> None:
    """落盘为 npz + 同名 json provenance。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in sample.items() if k != "provenance"}
    np.savez_compressed(out_path, **arrays)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(sample["provenance"], f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 自检：用解析几何验证 pipeline 正确性
# ---------------------------------------------------------------------------

def self_check(verbose: bool = True) -> bool:
    """用单位立方体和球体验证 pipeline，因为它们的内部判定有解析解。

    这是防止"pipeline 静默产出错误真值"的关键一步。如果这个检查不过，
    后面所有实验的数据都是垃圾。
    """
    import trimesh
    ok = True

    def log(*a):
        if verbose:
            print(*a)

    # --- 立方体：内部点必须落在 |x|,|y|,|z| < 0.5 内 ---
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    rep = check_watertight(box)
    log(f"[box] watertight={rep.is_watertight} volume={rep.volume:.4f} usable={rep.usable}")
    if not rep.usable or abs(rep.volume - 1.0) > 1e-6:
        log("  !! 立方体体积应为 1.0"); ok = False

    rng = np.random.default_rng(0)
    pts, hr = sample_interior(box, 2000, rng)
    max_abs = np.abs(pts).max()
    log(f"[box] 内部点 max|coord|={max_abs:.4f} (应 < 0.5), 命中率={hr:.2%} (应 ≈100%)")
    if max_abs >= 0.5:
        log("  !! 有内部点落在立方体外"); ok = False

    # --- 球体：内部点半径必须 < R ---
    sph = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    rep = check_watertight(sph)
    pts, hr = sample_interior(sph, 2000, rng)
    radii = np.linalg.norm(pts - sph.centroid, axis=1)
    log(f"[sphere] 内部点 max radius={radii.max():.4f} (应 < 1.0), "
        f"命中率={hr:.2%} (理论 π/6≈52.4%)")
    if radii.max() >= 1.0:
        log("  !! 有内部点落在球外"); ok = False
    if not (0.40 < hr < 0.65):
        log(f"  !! 球体命中率偏离理论值 52.4% 过多"); ok = False

    # --- 归一化一致性：输入与真值必须共享坐标系 ---
    spec = SampleSpec(n_input=256, n_surface_gt=512, n_interior_gt=512, seed=1)
    tmp = Path("__selfcheck_box.ply")
    box.export(tmp)
    try:
        s = build_sample(tmp, spec)
        r_surf = np.linalg.norm(s["surface_gt"], axis=1).max()
        r_inter = np.linalg.norm(s["interior_gt"], axis=1).max()
        log(f"[norm] surface max radius={r_surf:.4f} (应 ≈1.0), "
            f"interior max radius={r_inter:.4f} (应 < 1.0)")
        if abs(r_surf - 1.0) > 1e-4:
            log("  !! 表面真值未归一化到单位球面"); ok = False
        if r_inter >= r_surf:
            log("  !! 内部点半径不应超过表面点，坐标系可能不一致"); ok = False
    finally:
        tmp.unlink(missing_ok=True)
        Path("__selfcheck_box.json").unlink(missing_ok=True)

    log(f"\nself_check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
