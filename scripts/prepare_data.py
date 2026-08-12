"""生成数据集。

两种来源：
  --synthetic  程序生成带已知内部结构的水密网格（空腔、管道、多孔）
  --meshes     从真实网格目录读取（ShapeNet 等）

为什么先做 synthetic
--------------------
论文的核心主张是"能还原空腔、内部管道、复杂支撑结构"。合成数据的优势是
内部结构**已知且可控**，可以直接验证网络是否真的学到了内部拓扑，
而不是靠运气。真实数据（ShapeNet）随后再加，两者互补：
  合成数据 → 证明机制有效（可控变量）
  真实数据 → 证明泛化能力（贴近应用）

这也回答了审稿人必问的"内部真值从哪来"：合成部分完全可复现，
真实部分用水密化 + 拒绝采样，生成方式全部写进 provenance。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from puvnet.data.volumetric import SampleSpec, build_sample, save_sample


# ---------------------------------------------------------------------------
# 合成网格：带已知内部结构
# ---------------------------------------------------------------------------

def make_hollow_sphere(rng, r_out=1.0, thickness=None):
    """空心球：外壳 + 内腔。内部真值集中在壳体内，中心是空的。

    这是最能区分"真内部建模"与"表面偏移"的构型：
    若网络只做向内挤压，会把点填进中心空腔，interior_cd 立刻恶化。
    """
    import trimesh
    t = thickness if thickness is not None else rng.uniform(0.15, 0.35)
    outer = trimesh.creation.icosphere(subdivisions=3, radius=r_out)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=r_out - t)
    return trimesh.boolean.difference([outer, inner])


def make_pipe_block(rng):
    """带贯穿管道的立方体。内部有圆柱形空洞。"""
    import trimesh
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    r = rng.uniform(0.15, 0.3)
    cyl = trimesh.creation.cylinder(radius=r, height=2.0)
    axis = rng.integers(0, 3)
    if axis == 0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
    return trimesh.boolean.difference([box, cyl])


def make_porous_block(rng, n_holes=3):
    """多孔块：立方体挖若干随机球洞，模拟多孔材料。"""
    import trimesh
    solid = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    for _ in range(n_holes):
        r = rng.uniform(0.12, 0.22)
        c = rng.uniform(-0.3, 0.3, size=3)
        sph = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sph.apply_translation(c)
        solid = trimesh.boolean.difference([solid, sph])
    return solid


def make_solid_shape(rng):
    """实心基本体，作为"无内部空腔"对照组。"""
    import trimesh
    kind = rng.integers(0, 3)
    if kind == 0:
        return trimesh.creation.box(extents=rng.uniform(0.6, 1.2, size=3))
    if kind == 1:
        return trimesh.creation.icosphere(subdivisions=3,
                                          radius=rng.uniform(0.5, 1.0))
    return trimesh.creation.cylinder(radius=rng.uniform(0.3, 0.6),
                                     height=rng.uniform(0.8, 1.6))


GENERATORS = {
    "hollow_sphere": make_hollow_sphere,
    "pipe_block": make_pipe_block,
    "porous_block": make_porous_block,
    "solid": make_solid_shape,
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def gen_synthetic(out_root: Path, n_per_kind: int, spec: SampleSpec,
                  splits=(("train", 0.7), ("val", 0.15), ("test", 0.15))):
    import trimesh

    made, failed = 0, []
    tmp = out_root / "_tmp_mesh"
    tmp.mkdir(parents=True, exist_ok=True)

    tasks = []
    for kind in GENERATORS:
        for i in range(n_per_kind):
            tasks.append((kind, i))

    # 按 kind 分层划分，保证每个 split 都覆盖全部构型
    rng_split = np.random.default_rng(12345)
    assign = {}
    for kind in GENERATORS:
        idx = np.arange(n_per_kind)
        rng_split.shuffle(idx)
        # 小样本(如 smoke)时 int() 会把 val/test 抹成 0，这里保证每个 split 至少 1 个
        if n_per_kind >= 3:
            n_va = max(1, int(round(n_per_kind * splits[1][1])))
            n_te = max(1, int(round(n_per_kind * splits[2][1])))
            n_tr = max(1, n_per_kind - n_va - n_te)
            # 若三者相加溢出（n_per_kind==3 时刚好 1/1/1），按 train 优先回收
            over = (n_tr + n_va + n_te) - n_per_kind
            if over > 0:
                n_tr = max(1, n_tr - over)
        else:
            # n_per_kind < 3 无法三分，全部进 train 并复用作 val/test（仅 debug 用）
            n_tr, n_va = n_per_kind, 0
        for j, ii in enumerate(idx):
            assign[(kind, int(ii))] = (
                "train" if j < n_tr else "val" if j < n_tr + n_va else "test")

    for kind, i in tasks:
        split = assign[(kind, i)]
        name = f"{kind}_{i:04d}"
        try:
            rng = np.random.default_rng(hash((kind, i)) % (2**32))
            mesh = GENERATORS[kind](rng)
            if mesh is None or len(mesh.faces) == 0:
                raise RuntimeError("布尔运算返回空网格")
            mesh_path = tmp / f"{name}.ply"
            mesh.export(mesh_path)

            s = SampleSpec(n_input=spec.n_input,
                           n_surface_gt=spec.n_surface_gt,
                           n_interior_gt=spec.n_interior_gt,
                           seed=i)
            sample = build_sample(mesh_path, s)
            sample["provenance"]["kind"] = kind
            save_sample(sample, out_root / split / f"{name}.npz")
            mesh_path.unlink(missing_ok=True)
            made += 1
            print(f"  [{made}] {split:5s} {name}  "
                  f"hit_rate={sample['provenance']['interior_hit_rate']:.1%}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))

    try:
        tmp.rmdir()
    except OSError:
        pass

    print(f"\n生成完成: {made} 成功, {len(failed)} 失败")
    if failed:
        print("失败样本（前 10）:")
        for n, r in failed[:10]:
            print(f"  {n}: {r}")
    return made, failed


def gen_from_meshes(mesh_dir: Path, out_root: Path, spec: SampleSpec,
                    limit: int | None = None):
    exts = {".obj", ".ply", ".stl", ".off", ".glb"}
    files = sorted(p for p in mesh_dir.rglob("*") if p.suffix.lower() in exts)
    if limit:
        files = files[:limit]
    print(f"发现 {len(files)} 个网格文件")

    made, failed = 0, []
    for i, f in enumerate(files):
        split = "train" if i % 10 < 7 else "val" if i % 10 < 85 // 10 + 7 else "test"
        try:
            sample = build_sample(f, SampleSpec(
                spec.n_input, spec.n_surface_gt, spec.n_interior_gt, seed=i))
            save_sample(sample, out_root / split / f"{f.stem}_{i:05d}.npz")
            made += 1
            if made % 20 == 0:
                print(f"  已处理 {made}/{len(files)}")
        except Exception as e:
            failed.append((f.name, f"{type(e).__name__}: {e}"))

    print(f"\n生成完成: {made} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败原因分布（前 10）:")
        for n, r in failed[:10]:
            print(f"  {n}: {r}")
    return made, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--meshes", type=str, default=None)
    ap.add_argument("--n-per-kind", type=int, default=25)
    ap.add_argument("--n-input", type=int, default=1024)
    ap.add_argument("--n-surface", type=int, default=4096)
    ap.add_argument("--n-interior", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    spec = SampleSpec(a.n_input, a.n_surface, a.n_interior)

    if a.synthetic:
        gen_synthetic(out, a.n_per_kind, spec)
    elif a.meshes:
        gen_from_meshes(Path(a.meshes), out, spec, a.limit)
    else:
        ap.error("需指定 --synthetic 或 --meshes")


if __name__ == "__main__":
    main()
