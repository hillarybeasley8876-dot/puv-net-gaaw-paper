"""核对 PU1K 训练 h5 与测试集的真实结构。只读，不修改任何数据。"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(r"E:\AE-CC托管\puv-net")
H5 = ROOT / "data" / "PU1K_extract" / "PU1K" / "train" / \
    "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"
TEST = ROOT / "data" / "PU1K_extract" / "PU1K" / "test" / "input_2048"
PUGAN = ROOT / "data" / "raw" / "PUGAN_poisson_256_poisson_1024.h5"


def dump_h5(path: Path, tag: str) -> None:
    print("=" * 78)
    print(f"{tag}: {path.name}")
    print("=" * 78)
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  /{name:<24} shape={str(obj.shape):<22} dtype={obj.dtype}")
        f.visititems(visit)
        keys = list(f.keys())
        for k in keys:
            if not isinstance(f[k], h5py.Dataset):
                continue
            arr = f[k]
            sub = arr[:1]
            print(f"  --- /{k} 首样本统计 ---")
            print(f"      min={np.min(sub):.6f} max={np.max(sub):.6f} "
                  f"mean={np.mean(sub):.6f}")
            r = np.linalg.norm(sub[0] - sub[0].mean(axis=0), axis=-1).max()
            print(f"      首样本去心后最大半径={r:.6f}")
    print()


def main() -> int:
    dump_h5(H5, "PU1K 训练集")
    dump_h5(PUGAN, "PU-GAN 训练集")

    print("=" * 78)
    print("PU1K 测试集 (input_2048) 抽样")
    print("=" * 78)
    inp = sorted((TEST / "input_2048").glob("*.xyz"))
    gt = sorted((TEST / "gt_8192").glob("*.xyz"))
    print(f"  input 文件数={len(inp)}  gt 文件数={len(gt)}")
    print(f"  文件名一一对应: {[p.name for p in inp] == [p.name for p in gt]}")
    a = np.loadtxt(inp[0])
    b = np.loadtxt(gt[0])
    print(f"  样本 {inp[0].name}")
    print(f"    input shape={a.shape}  gt shape={b.shape}  比例={len(b)/len(a):.1f}x")
    print(f"    input 列数={a.shape[1]} (3=仅坐标, 6=坐标+法向)")
    mesh_dir = TEST.parent / "original_meshes"
    print(f"  original_meshes 数量={len(list(mesh_dir.glob('*.off')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
