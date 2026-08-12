"""PU1K / PU-GAN 数据管线。

实测规格（scripts/inspect_data.py 于 2026-08-11 核实，非文献抄录）
------------------------------------------------------------------
PU1K  train h5 : /poisson_256 (69000, 256, 3) + /poisson_1024 (69000, 1024, 3)
PU-GAN train h5: /poisson_256 (24000, 256, 3) + /poisson_1024 (24000, 1024, 3)
PU1K  test     : test/input_2048/{input_2048,gt_8192}/*.xyz  127 对，2048 -> 8192
PU1K  meshes   : test/original_meshes/*.off  127 个（P2F 用，无需另下 3.2GB 包）

⚠️ 关键实测坑：两个训练集的坐标尺度差约 40 倍
    PU1K   首样本去心后最大半径 = 0.062
    PU-GAN 首样本去心后最大半径 = 2.49
  => 必须【每个 patch 独立归一化】。若不归一化直接混训，
     两数据集的 loss 与梯度尺度会差两个数量级，表现为莫名不收敛且极难排查。

协议开关（SOTA_SURVEY 记录的领域陷阱）
--------------------------------------
PU-GCN README 原文警告：
  "If you favor uniform inputs, you have to retrain all models.
   Otherwise, the results might be really bad."
=> 输入下采样协议 random / fps 必须显式声明并在论文中写明，绝不混用。
   本项目主实验统一 random（与 PU-GCN/Grad-PU/RepKPU 默认一致）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import os

# 数据根目录。默认为本机路径, 但允许环境变量 PUVNET_ROOT 覆盖 ——
# 云端 Linux (5090) 上路径为 /root/puv-net, 硬编码 Windows 路径会直接失效。
# 不改默认值是为了保证本机既有 run 的行为逐字不变。
ROOT = Path(os.environ.get("PUVNET_ROOT") or r"E:\AE-CC托管\puv-net")
PU1K_DIR = ROOT / "data" / "PU1K_extract" / "PU1K"
PU1K_TRAIN_H5 = PU1K_DIR / "train" / \
    "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"
PUGAN_TRAIN_H5 = ROOT / "data" / "raw" / "PUGAN_poisson_256_poisson_1024.h5"
PU1K_TEST_DIR = PU1K_DIR / "test"
PU1K_MESH_DIR = PU1K_TEST_DIR / "original_meshes"

Protocol = Literal["random", "fps"]


# --------------------------------------------------------------------------
# 归一化：统一用「去心 + 最大半径」，与 metrics.normalize_point_cloud 一致
# --------------------------------------------------------------------------
def normalize_patch(pc: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """去心并缩放到单位球（按最大半径，非包围盒对角线）。

    返回 (归一化点云, center, scale)，便于评测时还原到原尺度。
    """
    center = pc.mean(axis=0, keepdims=True)
    centered = pc - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-12:
        scale = 1e-12
    return centered / scale, center, scale


def farthest_point_sample(pc: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """FPS 下采样，返回索引。纯 numpy 实现，仅用于协议对照，不进训练热路径。"""
    m = len(pc)
    if n >= m:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    idx = np.zeros(n, dtype=np.int64)
    idx[0] = rng.integers(m)
    dist = np.linalg.norm(pc - pc[idx[0]], axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(dist))
        d = np.linalg.norm(pc - pc[idx[i]], axis=1)
        dist = np.minimum(dist, d)
    return idx


def downsample(pc: np.ndarray, n: int, protocol: Protocol,
               seed: int = 0) -> np.ndarray:
    """按指定协议把点云降到 n 点。"""
    m = len(pc)
    if n >= m:
        return pc
    if protocol == "random":
        rng = np.random.default_rng(seed)
        sel = rng.choice(m, size=n, replace=False)
    elif protocol == "fps":
        sel = farthest_point_sample(pc, n, seed=seed)
    else:
        raise ValueError(f"未知协议: {protocol}")
    return pc[sel]


def add_gaussian_noise(pc: np.ndarray, beta: float, seed: int = 0) -> np.ndarray:
    """按比例 beta 注入高斯噪声（噪声鲁棒性实验用）。

    约定：pc 已归一化到单位球，故 sigma = beta * 1.0（半径）。
    与 PU-Transformer 论文 Tab.4 的 beta = 0.5% / 1% / 2% 对齐。
    """
    if beta <= 0:
        return pc
    rng = np.random.default_rng(seed)
    return pc + rng.standard_normal(pc.shape).astype(pc.dtype) * beta


# --------------------------------------------------------------------------
# 训练集
# --------------------------------------------------------------------------
class PUTrainDataset(Dataset):
    """PU1K / PU-GAN 训练 patch 数据集（256 -> 1024，即 4x）。

    参数
    ----
    source     : 'pu1k' | 'pugan' | 'both'
    up_ratio   : 上采样倍率，用于从 gt 中取前 r*n_in 点（4x 时正好用满 1024）
    noise_beta : 训练时注入噪声的比例，0 = 不注入
    augment    : 是否做随机旋转 + 镜像 + 尺度扰动
    limit      : 只取前 N 个样本（小规模跑通用），None = 全量
    """

    def __init__(self, source: str = "pu1k", up_ratio: int = 4,
                 noise_beta: float = 0.0, augment: bool = True,
                 limit: int | None = None, seed: int = 0) -> None:
        self.up_ratio = up_ratio
        self.noise_beta = noise_beta
        self.augment = augment
        self.seed = seed

        paths: list[Path] = []
        if source in ("pu1k", "both"):
            paths.append(PU1K_TRAIN_H5)
        if source in ("pugan", "both"):
            paths.append(PUGAN_TRAIN_H5)
        if not paths:
            raise ValueError(f"未知 source: {source}")

        inputs, gts = [], []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"训练集不存在: {p}")
            with h5py.File(p, "r") as f:
                inputs.append(np.asarray(f["poisson_256"], dtype=np.float32))
                gts.append(np.asarray(f["poisson_1024"], dtype=np.float32))
        self.inputs = np.concatenate(inputs, axis=0)
        self.gts = np.concatenate(gts, axis=0)

        if limit is not None:
            self.inputs = self.inputs[:limit]
            self.gts = self.gts[:limit]

        n_in = self.inputs.shape[1]
        need = n_in * up_ratio
        if need > self.gts.shape[1]:
            raise ValueError(
                f"up_ratio={up_ratio} 需要 gt {need} 点，但 h5 只有 "
                f"{self.gts.shape[1]} 点")
        self.n_gt = need

    def __len__(self) -> int:
        return len(self.inputs)

    def _augment(self, inp: np.ndarray, gt: np.ndarray, rng):
        # 随机绕任意轴旋转：input 与 gt 必须用同一个矩阵
        axis = rng.standard_normal(3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        ang = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = axis
        R = np.array([
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ], dtype=np.float32)
        inp, gt = inp @ R.T, gt @ R.T
        # 随机镜像
        flip = rng.integers(0, 2, size=3) * 2 - 1
        inp, gt = inp * flip, gt * flip
        # 轻微尺度扰动（归一化后再做，保持在单位球附近）
        sc = np.float32(rng.uniform(0.9, 1.1))
        return (inp * sc).astype(np.float32), (gt * sc).astype(np.float32)

    def __getitem__(self, i: int):
        inp = self.inputs[i]
        gt = self.gts[i][: self.n_gt]

        # 关键：input 与 gt 必须用【同一组 center/scale】归一化，
        # 否则两者不在同一坐标系，loss 无意义。以 gt 为基准。
        _, center, scale = normalize_patch(gt)
        inp = ((inp - center) / scale).astype(np.float32)
        gt = ((gt - center) / scale).astype(np.float32)

        rng = np.random.default_rng(self.seed + i)
        if self.augment:
            inp, gt = self._augment(inp, gt, rng)
        if self.noise_beta > 0:
            inp = add_gaussian_noise(inp, self.noise_beta,
                                     seed=self.seed + i).astype(np.float32)

        return torch.from_numpy(inp), torch.from_numpy(gt)


# --------------------------------------------------------------------------
# 测试集
# --------------------------------------------------------------------------
class PU1KTestSet:
    """PU1K 测试集：127 个模型，input_2048 -> gt_8192，附 original_meshes。

    不继承 Dataset —— 评测是整模型 patch 化推理，不走 DataLoader 批处理。
    """

    def __init__(self, input_n: int = 2048, up_ratio: int = 4) -> None:
        self.input_dir = PU1K_TEST_DIR / f"input_{input_n}" / f"input_{input_n}"
        self.gt_dir = PU1K_TEST_DIR / f"input_{input_n}" / \
            f"gt_{input_n * up_ratio}"
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"测试输入目录不存在: {self.input_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"测试 GT 目录不存在: {self.gt_dir}")
        self.names = sorted(p.stem for p in self.input_dir.glob("*.xyz"))

    def __len__(self) -> int:
        return len(self.names)

    def mesh_path(self, name: str) -> Path:
        return PU1K_MESH_DIR / f"{name}.off"

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        inp = np.loadtxt(self.input_dir / f"{name}.xyz", dtype=np.float32)
        gt = np.loadtxt(self.gt_dir / f"{name}.xyz", dtype=np.float32)
        return inp[:, :3], gt[:, :3]


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def self_check() -> bool:
    ok = True
    print("=" * 74)
    print("pu_dataset 自检")
    print("=" * 74)

    # 1. 归一化正确性
    rng = np.random.default_rng(0)
    pc = rng.standard_normal((500, 3)).astype(np.float32) * 7.0 + 3.0
    norm, c, s = normalize_patch(pc)
    r = float(np.max(np.linalg.norm(norm, axis=1)))
    print(f"[1] 归一化最大半径 = {r:.12f}  (期望 1.0)")
    ok &= abs(r - 1.0) < 1e-5
    back = norm * s + c
    err = float(np.max(np.abs(back - pc)))
    print(f"    还原误差 = {err:.3e}  (期望 ~0)")
    ok &= err < 1e-3

    # 2. 两数据集尺度差异 -> 归一化后应一致
    for tag, path, key in (("PU1K", PU1K_TRAIN_H5, "poisson_1024"),
                           ("PU-GAN", PUGAN_TRAIN_H5, "poisson_1024")):
        with h5py.File(path, "r") as f:
            sample = np.asarray(f[key][0], dtype=np.float32)
        raw_r = float(np.max(np.linalg.norm(sample - sample.mean(0), axis=1)))
        n, _, _ = normalize_patch(sample)
        new_r = float(np.max(np.linalg.norm(n, axis=1)))
        print(f"[2] {tag:<7} 原始半径={raw_r:>8.4f}  归一化后={new_r:.6f}")
        ok &= abs(new_r - 1.0) < 1e-5

    # 3. 协议开关：random vs fps 应给出不同子集
    pc2 = rng.standard_normal((1000, 3)).astype(np.float32)
    a = downsample(pc2, 200, "random", seed=1)
    b = downsample(pc2, 200, "fps", seed=1)
    same = np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))
    print(f"[3] random 与 fps 子集相同? {same}  (期望 False)")
    ok &= not same
    # fps 应该覆盖更大范围
    print(f"    random 跨度={np.ptp(a, axis=0).mean():.4f}  "
          f"fps 跨度={np.ptp(b, axis=0).mean():.4f}  (fps 应更大)")
    ok &= np.ptp(b, axis=0).mean() >= np.ptp(a, axis=0).mean()

    # 4. 噪声注入量级
    unit = pc2 / np.max(np.linalg.norm(pc2, axis=1))
    noisy = add_gaussian_noise(unit, 0.01, seed=0)
    disp = float(np.mean(np.linalg.norm(noisy - unit, axis=1)))
    print(f"[4] beta=1% 平均位移 = {disp:.6f}  (期望 ~0.017 = 0.01*sqrt(3)*0.98)")
    ok &= 0.005 < disp < 0.03

    # 5. 训练集加载 + input/gt 同坐标系
    ds = PUTrainDataset(source="pu1k", limit=8, augment=False, seed=0)
    inp, gt = ds[0]
    print(f"[5] 训练样本 input={tuple(inp.shape)} gt={tuple(gt.shape)}  "
          f"len={len(ds)}")
    ok &= tuple(inp.shape) == (256, 3) and tuple(gt.shape) == (1024, 3)
    gt_r = float(gt.norm(dim=1).max())
    inp_r = float(inp.norm(dim=1).max())
    print(f"    gt 最大半径={gt_r:.6f} (期望 1.0)  input 最大半径={inp_r:.6f}")
    ok &= abs(gt_r - 1.0) < 1e-4
    # input 是 gt 的子采样版本，半径应接近但不超过太多
    ok &= inp_r < 1.2

    # 6. 增强后 input/gt 仍在同一坐标系（用 CD 近似检验）
    ds_aug = PUTrainDataset(source="pu1k", limit=8, augment=True, seed=0)
    inp_a, gt_a = ds_aug[0]
    from puvnet.metrics.pointcloud import chamfer_distance
    cd_aligned = chamfer_distance(inp_a.numpy(), gt_a.numpy())
    cd_cross = chamfer_distance(inp_a.numpy(), gt.numpy())
    print(f"[6] 增强后 CD(inp_aug, gt_aug)={cd_aligned:.6f}  "
          f"CD(inp_aug, gt_orig)={cd_cross:.6f}")
    print("    期望：同坐标系的远小于跨坐标系的")
    ok &= cd_aligned < cd_cross

    # 7. 测试集
    ts = PU1KTestSet()
    inp_t, gt_t = ts.load(ts.names[0])
    mesh_ok = ts.mesh_path(ts.names[0]).exists()
    print(f"[7] 测试集 n={len(ts)}  input={inp_t.shape} gt={gt_t.shape}  "
          f"mesh存在={mesh_ok}")
    ok &= len(ts) == 127
    ok &= inp_t.shape == (2048, 3) and gt_t.shape == (8192, 3)
    ok &= mesh_ok

    print()
    print(f"自检结果: {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
