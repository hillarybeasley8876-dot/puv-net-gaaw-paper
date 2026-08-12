"""数据集与 DataLoader。

数据集在离线阶段由 scripts/prepare_data.py 生成为 npz，训练时只做读取，
不在训练循环里做 mesh.contains（那个太慢，会让 GPU 空转）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumetricPCDataset(Dataset):
    """读取预处理好的体积点云样本。

    返回的 tensor 均为 (3, N) 布局，与模型的 (B, 3, N) 约定一致，
    避免在训练循环里反复 transpose。
    """

    def __init__(self, root: str | Path, split: str = "train",
                 augment: bool = False, seed: int = 0):
        self.root = Path(root)
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"{self.root/split} 下没有 npz 样本。先运行 scripts/prepare_data.py")
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.files[i])
        inp = d["input"].astype(np.float32)
        surf = d["surface_gt"].astype(np.float32)
        inter = d["interior_gt"].astype(np.float32)
        nrm = d["input_normals"].astype(np.float32)

        if self.augment:
            # 随机旋转：输入、真值、法线必须用同一个 R，否则监督信号错位。
            # 这是数据增强最容易出错的地方。
            R = _random_rotation(self.rng).astype(np.float32)
            inp, surf, inter = inp @ R.T, surf @ R.T, inter @ R.T
            nrm = nrm @ R.T

        return {
            "input": torch.from_numpy(inp.T.copy()),
            "input_normals": torch.from_numpy(nrm.T.copy()),
            "surface_gt": torch.from_numpy(surf.T.copy()),
            "interior_gt": torch.from_numpy(inter.T.copy()),
            "name": self.files[i].stem,
        }


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """均匀随机旋转矩阵（QR 分解法，保证 det=+1 不含反射）。"""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def collate(batch: list[dict]) -> dict:
    """默认 collate 对 name 字段不友好，这里显式处理。"""
    out = {}
    for k in batch[0]:
        if k == "name":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
