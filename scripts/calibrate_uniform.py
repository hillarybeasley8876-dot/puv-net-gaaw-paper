# -*- coding: utf-8 -*-
"""改进 C 的 w_uniform 定标 —— 唯一无实测依据的超参，必须定死。

问题：C1 冒烟用 w_uniform=1.0，实测 cd=0.070 vs baseline 0.013 = 差 5.4 倍。
      uniform 项把 CD 完全压垮了，该组结论会变成"加了 uniform 就崩"，
      测不到"uniform 是否改善均匀性"这个真问题。

定标原则（与 B-001 的 M1 对抗权重同一套逻辑）：
    让 uniform 项的**梯度范数**约为 CD 项梯度范数的 target_ratio 倍。
    取 target_ratio = 0.1，与 adv_target_ratio 一致 —— 保持两个改进项
    的"辅助项强度"可比，避免 C 组因权重更激进而占优/吃亏。

⚠️ 为什么不用 loss 值比而用梯度范数比：
    loss 值比只反映"目标函数里占多少"，但优化实际吃的是梯度。
    B-001 已踩过这个坑：w_adv=1.0 时 loss 看着不小，
    实测对抗梯度只有 CD 的 1/82.65，等于没开对抗。

本脚本在 CPU 上跑（不与正在训练的 B-002 抢 GPU）。

输出：runs/ablation_design/uniform_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import PUTrainDataset
from puvnet.losses.upsampling import chamfer_loss_split
from puvnet.models.pu_gan import uniform_loss
from puvnet.models.pu_transformer import PUTransformer

OUT = ROOT / "runs" / "ablation_design"
TARGET_RATIO = 0.1          # 与 adv_target_ratio 一致
SEED = 20260811


def grad_norm(loss: torch.Tensor, params) -> float:
    g = torch.autograd.grad(loss, params, retain_graph=True,
                            allow_unused=True)
    tot = 0.0
    for t in g:
        if t is not None:
            tot += float((t ** 2).sum().item())
    return tot ** 0.5


def main() -> int:
    torch.manual_seed(SEED)
    dev = "cpu"           # 明确用 CPU：B-002 正占用 GPU
    print("=" * 72)
    print("改进 C（uniform 项）权重定标  [CPU，避免与 B-002 抢 GPU]")
    print("=" * 72)

    # 真实数据（不用随机点云 —— uniform_loss 对点分布极敏感，
    # 随机高斯点的均匀性与真实 patch 差别很大，定标会偏）
    ds = PUTrainDataset(source="pu1k", up_ratio=4, augment=False,
                        limit=64, noise_beta=0.0, seed=SEED)
    print(f"数据：{len(ds)} 个真实 patch")

    model = PUTransformer(up_ratio=4, dims=[32, 64, 128, 256, 256], k=20,
                          psi=4, head_dim=32, mlp_ratio=1,
                          attention_type="sc-msa", tail_mode="shuffle",
                          scale_qk=False).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"模型参数：{sum(p.numel() for p in params):,}")

    # 取若干 batch 平均，单 batch 的梯度范数波动很大
    n_batch, bs = 4, 4
    ratios, cd_gs, uni_gs, cd_vs, uni_vs = [], [], [], [], []
    print(f"\n{'batch':>6} {'CD值':>12} {'uni值':>12} "
          f"{'|g_CD|':>12} {'|g_uni|':>12} {'g比':>10}")
    for b in range(n_batch):
        idx = list(range(b * bs, (b + 1) * bs))
        inp = torch.stack([ds[i][0] for i in idx]).to(dev)
        gt = torch.stack([ds[i][1] for i in idx]).to(dev)
        pred = model(inp)

        fwd, bwd = chamfer_loss_split(pred, gt, squared=True)
        l_cd = fwd + bwd
        l_uni = uniform_loss(pred, percentages=(0.004, 0.006, 0.008,
                                                0.010, 0.012)).mean()
        g_cd = grad_norm(l_cd, params)
        g_uni = grad_norm(l_uni, params)
        r = g_uni / g_cd if g_cd > 0 else float("nan")
        ratios.append(r); cd_gs.append(g_cd); uni_gs.append(g_uni)
        cd_vs.append(float(l_cd.item())); uni_vs.append(float(l_uni.item()))
        print(f"{b:>6} {l_cd.item():>12.6f} {l_uni.item():>12.6f} "
              f"{g_cd:>12.4e} {g_uni:>12.4e} {r:>10.4f}")

    r_mean = sum(ratios) / len(ratios)
    w = TARGET_RATIO / r_mean
    print(f"\n未加权时 |g_uni| / |g_CD| 均值 = {r_mean:.4f}")
    print(f"目标比 = {TARGET_RATIO}")
    print(f"→ w_uniform = {TARGET_RATIO} / {r_mean:.4f} = {w:.4f}")

    # 对比：w=1.0 时实际比是多少（解释冒烟为何崩）
    print(f"\n[解释冒烟失败] w_uniform=1.0 时 uniform 梯度 = CD 梯度的 "
          f"{r_mean:.2f} 倍")
    if r_mean > 1:
        print(f"  → uniform 项梯度比 CD 大 {r_mean:.1f} 倍，"
              f"优化被均匀性主导，CD 必然恶化（实测 cd 0.070 vs 0.013）")
    print(f"  → 定标后 w={w:.4f}，uniform 梯度降为 CD 的 {TARGET_RATIO:.0%}")

    # 舍入到便于书写的值，并核对舍入后的实际比
    w_round = float(f"{w:.3g}")
    r_after = r_mean * w_round
    print(f"\n[取值] w_uniform = {w_round}（舍入自 {w:.6f}）")
    print(f"       舍入后实际梯度比 = {r_after:.4f}（目标 {TARGET_RATIO}）")

    verdict = {
        "target_ratio": TARGET_RATIO,
        "target_ratio_source": "与 adv_target_ratio 一致，保持辅助项强度可比",
        "grad_ratio_unweighted_mean": r_mean,
        "grad_ratio_per_batch": ratios,
        "cd_grad_norms": cd_gs, "uniform_grad_norms": uni_gs,
        "cd_values": cd_vs, "uniform_values": uni_vs,
        "w_uniform_exact": w, "w_uniform_rounded": w_round,
        "grad_ratio_after_rounding": r_after,
        "why_1p0_failed": (f"w=1.0 时 uniform 梯度为 CD 的 {r_mean:.2f} 倍，"
                           f"优化被均匀性主导；冒烟实测 cd=0.070 vs baseline 0.013"),
        "method_note": ("用梯度范数比而非 loss 值比定标 —— "
                        "B-001 已证实 loss 值比会严重误判（w_adv=1.0 时"
                        "对抗梯度仅为 CD 的 1/82.65）"),
        "device": dev, "n_batch": n_batch, "batch_size": bs, "seed": SEED,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "uniform_calibration.json"
    dst.write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
