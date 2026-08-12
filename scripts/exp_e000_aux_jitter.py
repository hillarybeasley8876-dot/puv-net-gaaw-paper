"""E-000 附属诊断：为什么 jitter_large 的 CD 优于 jitter_small？（E3 判据 FAIL 的根因排查）

背景
----
E-000 v2 预注册判据 E3（"噪声越大 CD 越差"）实测 FAIL：
    jitter_small (sigma=0.005) CD = 1.571667e-3
    jitter_large (sigma=0.020) CD = 1.123442e-3      <-- 反而更好

这与直觉冲突。三种可能：
  H-a  评测管线仍有 bug（量纲/归一化）
  H-b  copy4 基线本身病态：4 份重合点使 pred->gt 项极小、gt->pred 项极大；
       抖动把重合点散开，恰好"修复"了 gt->pred 覆盖，故一定范围内噪声越大越好
  H-c  E3 判据本身设错：它假设"噪声单调恶化"，但在【复制型上采样】这个特殊基线上，
       噪声兼具"破坏精度"与"改善覆盖"两种相反作用，非单调是应有行为

本脚本做的事：把 CD 拆成 forward(pred->gt) 与 backward(gt->pred) 两项，
扫一串 sigma，看两项各自的走势。这能直接区分 H-a / H-b / H-c。

诚信约束
--------
本脚本只做机理诊断，不修改任何判据，也不为了让 E3 通过而调参。
输出只报事实与分项数值，结论由人来下。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import PU1KTestSet  # noqa: E402
from puvnet.metrics.pointcloud import normalize_point_cloud, _nn_dist  # noqa: E402

OUT_DIR = ROOT / "runs" / "E000_metric_calibration"
SEED = 20260811
N_MODELS = 8
SIGMAS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]


def cd_split(pred: np.ndarray, gt: np.ndarray) -> dict:
    """按官方协议（各自独立归一化 + 平方距离）算 CD，并返回双向分项。"""
    pn, _, _ = normalize_point_cloud(pred)
    gn, _, _ = normalize_point_cloud(gt)
    d_fwd = _nn_dist(pn, gn) ** 2      # pred -> gt : 精度项（点是否贴在真值上）
    d_bwd = _nn_dist(gn, pn) ** 2      # gt -> pred : 覆盖项（真值是否被覆盖到）
    return {
        "fwd": float(d_fwd.mean()),
        "bwd": float(d_bwd.mean()),
        "cd": float(d_fwd.mean() + d_bwd.mean()),
    }


def main() -> int:
    ts = PU1KTestSet(input_n=2048, up_ratio=4)
    names = ts.names[:N_MODELS]

    print("=" * 96)
    print("E-000 附属诊断：CD 双向分项 vs 抖动强度（复制型上采样基线）")
    print("=" * 96)
    print(f"模型数={len(names)}  seed={SEED}")
    print("fwd = pred->gt（精度项）  bwd = gt->pred（覆盖项）  单位 ×1e-3")
    print()
    print(f"{'sigma':>8} | {'fwd':>12} | {'bwd':>12} | {'CD':>12} | {'bwd占比':>8}")
    print("-" * 96)

    table = []
    for sig in SIGMAS:
        acc = {"fwd": [], "bwd": [], "cd": []}
        for i, name in enumerate(names):
            inp, gt = ts.load(name)
            rng = np.random.default_rng(SEED + i)
            pred = np.repeat(inp, 4, axis=0).astype(np.float64)
            if sig > 0:
                pred = pred + rng.standard_normal(pred.shape) * sig
            s = cd_split(pred, gt)
            for k in acc:
                acc[k].append(s[k])
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        row["sigma"] = sig
        row["bwd_share"] = row["bwd"] / row["cd"] if row["cd"] > 0 else float("nan")
        table.append(row)
        print(f"{sig:>8.4f} | {row['fwd']*1e3:>12.6f} | {row['bwd']*1e3:>12.6f} | "
              f"{row['cd']*1e3:>12.6f} | {row['bwd_share']*100:>7.2f}%")

    print("-" * 96)

    # --- 事实陈述（不做判据）---
    fwd = [r["fwd"] for r in table]
    bwd = [r["bwd"] for r in table]
    cd = [r["cd"] for r in table]
    fwd_mono = all(fwd[i] <= fwd[i + 1] + 1e-15 for i in range(len(fwd) - 1))
    bwd_mono_dec = all(bwd[i] >= bwd[i + 1] - 1e-15 for i in range(len(bwd) - 1))
    i_best = int(np.argmin(cd))

    print()
    print("观测事实：")
    print(f"  1. fwd（精度项）随 sigma 单调不减? {fwd_mono}   "
          f"[{fwd[0]*1e3:.6f} -> {fwd[-1]*1e3:.6f}]")
    print(f"  2. bwd（覆盖项）随 sigma 单调不增? {bwd_mono_dec}  "
          f"[{bwd[0]*1e3:.6f} -> {bwd[-1]*1e3:.6f}]")
    print(f"  3. CD 最小值出现在 sigma={table[i_best]['sigma']}  "
          f"(CD={cd[i_best]*1e3:.6f}e-3) —— "
          f"{'内部极小值（非单调）' if 0 < i_best < len(cd)-1 else '端点'}")
    print(f"  4. sigma=0 时 bwd 占 CD 的 {table[0]['bwd_share']*100:.2f}%")

    payload = {
        "experiment": "E-000_aux_jitter_diagnosis",
        "purpose": "排查 E3 判据 FAIL 的根因，不修改判据",
        "seed": SEED, "n_models": len(names), "sigmas": SIGMAS,
        "protocol": "PU-GCN official: squared distance, independent normalization",
        "table": table,
        "facts": {
            "fwd_monotonic_increasing": bool(fwd_mono),
            "bwd_monotonic_decreasing": bool(bwd_mono_dec),
            "cd_argmin_sigma": table[i_best]["sigma"],
            "cd_has_interior_minimum": bool(0 < i_best < len(cd) - 1),
            "bwd_share_at_sigma0": table[0]["bwd_share"],
        },
        "note": "本脚本只报事实，结论写入 docs/EXPERIMENT_LOG.md 由人判定",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "aux_jitter_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n已落盘: {OUT_DIR / 'aux_jitter_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
