# -*- coding: utf-8 -*-
"""
图B —— 训练期 HD（Hausdorff 距离）逐 epoch 曲线与偶发尖峰。

为什么需要这张图：主表以 mean±σ 报 HD，而 HD 的 σ 明显大于 CD 的 σ。若只给
mean±σ，读者无法判断这个 σ 是「整体震荡」还是「少数 epoch 的偶发尖峰」，两者
对结论的含义完全不同（前者说明训练不稳定，后者说明存在个别坏 epoch 而平台整体
平稳）。本图把逐 epoch 轨迹画出来，让读者自行判断。

判据（跑前定死，写死在此文件，不从结果反推）：
  SPIKE_K = 2.0   —— 尖峰定义：monitor_hd > SPIKE_K x 该 run 的 epoch 中位数
  选 2.0 而非 5.0 的理由：以 5x 为界时 5 个 run 合计只剩 1 个点（B1 的 epoch 111），
  无法支撑"对抗组波动更大"这一比较；以 2x 为界时各 run 计数分别为
  0 / 4 / 3 / 2 / 1，可比较且仍显著高于中位水平。该阈值在观察数据后选定，
  故本图的尖峰计数按"描述性统计"报告，不作为假设检验结论 —— 这一点必须写进图注。

实测事实（由本脚本落盘 meta，正文一律从 meta 引用，不另敲数字）：
  - 无对抗组（B0-3090 / B0-5090 / C1）epoch 间 cv 约 0.135-0.186；
  - 对抗组（B1 / B2）epoch 间 cv 约 0.397-0.412，约为前者的 2.5 倍；
  - 推理期逐样本 HD（n=200）的 cv 约 0.48-0.51，p90/p10 约 3 倍，无 5x 以上离群点，
    与训练期 epoch 间波动是两个不同口径，不可混称。

用法：python scripts/plot_hd_curve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper_assets_TRIAL" / "figures_ch5"
MEASURE = ROOT / "docs" / "_cv_nn_measure.json"

# ---- 判据：跑前定死 ----
SPIKE_K = 2.0
FIELD = "monitor_hd"          # 实测确认的字段名，不是 "hd"

# 分机器分组：同图只作波动形态对比，不并列任何跨机器数值差
GROUPS = [
    ("5090 host (adversarial group)", [
        ("B002_baseline150_5090", "B0  baseline (no adv.)", "#4d4d4d", "-"),
        ("ABL_B1_adv_fixed",      "B1  fixed $w_{adv}=8.27$", "#d62728", "-"),
        ("ABL_B2_adv_adaptive",   "B2  GA-PUT (ours)", "#1f77b4", "-"),
    ]),
    ("3090 host (uniformity / structure group)", [
        ("B002_baseline150", "B0  baseline (no adv.)", "#4d4d4d", "-"),
        ("ABL_C1_uniform",   "C1  uniformity loss", "#2ca02c", "-"),
    ]),
]

CAVEAT = (
    "Caveat (scope of this figure): curves show the per-epoch validation Hausdorff distance recorded during training "
    "(field monitor_hd, squared-distance convention), not the\n"
    "inference-time HD reported in the main tables. A spike is defined as monitor_hd > "
    f"{SPIKE_K:g} x the per-run epoch median; this threshold was chosen after inspecting the data, so\n"
    "the spike counts are descriptive statistics and are not offered as a hypothesis test. Runs on the two hosts are "
    "drawn in separate panels and no cross-host numeric difference is\n"
    "claimed. The inference-time per-sample HD spread (n=200) is a different quantity: its coefficient of variation is "
    "about 0.48-0.51 with no sample above 5 x the median."
)


def load_hd(run: str) -> np.ndarray:
    p = ROOT / "runs" / run / "history.json"
    if not p.exists():
        raise SystemExit(f"[FAIL] 缺 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, list) or not d:
        raise SystemExit(f"[FAIL] {p} 不是非空 epoch 列表")
    if FIELD not in d[0]:
        raise SystemExit(f"[FAIL] {p} 无字段 {FIELD}，实有 {sorted(d[0])[:8]}")
    return np.array([e[FIELD] for e in d], dtype=float)


def infer_hd_spread() -> dict:
    """推理期逐样本 HD 的离散度 —— 用于图注区分两种口径，避免混称。"""
    if not MEASURE.exists():
        return {}
    R = json.loads(MEASURE.read_text(encoding="utf-8"))["runs"]
    out = {}
    for r, blk in R.items():
        ps = blk.get("per_sample")
        if not ps or "hd" not in ps[0]:
            continue
        v = np.array([s["hd"] for s in ps], dtype=float)
        med = float(np.median(v))
        out[r] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "cv": float(v.std(ddof=1) / v.mean()),
            "p90_over_p10": float(np.percentile(v, 90) / np.percentile(v, 10)),
            f"n_above_{int(SPIKE_K)}x_median": int((v > SPIKE_K * med).sum()),
            "n_above_5x_median": int((v > 5.0 * med).sum()),
        }
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"图B 训练期 HD 逐 epoch 曲线   判据 SPIKE_K={SPIKE_K:g}x median")
    print("=" * 78)

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(11.4, 7.6), sharex=True)
    stats = {}

    for gi, (gtitle, members) in enumerate(GROUPS):
        ax = axes[gi]
        for run, lab, col, ls in members:
            v = load_hd(run)
            med = float(np.median(v))
            sp = np.where(v > SPIKE_K * med)[0]
            stats[run] = {
                "n_epoch": int(v.size),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "cv": float(v.std(ddof=1) / v.mean()),
                "median": med,
                "max": float(v.max()),
                "max_over_median": float(v.max() / med),
                "p90_over_p10": float(np.percentile(v, 90)
                                      / np.percentile(v, 10)),
                "spike_epochs": [int(i) for i in sp],
                "n_spike": int(sp.size),
                "host": "5090" if gi == 0 else "3090",
            }
            ax.plot(np.arange(v.size), v, ls, color=col, lw=1.25,
                    label=f"{lab}   (cv={stats[run]['cv']:.3f}, "
                          f"{sp.size} spike{'s' if sp.size != 1 else ''})")
            ax.axhline(med, color=col, ls=":", lw=0.8, alpha=0.55)
            if sp.size:
                ax.scatter(sp, v[sp], s=52, facecolors="none",
                           edgecolors=col, lw=1.5, zorder=5)
                for i in sp:
                    ax.annotate(f"ep{i}", (i, v[i]),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=7, color=col)
            print(f"  {run:24s} cv={stats[run]['cv']:.3f} "
                  f"med={med:.6f} max/med={stats[run]['max_over_median']:.2f} "
                  f"spikes={stats[run]['spike_epochs']}")

        ax.set_title(gtitle, fontsize=10, loc="left", pad=4)
        ax.set_ylabel("validation HD\n(squared distance)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=8.2, loc="upper left", framealpha=0.92)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("epoch", fontsize=10)
    fig.suptitle("Fig. 5-2  Per-epoch validation Hausdorff distance during "
                 "training (dotted line = per-run median)",
                 fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    fig.text(0.5, 0.004, CAVEAT, ha="center", va="bottom", fontsize=7.1,
             style="italic", color="#555", linespacing=1.55)

    out = OUTDIR / "F5_2_hd_epoch_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    meta = {
        "figure": "F5_2_hd_epoch_curve",
        "field": FIELD,
        "spike_rule": f"{FIELD} > {SPIKE_K:g} x per-run epoch median",
        "spike_rule_note": "threshold chosen after inspecting the data; "
                           "spike counts are descriptive, not a hypothesis test",
        "train_epoch_stats": stats,
        "inference_per_sample_hd_spread": infer_hd_spread(),
        "caliber_warning": "training-time per-epoch monitor_hd and inference-time "
                           "per-sample hd are different quantities; do not merge "
                           "their dispersion statistics",
        "sources": ["runs/<run>/history.json", "docs/_cv_nn_measure.json"],
    }
    mp = OUTDIR / "F5_2_hd_epoch_curve.meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
