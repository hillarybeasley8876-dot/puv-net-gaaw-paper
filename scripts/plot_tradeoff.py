# -*- coding: utf-8 -*-
"""
图C —— cv_nn 与 CD 的权衡散点（按 run 着色，分 host 分面板）。

为什么需要这张图：本文的核心定量事实不是"某个指标变好了"，而是"改动损失项会
在均匀性(cv_nn)与几何精度(CD)之间移动工作点"。主表逐行给数字无法呈现这种二维
权衡；本图把每个 run 画成 (cv_nn, CD) 平面上一个带 SE 误差棒的点，读者可直接
看出哪些改动是纯改善、哪些是以一个指标换另一个。

三条纪律（本图为落实红线而设计）：
  1. **跨机器红线**：3090 组与 5090 组画在两个独立面板，各自以本组 baseline 为
     原点参考。绝不把两组点画进同一坐标系 —— 那等于把不同机器的绝对值并列。
  2. **误差棒必画**：只画点会让读者以为所有差异都真实。SE 用逐样本标准误
     (se_sample)，并在图注声明这是**跨样本** SE 而非跨 seed SE。
  3. **数字全部来自 docs/_thesis_results.json**（全文唯一数字入口），本脚本不
     自行计算任何均值/SE，只做读取与绘制。

用法：python scripts/plot_tradeoff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper_assets_TRIAL" / "figures_ch5"
RESULTS = ROOT / "docs" / "_thesis_results.json"

# 面板标题 -> (组 key, 该组 baseline run)
PANELS = [
    ("3090 host — uniformity / structure ablations", "3090_uniform_structure"),
    ("5090 host — adversarial ablations", "5090_adversarial"),
]

STYLE = {
    "B002_baseline150":      ("B0  baseline", "#4d4d4d", "o", 92),
    "B002_baseline150_5090": ("B0  baseline", "#4d4d4d", "o", 92),
    "ABL_A1_cd_balance":     ("A1  CD balance", "#ff7f0e", "^", 74),
    "ABL_A2_cd_boost_bwd":   ("A2  CD boost bwd", "#8c564b", "v", 74),
    "ABL_C1_uniform":        ("C1  uniformity loss", "#2ca02c", "s", 74),
    "ABL_AC_combo":          ("AC  A1+C1 combo", "#9467bd", "D", 68),
    "ABL_D1_scale_qk":       ("D1  QK scaling (not a contribution)",
                              "#7f7f7f", "X", 74),
    "ABL_B1_adv_fixed":      ("B1  fixed $w_{adv}=8.27$", "#d62728", "P", 92),
    "ABL_B2_adv_adaptive":   ("B2  GA-PUT (ours)", "#1f77b4", "*", 190),
}

# 需要移出坐标轴的 run：其取值比同组其余点高出数倍，同轴绘制会把本文真正
# 关心的点全部压成左下角一团（实测 A2 的 cv_nn=0.563 是其余点的 2.4 倍，
# D1 的 CD=5.72e-3 是其余点的 5 倍）。改为标注 + meta 落数，不丢信息。
OFFSCALE = ("ABL_D1_scale_qk", "ABL_A2_cd_boost_bwd")

CAVEAT = (
    "Caveat (scope of this figure): error bars are cross-sample standard errors over the n=200 validation samples "
    "(se_sample), i.e. they describe how much the mean would move if a\n"
    "different draw of these 200 shapes were used. They are NOT across-seed standard errors: no across-seed variance "
    "is measured or reported anywhere in this thesis, so no claim of\n"
    "statistical significance across random initialisations is made. The two hosts are plotted in separate panels with "
    "their own baselines; absolute values are never compared across\n"
    "panels. All coordinates are read from docs/_thesis_results.json, the single authoritative numeric source for "
    "this thesis. A2 (CD boost bwd) and D1 (QK scaling, an explicitly\n"
    "non-novel reference point) lie several times outside the plotted range; they are annotated with their exact "
    "values instead of being drawn to scale, so that the remaining points stay legible."
)


def main() -> int:
    if not RESULTS.exists():
        raise SystemExit(f"[FAIL] 缺 {RESULTS}，请先跑 scripts/build_thesis_results.py")
    R = json.loads(RESULTS.read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("图C cv_nn / CD 权衡散点")
    print("=" * 78)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9))
    plotted = {}

    for ax, (title, gkey) in zip(axes, PANELS):
        grp = R["groups"][gkey]
        base = grp["baseline"]
        offscale = []
        xs, ys = [], []
        for run, blk in grp["runs"].items():
            m = blk["metrics"]
            x = m["cv_nn"]["mean"]
            y = m["cd"]["mean"]
            xe = m["cv_nn"].get("se_sample")
            ye = m["cd"].get("se_sample")
            lab, col, mk, sz = STYLE.get(run, (run, "#333", "o", 60))
            if run in OFFSCALE:
                offscale.append((run, lab, x, y))
                continue
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none",
                        ecolor=col, elinewidth=1.1, capsize=2.6, alpha=0.85,
                        zorder=2)
            ax.scatter(x, y, s=sz, c=col, marker=mk, edgecolors="white",
                       linewidths=0.7, label=lab, zorder=3)
            xs.append(x); ys.append(y)
            plotted[run] = {"cv_nn": x, "cd": y,
                            "cv_nn_se": xe, "cd_se": ye,
                            "host": gkey, "is_baseline": run == base}
            print(f"  {run:24s} cv_nn={x:.6f}±{xe:.6f}  cd={y:.6e}±{ye:.1e}")

        # 以本组 baseline 为十字参考线：读者可直接看象限
        bm = grp["runs"][base]["metrics"]
        ax.axvline(bm["cv_nn"]["mean"], color="#999", ls="--", lw=0.9,
                   alpha=0.8, zorder=1)
        ax.axhline(bm["cd"]["mean"], color="#999", ls="--", lw=0.9,
                   alpha=0.8, zorder=1)
        # cv_nn 越小越均匀 -> 箭头必须指向左（早期版本写成向右，与坐标含义相反）
        ax.annotate("$\\leftarrow$ better uniformity", xy=(0.02, 0.965),
                    xycoords="axes fraction", fontsize=8, color="#666")
        ax.annotate("$\\downarrow$ better geometry", xy=(0.02, 0.055),
                    xycoords="axes fraction", fontsize=8, color="#666")

        if offscale:
            lines = ["off-scale (annotated, not to scale):"]
            for run, lab, x, y in offscale:
                lines.append(f"  {lab}:  cv$_{{nn}}$={x:.3f},  CD={y:.2e}")
                plotted[run] = {"cv_nn": x, "cd": y, "host": gkey,
                                "off_scale": True}
                print(f"  {run:24s} OFF-SCALE cv_nn={x:.6f} cd={y:.6e}")
            ax.annotate("\n".join(lines), xy=(0.975, 0.955),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=7.5, color="#666",
                        bbox=dict(boxstyle="round,pad=0.36", fc="white",
                                  ec="#bbb", lw=0.8))

        ax.set_title(title, fontsize=10.2, loc="left", pad=6)
        ax.set_xlabel("$\\mathrm{cv}_{nn}$  (nearest-neighbour spacing "
                      "coefficient of variation; lower = more uniform)",
                      fontsize=9)
        ax.set_ylabel("CD  (bidirectional Chamfer distance, squared; "
                      "lower = better)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.42)
        ax.legend(fontsize=8.0, loc="lower right", framealpha=0.93)

    # 图号跟随正稿章号：权衡平面属第 6 章 §6.5.4 → 图 6.2。
    # （文件名沿用 F5_3_* 不改，避免打断既有 .meta.json 与引用路径。）
    fig.suptitle("Fig. 6-2  Uniformity / geometry trade-off across ablations "
                 "(dashed cross = per-host baseline)", fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0.135, 1, 0.955))
    fig.text(0.5, 0.004, CAVEAT, ha="center", va="bottom", fontsize=7.1,
             style="italic", color="#555", linespacing=1.55)

    out = OUTDIR / "F5_3_tradeoff_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    meta = {
        "figure": "F5_3_tradeoff_scatter",
        "axes": {"x": "cv_nn mean (n=200)", "y": "cd mean (n=200)"},
        "errorbars": "cross-sample se_sample over the 200 validation samples; "
                     "NOT across-seed SE",
        "panels": [{"title": t, "group": g,
                    "baseline": R["groups"][g]["baseline"]}
                   for t, g in PANELS],
        "cross_host_rule": "separate panels; absolute values never compared "
                           "across panels",
        "off_scale": list(OFFSCALE),
        "points": plotted,
        "source": "docs/_thesis_results.json",
    }
    mp = OUTDIR / "F5_3_tradeoff_scatter.meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
