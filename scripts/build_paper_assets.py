# -*- coding: utf-8 -*-
"""论文素材汇总器 —— 把所有已完成 run 汇总成跨组对比图 + 主表。

设计约束（与项目铁律一致）:
  * 所有数字只从各 run 落盘的 summary_stats.json / metrics.json 读，
    脚本内不出现任何手填实验数值
  * 门槛与接受准则只从 runs/ablation_design/ablation_matrix.json 读（无魔数）
  * 每张图由 puvnet.viz._save 自动落同名 .data.json（数字可追溯）
  * 未完成的组自动跳过并列出，不用占位数字凑表

用法:
    python scripts/build_paper_assets.py
    python scripts/build_paper_assets.py --outdir paper_assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from puvnet.metrics.selection import convergence_check, plateau_stats  # noqa: E402
from puvnet.viz.visualize import (plot_ablation_bars, plot_metric_curves,  # noqa: E402
                                 _save)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DESIGN = ROOT / "runs" / "ablation_design" / "ablation_matrix.json"
BASE_RUN = "B002_baseline150"
METRICS = ("cd", "hd", "nuc")
MNAME = {"cd": "CD", "hd": "HD", "nuc": "NUC"}
# 论文里 CD/HD 惯例放大 1e3 展示；NUC 本身是比例量，不缩放
SCALE = {"cd": 1e3, "hd": 1e3, "nuc": 1.0}


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict | None:
    """读一个 run 的平台区统计 + 逐 epoch 曲线。

    summary_stats.json 缺失时（如 B-001 跑在选点器接入之前），用同一个
    plateau_stats() 对其 metrics.json 现场补算 —— 同函数同数据，不是造数。
    """
    mj = run_dir / "metrics.json"
    if not mj.exists():
        return None
    recs = json.loads(mj.read_text(encoding="utf-8"))["records"]
    if not recs:
        return None

    ss = run_dir / "summary_stats.json"
    derived = False
    if ss.exists():
        s = json.loads(ss.read_text(encoding="utf-8"))
        pl = s.get("plateau") or {}
        conv = s.get("convergence_multi_window")
        sel = s.get("selection")
    else:
        s, pl, conv, sel = {}, {}, None, None
    if not all(isinstance(pl.get(m), dict) for m in METRICS):
        pl = plateau_stats(recs, frac=0.5)
        conv = {f"w{w}": convergence_check(recs, window=w)
                for w in (5, 10, 15, 20, 25)}
        derived = True
    if not all(isinstance(pl.get(m), dict) for m in METRICS):
        return None

    return {
        "dir": run_dir.name,
        "plateau_derived": derived,
        "plateau": {m: {"mean": float(pl[m]["plateau_mean"]),
                        "std": float(pl[m]["plateau_std"]),
                        "best": pl[m].get("best"),
                        "best_epoch": pl[m].get("best_epoch")}
                    for m in METRICS},
        "epoch_range": pl.get("epoch_range"),
        "n_plateau": pl.get("plateau_n"),
        "curves": {m: [r[f"monitor_{m}"] for r in recs] for m in METRICS},
        "epochs": [r["epoch"] for r in recs],
        "convergence": conv,
        "selection": sel,
    }


def collect(base_run: str = BASE_RUN) -> tuple[dict, dict, dict]:
    """返回 (design, base_run_data, {group_name: run_data})。"""
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    base = load_run(ROOT / "runs" / base_run)
    groups = {}
    for name, g in design["groups"].items():
        d = load_run(ROOT / g["out_dir"])
        if d is not None:
            d["desc"] = g["desc"]
            d["changes"] = g.get("changes", {})
            groups[name] = d
    return design, base, groups


# --------------------------------------------------------------------------
# 判定（复用与 verdict_ablation 相同的规则，门槛从设计存档读）
# --------------------------------------------------------------------------
def verdict(base: dict, exp: dict, thr: dict) -> dict:
    rows, n_better, n_worse = [], 0, 0
    for m in METRICS:
        b = base["plateau"][m]["mean"]
        e = exp["plateau"][m]["mean"]
        imp = (b - e) / b * 100.0
        t = float(thr[m])
        if imp > t:
            tag = "BETTER"; n_better += 1
        elif imp < -t:
            tag = "WORSE"; n_worse += 1
        else:
            tag = "FLAT"
        rows.append({"metric": m, "improve_pct": imp, "tag": tag,
                     "threshold_pct": t})
    if n_worse:
        v = "REJECT_TRADE"
    elif n_better == len(METRICS):
        v = "ACCEPT_FULL"
    elif n_better:
        v = "ACCEPT_PART"
    else:
        v = "REJECT_NULL"
    return {"verdict": v, "rows": rows}


# --------------------------------------------------------------------------
# 图 1：跨组平台区柱状对比（每指标一张，含 ±σ 误差棒）
# --------------------------------------------------------------------------
def fig_group_bars(base: dict, groups: dict, outdir: Path) -> list[Path]:
    """平台区均值±σ 柱状对比。

    肉眼审阅修正（2026-08-11，压平类缺陷第 3 次复发）：
      * y 轴不再从 0 起 —— 各组差异只有几个百分点，从 0 起会把柱高压成
        齐平，图失去信息量。改为按 [min-σ, max+σ] 留边，并在轴标签明确
        标注「纵轴非零起点」，避免误导审稿人高估差异
      * 数值标签移到柱内顶部，否则被 errorbar 横帽遮住（原来 0.5684 只
        露出 684）
      * y 轴标签去掉竖排箭头文字，「越小越好」移到标题
    """
    if not groups:
        return []
    labels = ["baseline"] + list(groups.keys())
    out = []
    for m in METRICS:
        means = [base["plateau"][m]["mean"]] + \
                [g["plateau"][m]["mean"] for g in groups.values()]
        stds = [base["plateau"][m]["std"]] + \
               [g["plateau"][m]["std"] for g in groups.values()]
        sc = SCALE[m]
        fig, ax = plt.subplots(figsize=(max(6.8, 1.3 * len(labels)), 4.6))
        x = np.arange(len(labels))
        vals = [v * sc for v in means]
        errs = [v * sc for v in stds]
        colors = ["#666666"] + ["#4C72B0"] * len(groups)
        bars = ax.bar(x, vals, 0.62, yerr=errs, capsize=4, color=colors,
                      error_kw={"elinewidth": 1.0, "ecolor": "#222222"})

        # 非零起点：按 [min-σ, max+σ] 取范围再留 12% 边距
        lo = min(v - e for v, e in zip(vals, errs))
        hi = max(v + e for v, e in zip(vals, errs))
        pad = (hi - lo) * 0.28 if hi > lo else max(abs(hi), 1e-9) * 0.1
        ax.set_ylim(max(0.0, lo - pad * 0.45), hi + pad)

        # 数值标签放柱内顶部，避开 errorbar 横帽
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2,
                    v - (hi - lo) * 0.06,
                    f"{v:.4f}", ha="center", va="top", fontsize=7,
                    rotation=90, color="white", fontweight="bold")
        ax.axhline(vals[0], color="#444444", linestyle="--", linewidth=1.1,
                   alpha=0.8, label="baseline 平台区均值")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
        unit = f" (×{sc:.0e})" if sc != 1.0 else ""
        ax.set_ylabel(f"{MNAME[m]}{unit}")
        ax.set_title(f"平台区均值 ± σ 对比 —— {MNAME[m]}（越小越好，"
                     f"epochs {base['epoch_range']}）\n"
                     f"注：纵轴非零起点，用于显示组间差异",
                     fontsize=9.5)
        ax.grid(alpha=0.3, linestyle=":", axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = _save(fig, outdir / f"T1_plateau_{m}.png",
                  {"labels": labels, "mean": means, "std": stds,
                   "scale": sc, "epoch_range": base["epoch_range"],
                   "ylim": list(ax.get_ylim()),
                   "y_axis_starts_at_zero": False,
                   "note": "数字全部来自各 run summary_stats.json；"
                           "纵轴非零起点已在标题标注"})
        out.append(p)
    return out


# --------------------------------------------------------------------------
# 图 2：跨组训练曲线叠加（每指标一张）
# --------------------------------------------------------------------------
def fig_group_curves(base: dict, groups: dict, outdir: Path) -> list[Path]:
    out = []
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        sc = SCALE[m]
        ax.plot(base["epochs"], [v * sc for v in base["curves"][m]],
                color="#666666", linewidth=2.0, label="baseline", zorder=3)
        for i, (name, g) in enumerate(groups.items()):
            ax.plot(g["epochs"], [v * sc for v in g["curves"][m]],
                    linewidth=1.2, alpha=0.85, label=name)
        if base.get("epoch_range"):
            ax.axvspan(base["epoch_range"][0], base["epoch_range"][1],
                       color="#FFD9A0", alpha=0.25, zorder=0,
                       label="平台区(报数区间)")
        unit = f" (×{sc:.0e})" if sc != 1.0 else ""
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{MNAME[m]}{unit}  ↓ 越小越好")
        ax.set_title(f"验证集 {MNAME[m]} 收敛曲线（各消融组）", fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = _save(fig, outdir / f"T2_curve_{m}.png",
                  {"metric": m, "scale": sc,
                   "series": {"baseline": base["curves"][m],
                              **{k: v["curves"][m]
                                 for k, v in groups.items()}},
                   "epochs": base["epochs"]})
        out.append(p)
    return out


# --------------------------------------------------------------------------
# 图 3：改善率 + 显著性门槛（一张图看懂哪组过线）
# --------------------------------------------------------------------------
def fig_improvement(base: dict, groups: dict, thr: dict,
                    outdir: Path) -> Path | None:
    """改善率总览。

    肉眼审阅修正（2026-08-11）：
      * 柱色表示裁定（绿/灰/红），故图例改为「裁定色」而非系列色 ——
        原来用系列色做图例，三条全绿完全误导
      * 柱下加 C/H/N 字母区分指标（柱色已被裁定占用，无法再表示指标）
      * 负值标签改为置于柱内侧，避免与 x 轴组名重叠
      * 三条门槛虚线右端加 CD/HD/NUC 标注，否则分不清哪条属哪个指标
    """
    if not groups:
        return None
    from matplotlib.patches import Patch

    names = list(groups.keys())
    fig, ax = plt.subplots(figsize=(max(7.6, 1.45 * len(names)), 5.0))
    w = 0.8 / len(METRICS)
    x = np.arange(len(names))
    payload = {}
    C_OK, C_FLAT, C_BAD = "#55A868", "#BBBBBB", "#C44E52"

    for i, m in enumerate(METRICS):
        imps = [(base["plateau"][m]["mean"] - g["plateau"][m]["mean"])
                / base["plateau"][m]["mean"] * 100.0
                for g in groups.values()]
        payload[m] = imps
        pos = x + i * w - 0.4 + w / 2
        cols = [C_OK if v > thr[m] else (C_BAD if v < -thr[m] else C_FLAT)
                for v in imps]
        bars = ax.bar(pos, imps, w * 0.88, color=cols,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, imps):
            inside = v < 0            # 负值标签放柱内，避免压到 x 轴组名
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.10 if not inside else 0.10),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if not inside else "bottom",
                    fontsize=6.5, rotation=90,
                    color="black" if not inside else "#333333")
            # 柱底标指标首字母（柱色已用于裁定，需另加区分）
            ax.text(b.get_x() + b.get_width() / 2, 0,
                    MNAME[m][0], ha="center", va="top", fontsize=6.5,
                    color="#444444")

    # 门槛线 + 右端标注
    xmax = len(names) - 0.5
    for m in METRICS:
        ax.axhline(thr[m], color="#55A868", linestyle=":", linewidth=0.8,
                   alpha=0.55)
        ax.text(xmax + 0.02, thr[m], f" {MNAME[m]} {thr[m]}%",
                fontsize=6.5, color="#3B7A50", va="center")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlim(-0.6, xmax + 0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("相对 baseline 改善率 (%)   ↑ 正值 = 变好")
    ax.set_title("各消融组改善率与显著性门槛（柱下 C/H/N = CD/HD/NUC）",
                 fontsize=10)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.legend(handles=[Patch(facecolor=C_OK, label="过门槛（显著改善）"),
                       Patch(facecolor=C_FLAT, label="门槛内（判持平）"),
                       Patch(facecolor=C_BAD, label="劣化超门槛")],
              fontsize=8, loc="upper left", framealpha=0.95)
    ax.margins(y=0.14)
    fig.tight_layout()
    return _save(fig, outdir / "T3_improvement.png",
                 {"groups": names, "improve_pct": payload,
                  "thresholds_pct": thr,
                  "note": "门槛来自 ablation_matrix.json，跑前定死；"
                          "柱色=裁定，柱下字母=指标"})


# --------------------------------------------------------------------------
# 主表（Markdown + LaTeX）
# --------------------------------------------------------------------------
def build_tables(design: dict, base: dict, groups: dict,
                 outdir: Path) -> dict:
    thr = design["significance_thresholds_pct"]
    md = ["# 消融实验主表（自动生成，数字全部来自落盘产物）", "",
          f"- baseline: `{base['dir']}`，平台区 epochs "
          f"{base['epoch_range']}（n={base['n_plateau']}）",
          f"- 报数方式：**平台区均值 ± σ**",
          f"- 显著性门槛（跑前定死）：CD {thr['cd']}% / HD {thr['hd']}% / "
          f"NUC {thr['nuc']}%",
          "- ↓ 表示越小越好；改善率 = (baseline − 本组) / baseline", ""]

    head = ("| 组 | 说明 | CD×10³ ↓ | HD×10³ ↓ | NUC ↓ | "
            "ΔCD% | ΔHD% | ΔNUC% | 裁定 |")
    md += [head, "|" + "---|" * 9]
    b = base["plateau"]
    md.append("| **baseline** | 干净对照（四项改进全关） | "
              f"{b['cd']['mean']*1e3:.4f} ± {b['cd']['std']*1e3:.4f} | "
              f"{b['hd']['mean']*1e3:.4f} ± {b['hd']['std']*1e3:.4f} | "
              f"{b['nuc']['mean']:.4f} ± {b['nuc']['std']:.4f} | "
              "— | — | — | — |")

    verdicts = {}
    for name, g in groups.items():
        v = verdict(base, g, thr)
        verdicts[name] = v
        imp = {r["metric"]: r for r in v["rows"]}
        p = g["plateau"]
        mark = {"BETTER": "**", "WORSE": "", "FLAT": ""}
        cells = []
        for m in METRICS:
            r = imp[m]
            mk = mark[r["tag"]]
            cells.append(f"{mk}{r['improve_pct']:+.2f}{mk}")
        md.append(f"| {name} | {g['desc']} | "
                  f"{p['cd']['mean']*1e3:.4f} ± {p['cd']['std']*1e3:.4f} | "
                  f"{p['hd']['mean']*1e3:.4f} ± {p['hd']['std']*1e3:.4f} | "
                  f"{p['nuc']['mean']:.4f} ± {p['nuc']['std']:.4f} | "
                  + " | ".join(cells) + f" | `{v['verdict']}` |")

    md += ["", "**裁定含义**：`ACCEPT_FULL` 三项均过门槛（可入主表）/ "
           "`ACCEPT_PART` 部分过线其余持平（入附录，须写明持平项）/ "
           "`REJECT_TRADE` 有项劣化超门槛（判 trade-off）/ "
           "`REJECT_NULL` 三项均在门槛内（判无效）。", ""]

    if not groups:
        md += ["> ⚠️ 当前尚无已完成的消融组，表中仅有 baseline 一行。", ""]

    # LaTeX（三线表，学位论文常用）
    tex = [r"\begin{table}[htbp]", r"  \centering",
           r"  \caption{消融实验结果（平台区均值 $\pm$ 标准差）}",
           r"  \label{tab:ablation}",
           r"  \begin{tabular}{lcccccc}", r"    \toprule",
           r"    方法 & CD$\times10^{3}\downarrow$ & "
           r"HD$\times10^{3}\downarrow$ & NUC$\downarrow$ & "
           r"$\Delta$CD\% & $\Delta$HD\% & $\Delta$NUC\% \\",
           r"    \midrule",
           f"    baseline & {b['cd']['mean']*1e3:.4f} $\\pm$ "
           f"{b['cd']['std']*1e3:.4f} & {b['hd']['mean']*1e3:.4f} $\\pm$ "
           f"{b['hd']['std']*1e3:.4f} & {b['nuc']['mean']:.4f} $\\pm$ "
           f"{b['nuc']['std']:.4f} & -- & -- & -- \\\\"]
    for name, g in groups.items():
        v = verdicts[name]
        imp = {r["metric"]: r["improve_pct"] for r in v["rows"]}
        p = g["plateau"]
        esc = name.replace("_", r"\_")
        tex.append(f"    {esc} & {p['cd']['mean']*1e3:.4f} $\\pm$ "
                   f"{p['cd']['std']*1e3:.4f} & {p['hd']['mean']*1e3:.4f}"
                   f" $\\pm$ {p['hd']['std']*1e3:.4f} & "
                   f"{p['nuc']['mean']:.4f} $\\pm$ {p['nuc']['std']:.4f} & "
                   f"{imp['cd']:+.2f} & {imp['hd']:+.2f} & "
                   f"{imp['nuc']:+.2f} \\\\")
    tex += [r"    \bottomrule", r"  \end{tabular}",
            r"  \begin{tablenotes}\footnotesize",
            f"    \\item 显著性门槛：CD {thr['cd']}\\%, HD {thr['hd']}\\%, "
            f"NUC {thr['nuc']}\\%（跑前依 baseline 噪声定死，$2\\times$ 标准误）。",
            r"  \end{tablenotes}", r"\end{table}"]

    (outdir / "TABLE_ablation.md").write_text("\n".join(md),
                                              encoding="utf-8")
    (outdir / "TABLE_ablation.tex").write_text("\n".join(tex),
                                               encoding="utf-8")
    return verdicts


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_assets")
    ap.add_argument("--base", default=BASE_RUN,
                    help="baseline run 目录名（默认 B002_baseline150）")
    args = ap.parse_args()
    outdir = ROOT / args.outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"

    design, base, groups = collect(args.base)
    thr = design["significance_thresholds_pct"]

    print("=" * 72)
    print("论文素材汇总")
    print("=" * 72)
    if base is None:
        print(f"[WAIT] baseline `{args.base}` 未完成（无 summary_stats.json）")
        print("       B-002 跑完后再运行本脚本。")
        return 1
    print(f"baseline : {base['dir']}  平台区 {base['epoch_range']} "
          f"(n={base['n_plateau']})")
    done = list(groups.keys())
    todo = [n for n in design["groups"] if n not in groups]
    print(f"已完成组 : {done if done else '(无)'}")
    print(f"未完成组 : {todo if todo else '(无)'}")

    figs = []
    figs += fig_group_bars(base, groups, figdir)
    figs += fig_group_curves(base, groups, figdir)
    f3 = fig_improvement(base, groups, thr, figdir)
    if f3:
        figs.append(f3)
    verdicts = build_tables(design, base, groups, outdir)

    print(f"\n生成图 {len(figs)} 张 -> {figdir}")
    for p in figs:
        print(f"  {p.name}  {p.stat().st_size} B  "
              f"(data.json={'OK' if p.with_suffix('.data.json').exists() else 'MISSING'})")
    print(f"\n主表 -> {outdir/'TABLE_ablation.md'}")
    print(f"LaTeX -> {outdir/'TABLE_ablation.tex'}")
    if verdicts:
        print("\n裁定汇总:")
        for k, v in verdicts.items():
            print(f"  {k:<18} {v['verdict']}")
    (outdir / "MANIFEST.json").write_text(json.dumps(
        {"baseline": base["dir"], "done_groups": done, "todo_groups": todo,
         "thresholds_pct": thr, "verdicts": verdicts,
         "figures": [p.name for p in figs]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
