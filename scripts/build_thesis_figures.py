"""Build vector thesis schematics and data figures from archived JSON.

All numerical plots read docs/_cv_nn_measure.json and are restricted to the
post-hoc intersection with the true held-out validation split.  The script also
writes a machine-readable summary with the source SHA-256 so figures and prose
can be audited against the same input.  Sample-level intervals describe the 11
archived held-out records only; they are not validation-set or seed-level
uncertainty.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "_cv_nn_measure.json"
OUT = ROOT / "docs" / "figures_thesis"
SUMMARY = ROOT / "docs" / "_thesis_results_summary.json"
SPLIT_AUDIT = ROOT / "docs" / "_split_audit_and_holdout_subset.json"
BOOT_SEED = 20260813
INVALID_RUNS = {"ABL_D1_scale_qk"}
DISPLAY_NAMES = {
    "B002_baseline150": "C0",
    "B002_baseline150_5090": "B0",
    "ABL_A1_cd_balance": "A1",
    "ABL_A2_cd_boost_bwd": "A2",
    "ABL_AC_combo": "AC",
    "ABL_C1_uniform": "C1",
    "ABL_B1_adv_fixed": "B1",
    "ABL_B2_adv_adaptive": "B2",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 160,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def box(ax, xy, wh, text, fc="#F5F7FA", ec="#315A7D", fontsize=8.2,
        lw=1.0, style="round,pad=0.02"):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#172B3A")
    return p


def arrow(ax, a, b, color="#5D7184", style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls))


def schematic_canvas(figsize=(8.6, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def build_schematics() -> None:
    # F3.1 quality framework
    fig, ax = schematic_canvas()
    box(ax, (0.2, 2.35), (1.5, 1.15), "稀疏输入点云\n$\\mathcal{P}$", fc="#E8F1FB")
    box(ax, (2.1, 2.35), (1.6, 1.15), "生成器\n$G_\\theta$", fc="#EAF5EC", ec="#3C7A4B")
    box(ax, (4.1, 2.35), (1.5, 1.15), "密集输出\n$\\mathcal{Q}$", fc="#FFF5DE", ec="#9A6B16")
    arrow(ax, (1.7, 2.92), (2.1, 2.92)); arrow(ax, (3.7, 2.92), (4.1, 2.92))
    labels = [(6.2, 4.7, "平均几何保真\nCD"), (8.1, 4.7, "最坏区域\nHD"),
              (6.2, 2.35, "局部间距\n$cv_{nn}$ / NUC"), (8.1, 2.35, "曲面贴合\nP2F")]
    for x, y, t in labels:
        box(ax, (x, y), (1.55, 0.9), t, fc="#F6F3FB", ec="#6B4C8A")
        arrow(ax, (5.6, 2.92), (x, y + 0.45), color="#7B6A91")
    ax.text(7.55, 1.25, "单项指标回答不同问题；反向变化按权衡报告",
            ha="center", color="#9A3B32", fontsize=8.5)
    save(fig, "F3_1_quality_framework")

    # F3.2 evidence chain
    fig, ax = schematic_canvas((9.2, 4.5))
    names = ["配置与代码", "日志与\ncheckpoint", "固定样本推理", "逐样本结果", "统计与图表", "文字主张"]
    xs = np.linspace(0.2, 8.5, len(names))
    for i, (x, name) in enumerate(zip(xs, names)):
        fc = "#EAF5EC" if i in (0, 2, 3, 4) else "#FFF0EC"
        ec = "#3C7A4B" if i in (0, 2, 3, 4) else "#A44A3F"
        box(ax, (x, 3.2), (1.25, 0.9), name, fc=fc, ec=ec, fontsize=7.8)
        if i:
            arrow(ax, (xs[i-1] + 1.25, 3.65), (x, 3.65))
    box(ax, (1.65, 1.2), (2.3, 0.9), "当前缺口\n部分原始run包未入库", fc="#FFF0EC", ec="#A44A3F")
    box(ax, (5.7, 1.2), (2.3, 0.9), "阻断规则\n缺证据不进摘要和结论", fc="#FFF5DE", ec="#9A6B16")
    arrow(ax, (2.8, 3.2), (2.8, 2.1), color="#A44A3F", ls="--")
    arrow(ax, (3.95, 1.65), (5.7, 1.65), color="#A44A3F")
    save(fig, "F3_2_evidence_chain")

    # F3.3 closed loop
    fig, ax = schematic_canvas((8.6, 4.4))
    items = [(0.4, 2.6, "定义\n任务与质量"), (2.3, 4.3, "测量\n指标与梯度"),
             (5.0, 4.3, "对照\n批次内控制"), (7.5, 2.6, "裁定\n假设方向"),
             (5.0, 0.8, "边界\n失败与缺口"), (2.3, 0.8, "修正\n问题与实验")]
    for x, y, t in items:
        box(ax, (x, y), (1.45, 0.8), t, fc="#EDF3F8")
    centers = [(x + .725, y + .4) for x, y, _ in items]
    for i in range(len(centers)):
        arrow(ax, centers[i], centers[(i+1) % len(centers)], color="#315A7D")
    ax.text(5, 2.85, "可追溯证据闭环", ha="center", va="center",
            fontsize=11, color="#315A7D", weight="bold")
    save(fig, "F3_3_analysis_loop")

    # F4.1 hypothesis model
    fig, ax = schematic_canvas((9.4, 5.0))
    box(ax, (0.3, 4.45), (2.0, 0.8), "训练目标赋权方式", fc="#E8F1FB")
    for y, text in [(3.3, "固定权重 / 动态权重"), (2.15, "参数作用域 / 目标比"), (1.0, "判别器状态 / 引入阶段")]:
        box(ax, (0.3, y), (2.0, 0.7), text, fc="#F5F7FA")
        arrow(ax, (1.3, 4.45), (1.3, y + .7))
    box(ax, (3.6, 3.5), (2.1, 1.0), "机制变量\n$g_{cd},g_{adv},\\rho,w_t$", fc="#FFF5DE", ec="#9A6B16")
    box(ax, (7.0, 4.35), (2.0, 0.8), "H1：机制存在", fc="#F6F3FB", ec="#6B4C8A")
    box(ax, (7.0, 2.9), (2.0, 0.8), "H2/H3：B2 vs B1", fc="#F6F3FB", ec="#6B4C8A")
    box(ax, (7.0, 1.45), (2.0, 0.8), "H4/H5：整体价值", fc="#F6F3FB", ec="#6B4C8A")
    for y in (3.65, 2.5, 1.35):
        arrow(ax, (2.3, y), (3.6, 4.0), color="#6E8192")
    for y in (4.75, 3.3, 1.85):
        arrow(ax, (5.7, 4.0), (7.0, y), color="#6B4C8A")
    save(fig, "F4_1_hypothesis_model")

    # F4.2 comparison chains
    fig, ax = schematic_canvas((9.2, 4.6))
    ax.text(2.6, 5.4, "对抗赋权实验链", ha="center", weight="bold", color="#315A7D")
    for x, t, c in [(0.3, "无对抗基线", "#E8F1FB"), (2.2, "B1\n固定8.27", "#FFF0EC"), (4.1, "B2\nGAAW", "#EAF5EC")]:
        box(ax, (x, 3.55), (1.45, 0.95), t, fc=c)
    arrow(ax, (1.75, 4.02), (2.2, 4.02)); arrow(ax, (3.65, 4.02), (4.1, 4.02))
    ax.text(2.95, 3.0, "相对本批次B0计算变化", ha="center", fontsize=8.5)
    ax.text(7.3, 5.4, "均匀性与结构消融链", ha="center", weight="bold", color="#315A7D")
    box(ax, (6.0, 3.55), (1.45, 0.95), "无对抗基线", fc="#E8F1FB")
    box(ax, (8.0, 3.55), (1.45, 0.95), "C1\n均匀性损失", fc="#FFF5DE", ec="#9A6B16")
    arrow(ax, (7.45, 4.02), (8.0, 4.02))
    box(ax, (3.2, 1.0), (3.0, 0.9), "独立批次绑定各自基线\nseed=1只作方向性观察", fc="#FFF0EC", ec="#A44A3F")
    arrow(ax, (4.85, 3.0), (4.7, 1.9), color="#A44A3F", ls="--")
    arrow(ax, (7.7, 3.55), (6.2, 1.9), color="#A44A3F", ls="--")
    save(fig, "F4_2_control_matrix")

    # F5.1 inspiration relationship
    fig, ax = schematic_canvas((9.2, 4.6))
    box(ax, (0.3, 3.4), (2.0, 1.05), "固定权重\n$w_f$不随训练变化", fc="#F5F7FA")
    box(ax, (3.45, 3.4), (2.3, 1.05), "VQGAN直接先例\n最后一层梯度范数比+裁剪", fc="#F6F3FB", ec="#6B4C8A")
    box(ax, (6.9, 3.4), (2.0, 1.05), "本文GAAW\n全生成器+目标比", fc="#EAF5EC", ec="#3C7A4B")
    arrow(ax, (2.3, 3.92), (3.45, 3.92)); arrow(ax, (5.75, 3.92), (6.9, 3.92))
    box(ax, (2.3, 1.25), (4.6, 0.95), "贡献定位：点云场景适配、控制变量比较与失效边界\n不是梯度范数比公式首创",
        fc="#FFF5DE", ec="#9A6B16")
    arrow(ax, (4.6, 3.4), (4.6, 2.2), color="#9A6B16")
    save(fig, "F5_1_inspiration_relation")

    # F5.2 gradient path
    fig, ax = schematic_canvas((9.5, 5.0))
    box(ax, (0.25, 3.9), (1.55, 0.85), "Chamfer损失\n$\\mathcal{L}_{CD}$", fc="#E8F1FB")
    box(ax, (0.25, 1.5), (1.55, 0.85), "对抗损失\n$\\mathcal{L}_{adv}$", fc="#FFF0EC", ec="#A44A3F")
    box(ax, (2.65, 3.9), (1.55, 0.85), "梯度范数\n$g_{cd}$", fc="#F5F7FA")
    box(ax, (2.65, 1.5), (1.55, 0.85), "梯度范数\n$g_{adv}$", fc="#F5F7FA")
    box(ax, (5.0, 2.7), (1.8, 0.95), "动态权重\n$w_t=r g_{cd}/g_{adv}$", fc="#FFF5DE", ec="#9A6B16")
    box(ax, (7.6, 2.7), (1.55, 0.95), "总损失\n更新$\\theta$", fc="#EAF5EC", ec="#3C7A4B")
    arrow(ax, (1.8, 4.32), (2.65, 4.32), ls="--")
    arrow(ax, (1.8, 1.92), (2.65, 1.92), ls="--")
    arrow(ax, (4.2, 4.32), (5.0, 3.3), ls="--")
    arrow(ax, (4.2, 1.92), (5.0, 3.0), ls="--")
    arrow(ax, (1.8, 4.0), (7.6, 3.35), color="#315A7D")
    arrow(ax, (6.8, 3.17), (7.6, 3.17), color="#9A6B16")
    arrow(ax, (1.8, 2.15), (7.6, 2.95), color="#A44A3F")
    ax.text(4.65, 0.7, "虚线：只测量并detach；实线：最终反向传播路径",
            ha="center", fontsize=8.4, color="#5D7184")
    save(fig, "F5_2_gradient_path")


def bootstrap_mean_ci(x: np.ndarray, n_boot=10000):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def metrics(run):
    return {
        "cv_nn": float(run["cv_nn"]["mean"]),
        "cd": float(run["cd_infer"]["mean"]),
        "hd": float(run["hd_infer"]["mean"]),
        "q4_over_q1": float(run["strata"]["q4_over_q1"]),
    }


def restrict_to_heldout(data):
    """Recompute archived metrics on the true-validation intersection only."""
    audit = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))
    heldout = set(audit["archived_slice"]["heldout_indices"])
    corrected = copy.deepcopy(data)
    for name, run in corrected["runs"].items():
        rows = [row for row in run["per_sample"] if int(row["idx"]) in heldout]
        if len(rows) != len(heldout):
            raise RuntimeError(f"{name}: held-out intersection is incomplete")

        def stat(field):
            values = np.asarray([float(row[field]) for row in rows])
            return {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "se_sample": float(values.std(ddof=1) / np.sqrt(len(values))),
            }

        run["per_sample"] = rows
        run["n_sample"] = len(rows)
        run["cv_nn"] = stat("nn_pred_cv")
        run["cv_nn_gt"] = stat("nn_gt_cv")
        run["cd_infer"] = stat("cd")
        run["hd_infer"] = stat("hd")
        q = np.asarray([row["bwd_by_sparsity"] for row in rows], dtype=float).mean(axis=0)
        run["strata"] = {
            "bwd_mean_by_quartile": [float(x) for x in q],
            "q4_over_q1": float(q[3] / q[0]),
        }
        run["measurement_valid"] = name not in INVALID_RUNS

    corrected["n_sample"] = len(heldout)
    corrected["sample_scope"] = "post-hoc intersection with the true held-out validation split"
    return corrected


def paired_stats(a, b, key):
    field = {"cv_nn": "nn_pred_cv", "cd": "cd", "hd": "hd"}[key]
    xa = {int(x["idx"]): float(x[field]) for x in a["per_sample"]}
    xb = {int(x["idx"]): float(x[field]) for x in b["per_sample"]}
    ids = sorted(set(xa) & set(xb))
    d = np.array([xb[i] - xa[i] for i in ids], dtype=float)
    return {
        "n": len(ids), "mean_difference": float(d.mean()),
        "median_difference": float(np.median(d)),
        "sample_sd": float(d.std(ddof=1)),
        "sample_se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "bootstrap_95ci_mean": bootstrap_mean_ci(d),
        "fraction_improved": float(np.mean(d < 0)),
    }, d


def build_data_figures(data):
    runs = data["runs"]
    base3090, base5090 = "B002_baseline150", "B002_baseline150_5090"
    core = [base5090, "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive", base3090, "ABL_C1_uniform"]

    # F6.1 core cv_nn, separated by experiment batch
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    panels = [([base5090, "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"], "对抗赋权实验批次"),
              ([base3090, "ABL_C1_uniform"], "均匀性消融实验批次")]
    names = {base5090: "Baseline", "ABL_B1_adv_fixed": "B1 fixed",
             "ABL_B2_adv_adaptive": "B2 GAAW", base3090: "Baseline",
             "ABL_C1_uniform": "C1 uniform"}
    colors = {"Baseline": "#4C78A8", "B1 fixed": "#E45756", "B2 GAAW": "#54A24B", "C1 uniform": "#F2CF5B"}
    for ax, (ids, title) in zip(axes, panels):
        vals = [runs[i]["cv_nn"]["mean"] for i in ids]
        ses = [runs[i]["cv_nn"]["se_sample"] for i in ids]
        labs = [names[i] for i in ids]
        ax.bar(np.arange(len(ids)), vals, yerr=ses, capsize=3,
               color=[colors[x] for x in labs], edgecolor="#31424F", linewidth=.6)
        ax.set_xticks(np.arange(len(ids)), labs)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        for x, v in enumerate(vals):
            ax.text(x, v + .004, f"{v:.4f}", ha="center", fontsize=7.8)
    axes[0].set_ylabel(r"$cv_{nn}$（越小越好）")
    fig.text(.5, -.01, "真实验证交集 n=11；误差线为跨样本SE；当前均为单训练seed。", ha="center", fontsize=8, color="#9A3B32")
    save(fig, "F6_2_core_cvnn")

    # F6.2 paired B1 -> B2 differences
    diffs = []
    stats_out = {}
    for key in ("cv_nn", "cd", "hd"):
        s, d = paired_stats(runs["ABL_B1_adv_fixed"], runs["ABL_B2_adv_adaptive"], key)
        stats_out[key] = s
        diffs.append(d)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5), gridspec_kw={"wspace": .42})
    for ax, d, lab in zip(axes, diffs, [r"$cv_{nn}$", "CD", "HD"]):
        ax.boxplot(d, vert=True, widths=.45, showfliers=False,
                   boxprops={"color": "#315A7D"}, medianprops={"color": "#E45756"})
        jitter = np.random.default_rng(BOOT_SEED).normal(1, .035, len(d))
        ax.scatter(jitter, d, s=7, alpha=.22, color="#4C78A8", edgecolors="none")
        ax.axhline(0, color="#A44A3F", linestyle="--", linewidth=.8)
        ax.set_xticks([1], [lab])
        ax.tick_params(axis="y", labelsize=7.2)
        ax.grid(axis="y", alpha=.2)
        ax.set_title(f"改善样本占比 {np.mean(d < 0):.1%}")
    axes[0].set_ylabel("B2 − B1（负值为改善）")
    fig.text(.5, -.02, "真实验证交集 n=11；配对分布不是跨seed不确定性。", ha="center", fontsize=8, color="#9A3B32")
    save(fig, "F6_3_b1_b2_paired")

    # F6.3 relative changes vs the corresponding batch baseline
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, gpu, base in [(axes[0], "3090", base3090), (axes[1], "5090", base5090)]:
        ids = [k for k, v in runs.items()
               if str(v["gpu"]) == gpu and k != base and k not in INVALID_RUNS]
        ids.sort()
        y = np.arange(len(ids))
        for off, key, color in [(-.22, "cv_nn", "#4C78A8"), (0, "cd", "#F58518"), (.22, "hd", "#54A24B")]:
            changes = []
            for i in ids:
                m, b = metrics(runs[i]), metrics(runs[base])
                changes.append(100 * (m[key] - b[key]) / b[key])
            ax.barh(y + off, changes, height=.2, label=key, color=color)
        ax.axvline(0, color="#37474F", linewidth=.8)
        ax.set_yticks(y, [DISPLAY_NAMES[x] for x in ids], fontsize=8.2)
        batch_title = "均匀性与结构消融" if gpu == "3090" else "对抗赋权"
        ax.set_title(f"{batch_title}：相对本批次baseline")
        ax.set_xlabel("相对变化/%（负值为改善）")
        ax.grid(axis="x", alpha=.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015), ncol=3,
               frameon=False)
    fig.text(.5, -.10, "真实验证交集 n=11；D1因补测时模型配置重建错误而排除。",
             ha="center", fontsize=8, color="#9A3B32")
    save(fig, "F6_4_ablation_by_batch")

    # F6.4 CD vs cv trade-off
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, gpu, base in [(axes[0], "3090", base3090), (axes[1], "5090", base5090)]:
        ids = [k for k, v in runs.items()
               if str(v["gpu"]) == gpu and k not in INVALID_RUNS]
        for i in ids:
            m = metrics(runs[i])
            label = DISPLAY_NAMES[i]
            ax.scatter(m["cd"] * 1e3, m["cv_nn"], s=35, label=label)
            ax.annotate(label, (m["cd"] * 1e3, m["cv_nn"]), xytext=(3, 3), textcoords="offset points", fontsize=6.8)
        ax.margins(x=.13, y=.08)
        ax.set_title("均匀性与结构消融" if gpu == "3090" else "对抗赋权")
        ax.set_xlabel(r"CD $\times10^3$（越小越好）")
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r"$cv_{nn}$（越小越好）")
    fig.text(.5, -.01, "真实验证交集 n=11；仅用于探索性对照。",
             ha="center", fontsize=8, color="#9A3B32")
    save(fig, "F6_1_metric_tradeoff")
    return stats_out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    raw = SOURCE.read_bytes()
    data = restrict_to_heldout(json.loads(raw.decode("utf-8")))
    build_schematics()
    paired = build_data_figures(data)

    runs = data["runs"]
    result = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": data.get("seed"),
        "n_sample": data.get("n_sample"),
        "sample_scope": data["sample_scope"],
        "split_audit": str(SPLIT_AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "uncertainty_note": "n=11 sample-level exploration, not seed-level uncertainty",
        "runs": {k: {"batch": ("uniformity_ablation" if str(v["gpu"]) == "3090" else "adversarial_weighting"),
                     "best_epoch": v["best_epoch"],
                     "measurement_valid": v["measurement_valid"],
                     "sample_se": {
                         "cv_nn": float(v["cv_nn"]["se_sample"]),
                         "cd": float(v["cd_infer"]["se_sample"]),
                         "hd": float(v["hd_infer"]["se_sample"]),
                     },
                     **metrics(v)}
                 for k, v in runs.items()},
        "paired_B1_to_B2": paired,
        "comparisons": {},
    }
    pairs = [
        ("B0_to_B1", "B002_baseline150_5090", "ABL_B1_adv_fixed"),
        ("B1_to_B2", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"),
        ("B0_to_B2", "B002_baseline150_5090", "ABL_B2_adv_adaptive"),
        ("C0_to_C1", "B002_baseline150", "ABL_C1_uniform"),
    ]
    for name, a, b in pairs:
        ma, mb = metrics(runs[a]), metrics(runs[b])
        result["comparisons"][name] = {
            "control": a, "experiment": b,
            "relative_change_pct": {k: 100 * (mb[k] - ma[k]) / ma[k] for k in ma},
        }
    SUMMARY.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY}")
    print(f"Figures: {len(list(OUT.glob('*.svg')))} SVG + "
          f"{len(list(OUT.glob('*.pdf')))} PDF + {len(list(OUT.glob('*.png')))} PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
