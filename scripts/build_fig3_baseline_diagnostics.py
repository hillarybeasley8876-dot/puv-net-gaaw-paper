"""Build the chapter-3 baseline diagnostic figure from archived sample data.

The figure reads ``docs/_ch3_diag.json`` directly.  It describes one fixed
baseline model on 200 preselected validation samples; it is not a cross-seed
uncertainty analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_thesis_figures import save, setup_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "_ch3_diag.json"
AUDIT = ROOT / "docs" / "_split_audit_and_holdout_subset.json"


def main() -> int:
    setup_style()
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    heldout_indices = set(audit["archived_slice"]["heldout_indices"])
    per_sample = [row for row in data["per_sample"] if row["idx"] in heldout_indices]

    pred = np.asarray([row["nn_pred_cv"] for row in per_sample], dtype=float)
    truth = np.asarray([row["nn_gt_cv"] for row in per_sample], dtype=float)
    delta = pred - truth
    q_values = 1_000 * np.asarray(
        [row["bwd_by_sparsity"] for row in per_sample], dtype=float
    ).mean(axis=0)
    n_sample = len(per_sample)
    cv_ratio = float(pred.mean() / truth.mean())
    q4_over_q1 = float(q_values[3] / q_values[0])

    if n_sample != int(audit["archived_slice"]["heldout_validation_n"]):
        raise RuntimeError("held-out rows do not match the split-audit record")

    colors = {"truth": "#4C78A8", "pred": "#F58518", "delta": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), gridspec_kw={"wspace": 0.34})

    # (a) Fixed-model sample distributions.
    ax = axes[0]
    parts = ax.violinplot([truth, pred], positions=[0, 1], showmeans=False,
                          showmedians=False, showextrema=False, widths=0.72)
    for body, color in zip(parts["bodies"], [colors["truth"], colors["pred"]]):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
        body.set_linewidth(0.7)
    ax.boxplot([truth, pred], positions=[0, 1], widths=0.22, patch_artist=True,
               showfliers=False,
               boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.8},
               medianprops={"color": "#B22222", "linewidth": 1.1},
               whiskerprops={"color": "#333333", "linewidth": 0.8},
               capprops={"color": "#333333", "linewidth": 0.8})
    ax.set_xticks([0, 1], ["真值", "基线预测"])
    ax.set_ylabel(r"$cv_{nn}$（越小越好）")
    ax.text(0.5, 0.97,
            f"均值比 {cv_ratio:.2f}×（n={n_sample}）",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4)
    ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes, fontsize=9.2, weight="bold")
    ax.grid(axis="y", alpha=0.18)

    # (b) Paired differences; all positive means every prediction is more dispersed.
    ax = axes[1]
    bins = np.linspace(min(0.0, float(delta.min())), float(delta.max()) * 1.04, 24)
    ax.hist(delta, bins=bins, color=colors["delta"], alpha=0.78,
            edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#B22222", linestyle="--", linewidth=1.1)
    ax.axvline(float(delta.mean()), color="#333333", linestyle=":", linewidth=1.1)
    ax.set_xlabel(r"配对差值：预测 $cv_{nn}$ − 真值 $cv_{nn}$")
    ax.set_ylabel("样本数")
    ax.text(0.5, 0.97, f"{int(np.sum(delta > 0))}/{n_sample} 个样本差值为正",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4)
    ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes, fontsize=9.2, weight="bold")
    ax.grid(axis="y", alpha=0.18)

    # (c) Backward Chamfer component by input sparsity quartile.
    ax = axes[2]
    x = np.arange(4)
    bars = ax.bar(x, q_values, color=["#4C78A8", "#72A0C1", "#F2A65A", "#E45756"],
                  width=0.68, edgecolor="#444444", linewidth=0.45)
    ax.set_xticks(x, ["Q1\n最密", "Q2", "Q3", "Q4\n最疏"])
    ax.set_ylabel(r"后向 Chamfer 分量 $\times 10^3$")
    ax.set_ylim(0, float(q_values.max()) * 1.23)
    for bar, value in zip(bars, q_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.025,
                f"{value:.3f}", ha="center", va="bottom", fontsize=7.6)
    ax.text(0.5, 0.97, f"Q4/Q1 = {q4_over_q1:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4)
    ax.text(-0.14, 1.03, "(c)", transform=ax.transAxes, fontsize=9.2, weight="bold")
    ax.grid(axis="y", alpha=0.18)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save(fig, "F3_4_baseline_diagnostics")
    print(f"Wrote F3_4_baseline_diagnostics from {SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
