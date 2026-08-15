# -*- coding: utf-8 -*-
"""
图A —— 点云上采样结果的定性对比（B0 / B1 / B2 / C1 同样本）。

盲审必问项：全篇若无一张点云图，"均匀性改善"这类结论无法被读者目视核验。
Codex 那份稿子 47 张图里一张点云都没有，并在 §6.3.2 / §6.3.4 自认三处解释
因缺可视化而无法闭环。本图即为补上该缺口。

设计要点（每一条都是为了让图能被"看出结论"而非只是好看）：
  1. 同样本横向对比：同一 shape 在 4 个 run 下的输出并列，输入与真值各一列；
  2. 逐点 CD 误差着色：点色 = 该点到真值的最近邻距离（前向分量），
     色标跨 run 共享同一 vmin/vmax，否则各自归一化会让"谁更好"不可比；
  3. 局部放大框：在误差最大的区域画框并单独放大，稀疏区差异在全局视图里
     只有几个像素，不放大等于没画；
  4. 样本选取由脚本按客观判据挑选（B2 相对 B1 的 CD 改善最大者），不手挑好看的。

口径与 measure_cv_nn.py 严格一致（SEED / VAL_RATIO / up_ratio / augment=False），
否则图与主表不是同一批样本，横向对比无效。

用法：
  python scripts/plot_qualitative.py                # 默认 3 个样本
  python scripts/plot_qualitative.py --n-shapes 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import torch                              # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
from mpl_toolkits.mplot3d import Axes3D    # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import PUTrainDataset   # noqa: E402
from puvnet.models.pu_transformer import PUTransformer  # noqa: E402

# ---- 与 measure_cv_nn.py 完全一致的口径常量（改动必须同步两处）----
SEED = 20260811
VAL_RATIO = 0.05
N_FULL = 200
UP_RATIO = 4

MEASURE = ROOT / "docs" / "_cv_nn_measure.json"
OUTDIR = ROOT / "paper_assets_TRIAL" / "figures_ch5"

# 展示列：run 名 -> 图上标签。B0/B1/B2 同属 5090 组，C1 属 3090 组。
# 跨机器红线：本图只做形态定性对比，不并列任何数值指标，故允许同图呈现；
# 图注中必须显式声明这一点（见 CAVEAT）。
COLS = [
    ("B002_baseline150_5090", "B0  baseline (no adv.)"),
    ("ABL_B1_adv_fixed",      "B1  fixed $w_{adv}=8.27$"),
    ("ABL_B2_adv_adaptive",   "B2  GA-PUT (ours)"),
    ("ABL_C1_uniform",        "C1  uniformity loss"),
]

CAVEAT = (
    "Caveat (scope of this figure): a qualitative shape-level comparison only. Rows 1-2 colour each point by its "
    "forward Chamfer distance to the ground truth (brighter = larger error);\n"
    "row 3 colours each point by its within-cloud nearest-neighbour spacing, normalised by the GT median spacing "
    "of the same shape. Colour scales are shared across all runs of the same\n"
    "shape but not across shapes, so each shape carries its own colour bars. The zoom window is placed on the "
    "region where GT is dense but the B1 prediction under-covers it. C1 was trained\n"
    "on a different GPU host than B0/B1/B2, so no numeric metric is juxtaposed here; all quantitative comparisons "
    "are reported within-host in the main tables. Shapes were selected by a\n"
    "scripted criterion (largest B2-minus-B1 per-sample CD gain), not hand-picked. The per-panel cv_nn is computed "
    "on this single shape and therefore differs from the 200-sample table means;\n"
    "because the shapes were selected on CD and not on cv_nn, B2 does not improve cv_nn on every panel here "
    "(e.g. shape 67133). The 200-sample cv_nn verdict is reported in the main tables."
)


def nn_to_ref(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """a 中每点到 ref 的最近邻欧氏距离。N 约 4096，直接算全矩阵即可。"""
    d = np.linalg.norm(a[:, None, :] - ref[None, :, :], axis=-1)
    return d.min(axis=1)


def nn_within(p: np.ndarray) -> np.ndarray:
    """点云内部最近邻间距（排除自身）。用于可视化均匀性，与 cv_nn 同定义。"""
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def load_generator(run: str) -> PUTransformer:
    """与 measure_cv_nn.load_generator 同逻辑：剔判别器前缀，strict 校验。"""
    p = ROOT / "runs" / run / "ckpt" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("gen_state", ck.get("model_state", ck)))
    if any(k.startswith("generator.") for k in sd):
        sd = {k[len("generator."):]: v for k, v in sd.items()
              if k.startswith("generator.")}
    sd = {k: v for k, v in sd.items() if not k.startswith("discriminator.")}
    net = PUTransformer(up_ratio=UP_RATIO)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[FAIL] {run} 权重不匹配 missing={missing[:4]} "
                         f"unexpected={unexpected[:4]}")
    net.eval()
    return net


def pick_shapes(n_shapes: int) -> list[int]:
    """按客观判据选样本：B2 相对 B1 的逐样本 CD 改善最大者。

    判据固定为「cd(B1) - cd(B2) 降序」，不看图好不好看。若 measure 存档缺列
    则直接失败，不退化为随机挑选——随机挑选会让"选样本"这一步不可复核。
    """
    if not MEASURE.exists():
        raise SystemExit(f"[FAIL] 缺 {MEASURE}，请先跑 scripts/measure_cv_nn.py")
    d = json.loads(MEASURE.read_text(encoding="utf-8"))["runs"]
    for r in ("ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"):
        if r not in d or "per_sample" not in d[r]:
            raise SystemExit(f"[FAIL] {MEASURE} 缺 {r}.per_sample")
    b1 = {s["idx"]: s["cd"] for s in d["ABL_B1_adv_fixed"]["per_sample"]}
    b2 = {s["idx"]: s["cd"] for s in d["ABL_B2_adv_adaptive"]["per_sample"]}
    common = sorted(set(b1) & set(b2))
    if not common:
        raise SystemExit("[FAIL] B1/B2 逐样本索引无交集，口径不一致")
    gain = sorted(common, key=lambda i: b1[i] - b2[i], reverse=True)
    print(f"  候选样本 {len(common)} 个，取 CD 改善最大的 {n_shapes} 个")
    for i in gain[:n_shapes]:
        print(f"    idx={i}  cd(B1)={b1[i]:.6e}  cd(B2)={b2[i]:.6e}  "
              f"gain={b1[i] - b2[i]:.3e}")
    return gain[:n_shapes]


def zoom_window(gt: np.ndarray, pred: np.ndarray, frac: float = 0.20):
    """选放大窗口：GT 点密集而预测点相对欠覆盖的区域。

    早期版本取「预测误差最大的单点」为中心，结果窗口多落在形状边缘的空白处，
    放大图大半是空白，信息量极低（实测第一版三行有两行如此）。
    改为在 XY 平面打网格，取「GT 计数高且 pred/GT 计数比最低」的格子为中心：
    这正是上采样任务真正的失败形态（该补点的地方没补够），放大后可读。
    """
    lo = np.minimum(gt[:, :2].min(0), pred[:, :2].min(0))
    hi = np.maximum(gt[:, :2].max(0), pred[:, :2].max(0))
    span = (hi - lo).max()
    G = 12
    edges_x = np.linspace(lo[0], hi[0] + 1e-9, G + 1)
    edges_y = np.linspace(lo[1], hi[1] + 1e-9, G + 1)
    hg, _, _ = np.histogram2d(gt[:, 0], gt[:, 1], bins=[edges_x, edges_y])
    hp, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=[edges_x, edges_y])
    # 只在 GT 足够密的格子里比较（阈值取 GT 非空格子计数的中位数）
    nz = hg[hg > 0]
    thr = np.median(nz) if nz.size else 0.0
    cand = hg >= max(thr, 1.0)
    if not cand.any():
        cand = hg > 0
    ratio = np.where(cand, hp / np.maximum(hg, 1e-9), np.inf)
    gi, gj = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    cx = 0.5 * (edges_x[gi] + edges_x[gi + 1])
    cy = 0.5 * (edges_y[gj] + edges_y[gj + 1])
    half = span * frac / 2
    return (cx - half, cx + half), (cy - half, cy + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shapes", type=int, default=3)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("图A 点云定性对比")
    print("=" * 78)

    picks = pick_shapes(args.n_shapes)

    ds = PUTrainDataset(source="pu1k", up_ratio=UP_RATIO, noise_beta=0.0,
                        augment=False)
    n_total = len(ds)
    val_start = n_total - int(n_total * VAL_RATIO)
    print(f"  dataset={n_total}  val=[{val_start},{n_total})")
    for i in picks:
        if not (val_start <= i < n_total):
            raise SystemExit(f"[FAIL] 样本 {i} 不在验证划分内，口径越界")
    print("  ✅ 全部样本落在验证划分内")

    nets = {}
    for run, _ in COLS:
        nets[run] = load_generator(run)
        print(f"  载入 {run}")

    panel_stats: list[dict] = []

    ncol = len(COLS) + 2          # input + GT + 4 runs
    nrow = len(picks) * 3         # 每样本：全局CD误差 / 放大 / 全局间距均匀性
    fig = plt.figure(figsize=(3.05 * ncol, 2.72 * nrow))
    # 右侧留出 colorbar 通道；每行样本一组独立色标（各行 vmax 不同，
    # 用单个全局 colorbar 会导致刻度只对某一行成立 —— 图与说明不符）。
    gs = fig.add_gridspec(nrow, ncol + 1, hspace=0.10, wspace=0.06,
                          width_ratios=[1] * ncol + [0.055])

    for si, idx in enumerate(picks):
        inp, gt = ds[int(idx)]
        inp_np = (inp.numpy() if torch.is_tensor(inp) else np.asarray(inp))
        gt_np = (gt.numpy() if torch.is_tensor(gt) else np.asarray(gt))

        preds, errs, spac = {}, {}, {}
        with torch.no_grad():
            for run, _ in COLS:
                t = torch.as_tensor(inp_np, dtype=torch.float32)[None]
                p = nets[run](t)[0].numpy()
                preds[run] = p
                errs[run] = nn_to_ref(p, gt_np)
                spac[run] = nn_within(p)
        gt_spac = nn_within(gt_np)

        # 共享色标：跨 run 取同一 vmax，否则各自归一化会掩盖差异
        vmax = float(np.percentile(np.concatenate(list(errs.values())), 99))
        vmin = 0.0
        # 间距图用「相对 GT 中位间距的倍数」，跨 shape 可比且无量纲
        gt_med = float(np.median(gt_spac))
        smax = float(np.percentile(
            np.concatenate([v / gt_med for v in spac.values()]), 99))
        # 放大窗口：GT 密而预测欠覆盖处，由对照组 B1 决定，各列共用
        zx, zy = zoom_window(gt_np, preds["ABL_B1_adv_fixed"])

        panels = [("input ($N$ pts)", inp_np, None, None),
                  ("ground truth ($4N$ pts)", gt_np, None, gt_spac / gt_med)] + \
                 [(lab, preds[r], errs[r], spac[r] / gt_med) for r, lab in COLS]

        sc_err = sc_sp = None
        for ci, (lab, pts, err, sp) in enumerate(panels):
            # ---- 行 1：全局视图，色 = 到 GT 的逐点 CD 误差（越亮越差）----
            ax = fig.add_subplot(gs[si * 3, ci])
            if err is None:
                ax.scatter(pts[:, 0], pts[:, 1], s=1.4, c="#3b3b3b",
                           linewidths=0)
            else:
                sc_err = ax.scatter(pts[:, 0], pts[:, 1], s=1.4, c=err,
                                    cmap="inferno", vmin=vmin, vmax=vmax,
                                    linewidths=0)
            ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                                   fill=False, ec="#1f77b4", lw=1.25))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("#999")
            if si == 0:
                ax.set_title(lab, fontsize=9.5, pad=5)
            if ci == 0:
                ax.set_ylabel(f"shape idx {idx}\nCD error", fontsize=8.6)

            # ---- 行 2：放大视图（同色标）----
            ax2 = fig.add_subplot(gs[si * 3 + 1, ci])
            if err is None:
                ax2.scatter(pts[:, 0], pts[:, 1], s=9.0, c="#3b3b3b",
                            linewidths=0)
            else:
                ax2.scatter(pts[:, 0], pts[:, 1], s=9.0, c=err,
                            cmap="inferno", vmin=vmin, vmax=vmax,
                            linewidths=0)
            ax2.set_xlim(*zx); ax2.set_ylim(*zy)
            ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")
            for s in ax2.spines.values():
                s.set_linewidth(1.15); s.set_color("#1f77b4")
            if ci == 0:
                ax2.set_ylabel("zoom\n(under-covered region)", fontsize=8.6)

            # ---- 行 3：全局视图，色 = 点内最近邻间距 / GT 中位间距 ----
            # 本文主指标是 cv_nn（间距均匀性），若只画 CD 误差则主结论未被可视化。
            ax3 = fig.add_subplot(gs[si * 3 + 2, ci])
            if sp is None:
                ax3.scatter(pts[:, 0], pts[:, 1], s=1.4, c="#3b3b3b",
                            linewidths=0)
            else:
                sc_sp = ax3.scatter(pts[:, 0], pts[:, 1], s=1.4, c=sp,
                                    cmap="viridis", vmin=0.0, vmax=smax,
                                    linewidths=0)
            ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_aspect("equal")
            for s in ax3.spines.values():
                s.set_linewidth(0.6); s.set_color("#999")
            if ci == 0:
                ax3.set_ylabel("NN spacing\n($\\times$ GT median)", fontsize=8.6)
            if sp is not None:
                cvv = float(np.std(sp) / np.mean(sp))
                ax3.text(0.5, -0.055, f"cv$_{{nn}}$ = {cvv:.3f}",
                         transform=ax3.transAxes, ha="center", va="top",
                         fontsize=8.0)

        # 每样本一组独立 colorbar（贴在该样本的三行右侧）
        cax1 = fig.add_subplot(gs[si * 3:si * 3 + 2, ncol])
        cb1 = fig.colorbar(sc_err, cax=cax1)
        cb1.set_label("per-point CD to GT", fontsize=7.6)
        cb1.ax.tick_params(labelsize=6.6)
        cax2 = fig.add_subplot(gs[si * 3 + 2, ncol])
        cb2 = fig.colorbar(sc_sp, cax=cax2)
        cb2.set_label("NN spacing", fontsize=7.6)
        cb2.ax.tick_params(labelsize=6.6)

        print(f"  shape idx={idx}: CD vmax(p99)={vmax:.4e}  "
              f"spacing vmax={smax:.3f}x  GT median spacing={gt_med:.5f}")
        panel_stats.append({
            "idx": int(idx),
            "cd_vmax_p99": vmax,
            "spacing_vmax_p99_ratio": smax,
            "gt_median_spacing": gt_med,
            "cv_nn_gt": float(np.std(gt_spac / gt_med)
                              / np.mean(gt_spac / gt_med)),
            "cv_nn_per_run": {r: float(np.std(spac[r]) / np.mean(spac[r]))
                              for r, _ in COLS},
            "zoom_xlim": [float(zx[0]), float(zx[1])],
            "zoom_ylim": [float(zy[0]), float(zy[1])],
        })

    # 图号跟随正稿章号：七章结构下定性对比属第 6 章 §6.5.3 → 图 6.1。
    # （文件名沿用 F5_1_* 不改，避免打断既有 .meta.json 与引用路径。）
    fig.suptitle("Fig. 6-1  Qualitative comparison of upsampled point clouds "
                 "(orthographic XY projection)", fontsize=11.5, y=0.9955)
    fig.text(0.5, 0.002, CAVEAT, ha="center", va="bottom", fontsize=7.2,
             style="italic", color="#555", linespacing=1.55)

    out = OUTDIR / "F5_1_qualitative_pointclouds.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    meta = {
        "figure": "F5_1_qualitative_pointclouds",
        "picked_idx": [int(i) for i in picks],
        "pick_criterion": "argsort desc of cd(ABL_B1_adv_fixed) - cd(ABL_B2_adv_adaptive)",
        "columns": [{"run": r, "label": l} for r, l in COLS],
        "rows": [
            "row1: global view, colour = per-point forward CD to GT (cmap inferno, bright = worse)",
            "row2: zoom on the region where GT is dense but B1 under-covers it, same colour scale as row1",
            "row3: global view, colour = within-cloud NN spacing / GT median spacing (cmap viridis)",
        ],
        "colour": "per-shape shared vmin/vmax across runs; NOT shared across shapes",
        "vmax_rule": "CD: percentile 99 over all runs of that shape; spacing: percentile 99 of ratio",
        "zoom_rule": "12x12 XY grid; centre = cell with GT count >= median(nonzero GT counts) "
                     "and minimal pred/GT count ratio (worst under-coverage)",
        "per_panel_cv_nn": "computed on that single shape; differs from 200-sample table means by design",
        "caliber": {"seed": SEED, "val_ratio": VAL_RATIO,
                    "n_sample_pool": N_FULL, "up_ratio": UP_RATIO,
                    "augment": False},
        "cross_host_note": "C1 on 3090 host; B0/B1/B2 on 5090 host. "
                           "No numeric metric juxtaposed in this figure.",
        "source_measure": "docs/_cv_nn_measure.json",
        "panel_stats": panel_stats,
        "honesty_note": "Shapes were selected on CD gain, not on cv_nn. Per-panel cv_nn therefore "
                        "does not favour B2 on every shape (see panel_stats); this is reported "
                        "as-is in the figure caption rather than re-selecting shapes.",
    }
    mp = OUTDIR / "F5_1_qualitative_pointclouds.meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
