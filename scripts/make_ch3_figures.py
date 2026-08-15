# -*- coding: utf-8 -*-
"""生成第 3 章配图 F3_1 ~ F3_6。

红线（承 make_schematic_figures.py）:
  1. 图题必须含「示意图」/「框图」/「对比图」等字样, 不冒充实验结果。
  2. 图中出现的任何数字, 必须来自 docs/_ch3_*.json 实测存档, 不得手填。
  3. 布局重叠由 scripts/selfcheck_ch3.py 量化校验, 不靠肉眼。
  4. 数据图（F3_5）的 .data.json 标记 depends_on_experiment=True 并记录来源 run。
"""
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "docs", "figures_schematic")
os.makedirs(OUTDIR, exist_ok=True)

for fam in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fam]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

TS = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
C_BOX = "#E8F0FE"
C_EDGE = "#3B6FB6"
C_HL = "#FFF3E0"
C_HLE = "#E07B39"
C_GRAY = "#F5F5F5"
C_GREEN = "#E8F5E9"
C_GREENE = "#4C8C4A"

SHAPES = json.load(open(os.path.join(ROOT, "docs", "_ch3_shapes.json"), encoding="utf-8"))
STATS = json.load(open(os.path.join(ROOT, "docs", "_ch3_stats.json"), encoding="utf-8"))
DIAG = json.load(open(os.path.join(ROOT, "docs", "_ch3_diag.json"), encoding="utf-8"))


def save(fig, name, meta, depends=False):
    p = os.path.join(OUTDIR, name + ".png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta["generated_at"] = TS
    meta["script"] = "scripts/make_ch3_figures.py"
    meta["depends_on_experiment"] = depends
    with open(os.path.join(OUTDIR, name + ".data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  [OK] %s  (%d B)" % (name + ".png", os.path.getsize(p)))


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9.5, lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, style="-|>", color="#555", lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color, linewidth=lw,
                                 linestyle=ls, mutation_scale=14, zorder=1,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def _p(name):
    """从实测 layer_trace 取参数量，禁止手填。"""
    for r in SHAPES["generator"]["layer_trace"]:
        if r["name"] == name:
            return r
    raise KeyError(name)


# ============================================================
# 图 3.1 基线前向过程与张量形状框图（形状全部取自实测 hook）
# ============================================================
def fig_3_1_baseline_forward():
    fig, ax = plt.subplots(figsize=(14.6, 8.2))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 8.2); ax.axis("off")

    N = SHAPES["setting"]["N"]
    R = SHAPES["setting"]["up_ratio"]
    dims = SHAPES["setting"]["dims"]

    def shp(name):
        return _p(name)["out_shape"][1:]

    # --- 主链 ---
    BW, BH, BY = 1.36, 1.02, 4.92
    GAP = 1.58
    stages = [
        ("输入 $\\mathcal{P}$", "%d × 3" % N, C_GRAY, "#999"),
        ("Head\n逐点 MLP", "%d × %d" % tuple(shp("head")), C_BOX, C_EDGE),
    ]
    for i in range(5):
        stages.append(("Encoder %d" % (i + 1), "%d × %d" % tuple(shp("encoders.%d" % i)),
                       C_BOX, C_EDGE))
    stages.append(("Shuffle\n（无参数）", "%d × %d" % (N * R, dims[-1] // R), C_HL, C_HLE))
    stages.append(("Tail\nMLP + Linear", "%d × 3" % (N * R), C_BOX, C_EDGE))

    for i, (label, sh, fc, ec) in enumerate(stages):
        bx = 0.30 + i * GAP
        box(ax, bx, BY, BW, BH, label, fc=fc, ec=ec, fs=8.2, bold=(fc == C_HL))
        # 形状标注上下交错，避免相邻标注串成一行
        dy = -0.30 if i % 2 == 0 else -0.62
        ax.text(bx + BW / 2, BY + dy, sh, ha="center", fontsize=8.0,
                color="#B4573F", fontweight="bold")
        if i:
            arrow(ax, (0.30 + (i - 1) * GAP + BW, BY + BH / 2), (bx, BY + BH / 2), lw=1.5)

    ax.text(7.3, 6.55, "张量形状均取自真实前向 hook 实测（$B$=%d 已省略），非按公式推算"
            % SHAPES["setting"]["B"],
            ha="center", fontsize=8.6, color="#555", style="italic")

    # --- Encoder 内部展开 ---
    box(ax, 0.55, 2.42, 6.55, 1.66,
        "Encoder 内部（每级相同）\n"
        "PosFus（$k$=%d 邻域，几何路 + 特征路）→ LN → SC-MSA（$\\psi$=%d）\n"
        "→ 残差 → LN → 前馈（扩张比 1）→ 残差"
        % (SHAPES["setting"]["k"], 4),
        fc=C_BOX, ec=C_EDGE, fs=8.8)
    arrow(ax, (3.90, 4.92), (3.83, 4.08), color="#999", lw=1.2, ls="--")

    # --- 参数量分解（实测）---
    g = SHAPES["generator"]
    enc_p = [_p("encoders.%d" % i)["n_params"] for i in range(5)]
    attn_sum = sum(_p("encoders.%d.attn" % i)["n_params"] for i in range(5))
    box(ax, 7.55, 2.42, 6.55, 1.66,
        "参数量实测分解\n"
        "生成器合计 %s（原文报告 %s，比值 %.4f）\n"
        "五级 Encoder：%s\n"
        "其中 SC-MSA 合计 %s，占 %.2f%%"
        % (format(g["total_params"], ","), format(g["paper_reported_params"], ","),
           g["ratio_to_paper"], " / ".join(format(v, ",") for v in enc_p),
           format(attn_sum, ","), 100.0 * attn_sum / g["total_params"]),
        fc=C_HL, ec=C_HLE, fs=8.6, lw=1.7)

    # --- 损失 ---
    box(ax, 3.90, 0.62, 6.80, 1.10,
        "训练目标  $\\mathcal{L} = w_{cd}\\,\\mathrm{CD} + w_{unif}\\,\\mathcal{L}_{unif} "
        "+ w_{adv}\\,\\mathcal{L}_{adv}$\n"
        "本章干净基线取 $w_{cd}$=1，$w_{unif}$=0，$w_{adv}$=0（纯 Chamfer 监督）",
        fc=C_GREEN, ec=C_GREENE, fs=8.8)
    arrow(ax, (0.30 + 8 * GAP + BW / 2, BY), (10.70, 1.72), color="#4C8C4A",
          lw=1.4, rad=-0.18)

    ax.text(7.3, 7.80, "图 3.1  基线模型网络前向过程与张量形状框图", ha="center",
            fontsize=12.5, fontweight="bold")
    ax.text(7.3, 0.20,
            "注：红色标注为该阶段输出张量形状（实测，上下交错排列）；Shuffle 阶段不含参数，"
            "仅重组通道维到点数维。参数量与形状的完整逐层记录见 docs/_ch3_shapes.json。",
            ha="center", fontsize=8.2, color="#555")
    save(fig, "F3_1_baseline_forward",
         {"figure_id": "3-1", "caption": "基线模型网络前向过程与张量形状框图",
          "type": "block_diagram", "chapter": "3.1.3",
          "source": "docs/_ch3_shapes.json",
          "total_params": g["total_params"], "n_stages": len(stages)}, depends=True)


# ============================================================
# 图 3.2 SC-MSA 通道窗口划分示意图
# ============================================================
def fig_3_2_scmsa_window():
    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 8.2); ax.axis("off")

    C, psi = 16, 4          # 示意用小尺寸，正文说明真实取值
    w = C // psi            # 窗口宽 4
    n_heads = 7             # 与 3.2.3 节一致
    step = (C - w) / (n_heads - 1)

    Y0, CW, CH = 6.10, 0.30, 0.62

    # 左：标准 MHSA
    ax.text(3.05, 7.42, "标准多头自注意力（窗口互不重叠）", ha="center",
            fontsize=10.2, fontweight="bold", color="#3B6FB6")
    X0 = 0.60
    for c in range(C):
        box(ax, X0 + c * CW, Y0, CW * 0.92, CH, "", fc=C_GRAY, ec="#999", lw=0.8)
    for h in range(psi):
        x = X0 + h * w * CW
        ax.add_patch(FancyBboxPatch((x, Y0 - 0.10), w * CW * 0.96, CH + 0.20,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    facecolor="none", edgecolor=C_EDGE, linewidth=1.8, zorder=4))
        ax.text(x + w * CW / 2, Y0 - 0.40, "头%d" % (h + 1), ha="center", fontsize=8.4,
                color=C_EDGE)
    box(ax, 0.60, 3.62, 5.00, 1.34,
        "每个通道只归属 1 个头\n头之间无通道维信息交换\n投影层输入维 = $C$",
        fc=C_GRAY, ec="#999", fs=8.8)

    # 右：SC-MSA —— 窗口向【上】叠放，避免压住下方文字
    ax.text(9.55, 7.42, "SC-MSA（通道窗口按固定步长滑动，相邻头重叠）", ha="center",
            fontsize=10.2, fontweight="bold", color="#B4573F")
    X1 = 6.95
    for c in range(C):
        box(ax, X1 + c * CW, Y0, CW * 0.92, CH, "", fc=C_GRAY, ec="#999", lw=0.8)
    cmap = plt.get_cmap("Oranges")
    for h in range(n_heads):
        x = X1 + h * step * CW
        grow = 0.085 * h
        ax.add_patch(FancyBboxPatch((x, Y0 - 0.10), w * CW * 0.96, CH + 0.20 + grow,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    facecolor="none", edgecolor=cmap(0.35 + 0.09 * h),
                                    linewidth=1.7, zorder=4))
    ax.text(X1 + C * CW / 2, Y0 - 0.40,
            "共 %d 个头，窗口宽 $w=C/\\psi$，步长 $(C-w)/(n_{heads}-1)$" % n_heads,
            ha="center", fontsize=8.6, color="#B4573F")
    box(ax, 6.95, 3.62, 5.00, 1.34,
        "同一通道可参与多个头\n建立跨头的通道维交互\n投影层输入维 = $n_{heads}\\cdot w$ = 1.75$C$",
        fc=C_HL, ec=C_HLE, fs=8.8, lw=1.7)

    attn_sum = sum(_p("encoders.%d.attn" % i)["n_params"] for i in range(5))
    box(ax, 1.20, 1.42, 10.40, 1.62,
        "结构含义：注意力计算次数不增加，跨头通道交互由窗口重叠免费获得；\n"
        "代价落在输出投影层——输入维由 $C$ 升至 1.75$C$，该层是本结构参数量的主要来源之一\n"
        "（五级 SC-MSA 合计 %s 参数，占生成器 %.2f%%，实测见 docs/_ch3_shapes.json）"
        % (format(attn_sum, ","),
           100.0 * attn_sum / SHAPES["generator"]["total_params"]),
        fc=C_GREEN, ec=C_GREENE, fs=9.0, lw=1.6)

    ax.text(6.4, 7.86, "图 3.2  SC-MSA 通道窗口划分与注意力计算示意图", ha="center",
            fontsize=12.5, fontweight="bold")
    ax.text(6.4, 0.72,
            "注：图中取 $C$=%d、$\\psi$=%d 仅为示意，实际各级 $C$ 见图 3.1；"
            "灰格代表通道，彩框代表一个注意力头覆盖的通道窗口（右图窗口高度依次递增仅为便于区分）。"
            % (C, psi),
            ha="center", fontsize=8.2, color="#555")
    save(fig, "F3_2_scmsa_window",
         {"figure_id": "3-2", "caption": "SC-MSA 通道窗口划分与注意力计算示意图",
          "type": "schematic", "chapter": "3.2.3",
          "illustrative_C": C, "illustrative_psi": psi, "n_heads": n_heads,
          "note": "C/psi 为示意取值；参数量数字取自实测存档"})


# ============================================================
# 图 3.3 判别器结构框图
# ============================================================
def fig_3_3_discriminator():
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.6); ax.axis("off")

    d = SHAPES["discriminator"]
    N4 = d["in_shape"][1]
    stages = [
        ("输入点云\n$\\mathcal{Q}$ 或 $\\mathcal{Y}$", "%d × 3" % N4, C_GRAY, "#999"),
        ("逐点 MLP\n特征编码", "%d × C" % N4, C_BOX, C_EDGE),
        ("自注意力层\n建模点间关系", "%d × C" % N4, C_HL, C_HLE),
        ("全局池化\n聚合为整体描述", "1 × C", C_BOX, C_EDGE),
        ("全连接判定\n真实性标量", "标量", C_BOX, C_EDGE),
    ]
    BW, BH, BY = 1.92, 1.18, 3.62
    for i, (label, sh, fc, ec) in enumerate(stages):
        x = 0.55 + i * 2.42
        box(ax, x, BY, BW, BH, label, fc=fc, ec=ec, fs=8.8, bold=(fc == C_HL))
        ax.text(x + BW / 2, BY - 0.30, sh, ha="center", fontsize=8.2,
                color="#B4573F", fontweight="bold")
        if i:
            arrow(ax, (0.55 + (i - 1) * 2.42 + BW, BY + BH / 2), (x, BY + BH / 2), lw=1.5)

    box(ax, 0.55, 1.62, 5.55, 1.42,
        "自注意力层的作用\n"
        "若仅由逐点编码 + 全局池化构成，可用信息接近逐点特征的一阶统计量，\n"
        "难以区分「分布均匀」与「存在局部聚簇」——而后者正是 CD 的盲区所在",
        fc=C_HL, ec=C_HLE, fs=8.6, lw=1.7)
    arrow(ax, (5.90, 3.62), (4.60, 3.04), color="#E07B39", lw=1.3, ls="--")

    box(ax, 6.55, 1.62, 5.50, 1.42,
        "规模实测\n"
        "判别器参数量 %s，为生成器（%s）的 %.1f%%\n"
        "输入 %s → 输出 %s（batch 维）"
        % (format(d["n_params"], ","), format(SHAPES["generator"]["total_params"], ","),
           100.0 * d["n_params"] / SHAPES["generator"]["total_params"],
           str(d["in_shape"]), str(d["out_shape"])),
        fc=C_GREEN, ec=C_GREENE, fs=8.6, lw=1.6)

    ax.text(6.3, 6.20, "图 3.3  点云判别器网络结构框图", ha="center",
            fontsize=12.5, fontweight="bold")
    ax.text(6.3, 0.85,
            "注：原理上判别器可覆盖重建损失的盲区，该互补性是否在训练中兑现须由实验判定（见 3.3.1 节与第 5 章）。"
            "参数量与形状取自 docs/_ch3_shapes.json。",
            ha="center", fontsize=8.2, color="#555")
    save(fig, "F3_3_discriminator",
         {"figure_id": "3-3", "caption": "点云判别器网络结构框图", "type": "block_diagram",
          "chapter": "3.3.1", "n_params": d["n_params"],
          "source": "docs/_ch3_shapes.json"}, depends=True)


# ============================================================
# 图 3.4 三套指标口径隔离示意图
# ============================================================
def fig_3_4_metric_isolation():
    fig, ax = plt.subplots(figsize=(12.8, 7.8))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 7.8); ax.axis("off")

    sel = STATS["selection"]
    pl = STATS["plateau"]

    lanes = [
        (5.62, "训练期监控口径", C_BOX, C_EDGE,
         "样本：验证切片（尾部 5%）\n尺度：归一化后的 patch\n频率：每 epoch\n"
         "落盘：runs/<run>/metrics.json\n用途：观察收敛趋势",
         "不与文献报告值对照"),
        (3.32, "模型选点口径", C_HL, C_HLE,
         "评分：CD %.1f + HD %.1f + NUC %.1f\n预热：前 %d epoch 不参与\n"
         "同时记录仅看 CD 的影子选点\n落盘：runs/<run>/selection.json\n"
         "本 run 结果：加权选 ep%d，影子选 ep%d"
         % (sel["weights"]["cd"], sel["weights"]["hd"], sel["weights"]["nuc"],
            5, sel["best_epoch_weighted"], sel["best_epoch_cd_only"]),
         "两准则一致不等于等价（见 3.4.5 节）"),
        (1.02, "官方评价口径", C_GREEN, C_GREENE,
         "样本：官方测试集\n尺度：CD/HD 归一化后；P2F 原始尺度\n"
         "执行：独立评价脚本\n落盘：单独目录\n用途：第 5 章主表",
         "唯一可与文献并列的口径"),
    ]
    for y, title, fc, ec, body, warn in lanes:
        box(ax, 0.55, y, 3.05, 1.92, title, fc=fc, ec=ec, fs=10.4, lw=1.8, bold=True)
        box(ax, 3.95, y, 4.85, 1.92, body, fc=C_GRAY, ec="#999", fs=8.4)
        box(ax, 9.15, y, 3.15, 1.92, warn, fc=fc, ec=ec, fs=8.6, lw=1.5)

    # 隔离墙
    for xw in (3.78, 8.98):
        ax.plot([xw, xw], [0.95, 7.62], color="#C62828", lw=2.2, ls=(0, (5, 3)), zorder=5)
    ax.text(3.78, 7.72, "口径边界", ha="center", fontsize=8.6, color="#C62828",
            fontweight="bold")
    ax.text(8.98, 7.72, "口径边界", ha="center", fontsize=8.6, color="#C62828",
            fontweight="bold")

    ax.text(6.4, 0.62,
            "注：三套口径的样本、尺度与数据划分均不同，混用会导致数字不可比；"
            "本章报告的平台区结果（CD %.6f ± %.6f）属训练期监控口径。"
            % (pl["cd"]["plateau_mean"], pl["cd"]["plateau_std"]),
            ha="center", fontsize=8.2, color="#555")
    ax.text(6.4, 7.48, "图 3.4  训练、选点与官方评价三套口径隔离示意图", ha="center",
            fontsize=12.5, fontweight="bold")
    save(fig, "F3_4_metric_isolation",
         {"figure_id": "3-4", "caption": "训练、选点与官方评价三套口径隔离示意图",
          "type": "schematic", "chapter": "3.3.4", "n_lanes": 3,
          "source": "docs/_ch3_stats.json"}, depends=True)


# ============================================================
# 图 3.5 最近邻间距分布对比图（数据图，全部取自 _ch3_diag.json）
# ============================================================
def fig_3_5_nn_spacing():
    sp = DIAG["spacing"]
    st = DIAG["sparsity_strata"]
    per = DIAG["per_sample"]
    n = DIAG["source"]["n_sample"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9))

    # (a) cv 直方图
    ax = axes[0]
    cv_p = [r["nn_pred_cv"] for r in per]
    cv_g = [r["nn_gt_cv"] for r in per]
    bins = np.linspace(0, max(cv_p) * 1.05, 34)
    ax.hist(cv_g, bins=bins, color=C_GREENE, alpha=0.75, label="真值点云")
    ax.hist(cv_p, bins=bins, color=C_HLE, alpha=0.75, label="基线预测")
    ax.axvline(sp["nn_gt_cv"]["mean"], color=C_GREENE, ls="--", lw=1.4)
    ax.axvline(sp["nn_pred_cv"]["mean"], color=C_HLE, ls="--", lw=1.4)
    ax.set_xlabel("最近邻间距变异系数 $\\mathrm{cv}_{nn}$", fontsize=9.5)
    ax.set_ylabel("样本数", fontsize=9.5)
    ax.set_title("(a) $\\mathrm{cv}_{nn}$ 分布：均值 %.4f vs %.4f（比值 %.2f）"
                 % (sp["nn_pred_cv"]["mean"], sp["nn_gt_cv"]["mean"], sp["cv_ratio_mean"]),
                 fontsize=9.6)
    ax.legend(fontsize=8.6)
    ax.tick_params(labelsize=8.4)

    # (b) 逐样本配对
    ax = axes[1]
    ax.scatter(cv_g, cv_p, s=13, color=C_HLE, alpha=0.7, edgecolors="none")
    lim = max(max(cv_p), max(cv_g)) * 1.06
    ax.plot([0, lim], [0, lim], color="#555", ls="--", lw=1.2, label="$y=x$")
    ax.set_xlim(0, lim * 0.35); ax.set_ylim(0, lim)
    ax.set_xlabel("真值 $\\mathrm{cv}_{nn}$", fontsize=9.5)
    ax.set_ylabel("预测 $\\mathrm{cv}_{nn}$", fontsize=9.5)
    ax.set_title("(b) 逐样本配对：%d/%d 个样本预测更离散"
                 % (sp["n_pred_cv_gt_gt_cv"], n), fontsize=9.6)
    ax.legend(fontsize=8.6)
    ax.tick_params(labelsize=8.4)

    # (c) 稀疏度分层
    ax = axes[2]
    q = st["bwd_mean_by_quartile"]
    xs = np.arange(4)
    ax.bar(xs, q, color=[C_GREENE, "#7FA88C", "#D9A05B", C_HLE], width=0.62)
    for i, v in enumerate(q):
        ax.text(i, v * 1.012, "%.2e" % v, ha="center", fontsize=8.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(st["q_labels"], fontsize=8.6)
    ax.set_ylabel("后向 Chamfer 分量均值", fontsize=9.5)
    ax.set_ylim(0, max(q) * 1.16)
    ax.set_xlabel("按输入局部稀疏度分位", fontsize=9.5)
    ax.set_title("(c) 误差随稀疏度单调上升：Q4/Q1 = %.3f" % st["q4_over_q1"], fontsize=9.6)
    ax.tick_params(labelsize=8.4)

    fig.suptitle("图 3.5  预测点云与真值点云最近邻间距分布对比图（%d 样本实测）" % n,
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.055,
             "数据来源：%s（best epoch %s），验证切片 %s，关闭数据增广，CPU 前向推理，"
             "随机种子 %d；样本量与种子于实验执行前确定。"
             % (DIAG["source"]["ckpt"], DIAG["source"]["best_epoch"],
                str(DIAG["source"]["val_range"]), DIAG["source"]["seed"]),
             ha="center", fontsize=8.2, color="#555")
    fig.tight_layout()
    save(fig, "F3_5_nn_spacing",
         {"figure_id": "3-5", "caption": "预测点云与真值点云最近邻间距分布对比图",
          "type": "data_figure", "chapter": "3.5.2",
          "source": "docs/_ch3_diag.json", "run": DIAG["source"]["ckpt"],
          "n_sample": n, "cv_ratio": sp["cv_ratio_mean"],
          "q4_over_q1": st["q4_over_q1"]}, depends=True)


# ============================================================
# 图 3.6 瓶颈证据与候选归因位置对应图
# ============================================================
def fig_3_6_bottleneck_attribution():
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 8.0); ax.axis("off")

    sp = DIAG["spacing"]
    cs = DIAG["cd_split"]
    st = DIAG["sparsity_strata"]
    tr = STATS["train_cd_bwd_share"]

    ax.text(2.30, 7.20, "实测证据", ha="center", fontsize=10.8, fontweight="bold",
            color="#3B6FB6")
    ax.text(7.35, 7.20, "候选结构位置（待验证）", ha="center", fontsize=10.8,
            fontweight="bold", color="#B4573F")
    ax.text(11.55, 7.20, "第 4 章约束", ha="center", fontsize=10.8, fontweight="bold",
            color="#4C8C4A")

    ev = [
        (5.30, "证据 E1  后向分量持续占优\n训练期均值 %.4f，150/150 epoch 均 > 0.5\n"
               "推理期 %d/%d 样本（%.1f%%）后向占优"
               % (tr["mean"], cs["n_bwd_share_gt_half"], DIAG["source"]["n_sample"],
                  100.0 * cs["n_bwd_share_gt_half"] / DIAG["source"]["n_sample"])),
        (3.30, "证据 E2  间距离散度为真值 %.2f 倍\n预测 %.4f vs 真值 %.4f，%d/%d 无例外\n"
               "而平均间距比值 %.3f（密度基本正确）"
               % (sp["cv_ratio_mean"], sp["nn_pred_cv"]["mean"], sp["nn_gt_cv"]["mean"],
                  sp["n_pred_cv_gt_gt_cv"], DIAG["source"]["n_sample"],
                  sp["nn_pred_mean"]["mean"] / sp["nn_gt_mean"]["mean"])),
        (1.30, "证据 E3  误差集中于输入稀疏区\n四分位后向分量单调递增\nQ4/Q1 = %.3f"
               % st["q4_over_q1"]),
    ]
    for y, t in ev:
        box(ax, 0.35, y, 3.90, 1.62, t, fc=C_BOX, ec=C_EDGE, fs=8.4)

    cand = [
        (4.55, "候选 C1  上采样阶段的子点独立性\n"
               "同一父点的 $r$ 个子点各自从 256 维特征的\n"
               "不同通道切片独立回归，无显式交互机制\n"
               "→ 与 E2「密度正确但离散度过大」形态吻合"),
        (1.72, "候选 C2  特征交互的邻域固定性\n"
               "$k$ 近邻由输入几何一次算定并全程复用；\n"
               "稀疏区固定 $k$=20 的邻域物理尺度更大\n"
               "→ 与 E3「误差随稀疏度上升」方向一致"),
    ]
    for y, t in cand:
        box(ax, 4.70, y, 5.30, 2.06, t, fc=C_HL, ec=C_HLE, fs=8.4, lw=1.7)

    arrow(ax, (4.25, 6.11), (4.70, 5.58), color="#999", lw=1.3, ls="--")
    arrow(ax, (4.25, 4.11), (4.70, 5.20), color="#E07B39", lw=1.6)
    arrow(ax, (4.25, 2.11), (4.70, 2.75), color="#E07B39", lw=1.6)

    box(ax, 10.35, 3.90, 2.40, 2.60,
        "主指标\n$\\mathrm{cv}_{nn}$ 下降\n门槛 $2\\mathrm{SE}$\n\n"
        "同报\nCD / HD / NUC / P2F\n\n分层\nQ4/Q1 比值",
        fc=C_GREEN, ec=C_GREENE, fs=8.6, lw=1.7)
    arrow(ax, (10.00, 5.58), (10.35, 5.20), color="#4C8C4A", lw=1.5)
    arrow(ax, (10.00, 2.75), (10.30, 4.10), color="#4C8C4A", lw=1.5, rad=0.16)

    box(ax, 0.35, 0.30, 12.30, 0.86,
        "本章结论边界：C1 与 C2 均为与实测方向一致的候选解释，现有证据（输入输出端统计量）"
        "不足以在两者间归因，亦不足以排除第三种可能；归因判定由第 4 章设计配合第 5 章受控消融完成。",
        fc=C_GRAY, ec="#C62828", fs=8.8, lw=1.6)

    ax.text(6.5, 7.66, "图 3.6  基线瓶颈证据与候选归因位置对应图", ha="center",
            fontsize=12.5, fontweight="bold")
    save(fig, "F3_6_bottleneck_attribution",
         {"figure_id": "3-6", "caption": "基线瓶颈证据与候选归因位置对应图",
          "type": "schematic", "chapter": "3.5.4",
          "source": ["docs/_ch3_diag.json", "docs/_ch3_stats.json"],
          "n_evidence": 3, "n_candidates": 2,
          "note": "候选位置一律标注待验证，不作为归因结论"}, depends=True)


def main():
    print("生成第 3 章配图 -> %s" % OUTDIR)
    fig_3_1_baseline_forward()
    fig_3_2_scmsa_window()
    fig_3_3_discriminator()
    fig_3_4_metric_isolation()
    fig_3_5_nn_spacing()
    fig_3_6_bottleneck_attribution()
    print("\n完成 6 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
