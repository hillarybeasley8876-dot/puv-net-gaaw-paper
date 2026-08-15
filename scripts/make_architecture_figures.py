# -*- coding: utf-8 -*-
"""
生成第 4 章两张核心架构图 (大道至简, 信息优先):

  F4_1_network_forward.png   链A: 生成器前向过程 + 判别器, 每步标注真实张量形状
  F4_2_train_eval_pipeline.png 链B: 训练与评价全流程, 标注 M1 介入点与产物文件名

所有形状/参数量来自 scripts/probe_architecture.py 实测 (docs/_arch_probe.json),
不允许凭印象填数字。

图规范 (docs/STYLE_GUIDE.md §4):
  * 无渐变/阴影/3D, 黑白可读, 靠线型与灰阶区分
  * 中文字体下禁用 ⚠✓✗✅❌
  * 一图一论点, 图题即论点
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUTDIR = os.path.join(DOCS, "figures_schematic")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

PROBE = json.load(open(os.path.join(DOCS, "_arch_probe.json"), encoding="utf-8"))
G_INFO = PROBE["generator"]
D_INFO = PROBE["discriminator"]
DIMS = G_INFO["dims"]                      # [32,64,128,256,256]
G_PARAMS = G_INFO["n_params"]              # 1152803
D_PARAMS = D_INFO["n_params"]              # 255426

# 统一视觉语言: 只用灰阶 + 线型
FILL_IN = "#f2f2f2"      # 输入/输出数据
FILL_OP = "#ffffff"      # 常规算子
FILL_KEY = "#d9d9d9"     # 关键块(注意力/扩张)
FILL_M1 = "#bfbfbf"      # M1 创新点
EDGE = "#1a1a1a"


def box(ax, x, y, w, h, text, fill=FILL_OP, lw=1.1, ls="-", fs=8.5, weight="normal"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE,
                           linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, text=None, ls="-", lw=1.1, fs=7.6,
          tdx=0.0, tdy=0.13, ha="center"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, linestyle=ls,
                                 color=EDGE, zorder=2,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tdx, (y1 + y2) / 2 + tdy, text,
                ha=ha, va="center", fontsize=fs, zorder=3,
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="none", alpha=0.92))


# =====================================================================
# 图 4.1 链A: 生成器前向过程
# =====================================================================
def fig_forward():
    fig, ax = plt.subplots(figsize=(13.6, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    ax.text(50, 53.6, "图 4.1  生成器前向过程与张量形状（框图）",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 51.0,
            "所有形状与参数量取自 scripts/probe_architecture.py 实测；batch 维以 B 表示，本图示例 B=2",
            ha="center", fontsize=8, color="#444444")

    # ---- 主干流水线 y=38 ----
    y0, h0 = 37.0, 7.2
    box(ax, 0.8, y0, 10.2, h0,
        "输入 patch\n$\\mathcal{P}$\nB × 256 × 3", fill=FILL_IN, fs=8.6)
    arrow(ax, 11.0, y0 + h0 / 2, 14.6, y0 + h0 / 2)
    box(ax, 14.6, y0, 11.6, h0,
        "Head MLP\nLinear(3→32)\n+BN+ReLU\nB × 256 × 32", fs=8.0)

    # 5 个 Encoder
    xs = [30.2, 43.2, 56.2, 69.2, 82.2]
    for i, (x, c) in enumerate(zip(xs, DIMS)):
        arrow(ax, x - 3.9, y0 + h0 / 2, x, y0 + h0 / 2)
        box(ax, x, y0, 11.0, h0,
            "Encoder %d\nPSA + SC-MSA\nB × 256 × %d" % (i + 1, c),
            fill=FILL_KEY, fs=8.2)

    # ---- Encoder 内部展开 (放大说明) y=20 ----
    ax.text(2.0, 32.6, "Encoder 内部结构（以 Encoder $l$ 为例，通道 $C_{l-1}\\!\\to\\!C_l$）：",
            fontsize=9.4, fontweight="bold")

    ey, eh = 23.4, 6.6
    box(ax, 2.0, ey, 14.2, eh,
        "位置融合 PSA\n$g=[\\,p_i\\,\\|\\,p_i\\!-\\!p_j\\,\\|\\,f_i\\,\\|\\,f_i\\!-\\!f_j\\,]$\n"
        "kNN 聚合 + maxpool", fill=FILL_KEY, fs=7.9)
    arrow(ax, 16.2, ey + eh / 2, 21.4, ey + eh / 2, "B×N×$C_l$", tdy=0.95)
    box(ax, 21.4, ey, 14.2, eh,
        "移位通道多头注意力\nSC-MSA\n"
        "$h$ 组窗口宽 $w\\!=\\!C_l/\\psi$", fill=FILL_KEY, fs=7.9)
    arrow(ax, 35.6, ey + eh / 2, 39.6, ey + eh / 2)
    box(ax, 39.6, ey, 12.0, eh,
        "残差 + LayerNorm\n前馈 MLP\nB × N × $C_l$", fs=7.9)

    ax.annotate("", xy=(9.0, y0), xytext=(9.0, ey + eh),
                arrowprops=dict(arrowstyle="-", ls=":", lw=1.0, color="#666666"))
    ax.text(53.4, ey + eh / 2,
            "注：SC-MSA 在通道维滑动窗口做多头注意力，\n"
            "窗口须铺满通道（$(h\\!-\\!1)d\\!+\\!w=C_l$），\n"
            "由模型构造时断言校验，不满足即报错。",
            fontsize=8.0, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#999999", lw=0.8))

    # ---- 上采样与回归 y=9 ----
    ty, th = 9.0, 7.2
    # 从 Encoder5 下行到特征扩张: 竖线走在框右侧空白, 不穿文字
    ax.add_patch(FancyArrowPatch((87.7, y0), (87.7, ty + th),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=EDGE, zorder=2,
                                 shrinkA=0, shrinkB=0))
    ax.text(88.6, (y0 + ty + th) / 2, "特征扩张", fontsize=7.8,
            ha="left", va="center", rotation=90,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92))
    box(ax, 74.6, ty, 16.6, th,
        "特征扩张（pixel-shuffle 1D）\nB × 256 × %d\n→  B × 1024 × %d"
        % (DIMS[-1], DIMS[-1] // G_INFO["up_ratio"]), fill=FILL_KEY, fs=7.9)
    arrow(ax, 74.6, ty + th / 2, 70.6, ty + th / 2)
    box(ax, 56.0, ty, 14.6, th,
        "坐标回归 Tail\nPointMLP(64) + Linear(3)\nB × 1024 × 3", fs=8.0)
    arrow(ax, 56.0, ty + th / 2, 52.0, ty + th / 2)
    box(ax, 39.6, ty, 12.4, th,
        "输出稠密点云\n$\\mathcal{Q}$\nB × 1024 × 3", fill=FILL_IN, fs=8.6)

    # 判别器分支
    arrow(ax, 39.6, ty + th / 2, 35.2, ty + th / 2)
    box(ax, 18.4, ty, 16.8, th,
        "判别器 $D_\\phi$（PU-GAN）\nMLP → 自注意力 → MLP → FC\n"
        "B × 1024 × 3  →  B × 1", fill=FILL_OP, fs=7.9, ls="--")

    # 参数量: 明确写清归属, 避免误读为相邻框的参数量
    ax.text(26.8, ty - 2.0, "判别器 $D_\\phi$ 参数量 %s" % format(D_PARAMS, ","),
            ha="center", fontsize=7.9, style="italic")
    ax.text(73.0, ty - 2.0,
            "生成器 $G_\\theta$ 参数量合计 %s（Head + 5×Encoder + Tail）"
            % format(G_PARAMS, ","),
            ha="center", fontsize=7.9, style="italic")

    # 倍率关系标注 (置于 Encoder5 下方空白, 避开竖排箭头标签)
    ax.text(70.0, 30.4, "上采样倍率 $r=4$：256 → 1024",
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec=EDGE, lw=1.0))

    fig.savefig(os.path.join(OUTDIR, "F4_1_network_forward.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure_id": "F4-1",
        "title": "生成器前向过程与张量形状（框图）",
        "depends_on_experiment": False,
        "source": "scripts/probe_architecture.py -> docs/_arch_probe.json",
        "generator_params": G_PARAMS,
        "discriminator_params": D_PARAMS,
        "dims": DIMS,
        "up_ratio": G_INFO["up_ratio"],
        "input_shape": G_INFO["input_shape"],
        "output_shape": G_INFO["output_shape"],
    }
    json.dump(meta, open(os.path.join(OUTDIR, "F4_1_network_forward.data.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# =====================================================================
# 图 4.2 链B: 训练与评价流程 (含 M1 介入点)
# =====================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.6, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    ax.text(50, 51.8, "图 4.2  训练与评价流程及梯度自适应对抗权重的介入点（流程图）",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 49.4,
            "深灰块为本文改进 M1；虚线框标注各阶段落盘产物；训练期监控口径与论文主表口径严格隔离",
            ha="center", fontsize=8, color="#444444")

    # --- 第一行: 数据到损失 ---
    y1, h = 39.0, 7.0
    box(ax, 1.0, y1, 12.5, h,
        "数据采样\nPU1K patch\n256 → 1024", fill=FILL_IN, fs=8.2)
    arrow(ax, 13.5, y1 + h / 2, 17.2, y1 + h / 2)
    box(ax, 17.2, y1, 12.5, h, "生成器 $G_\\theta$\n前向\n（图 4.1）", fs=8.2)
    arrow(ax, 29.7, y1 + h / 2, 33.4, y1 + h / 2, "$\\mathcal{Q}$", tdy=0.75)
    box(ax, 33.4, y1, 13.5, h,
        "损失计算\n$\\mathcal{L}_{cd}$, $\\mathcal{L}_{unif}$, $\\mathcal{L}_{adv}$", fs=8.2)

    # --- M1 核心: 梯度尺度测量 -> 权重自适应 ---
    arrow(ax, 46.9, y1 + h / 2, 50.4, y1 + h / 2)
    box(ax, 50.4, y1, 17.4, h,
        "梯度尺度测量\n$g_{cd}=\\|\\nabla_\\theta\\mathcal{L}_{cd}\\|_2$\n"
        "$g_{adv}=\\|\\nabla_\\theta\\mathcal{L}_{adv}\\|_2$",
        fill=FILL_M1, fs=8.0, lw=1.7)
    # rho 标签放到箭头上方, 不压框
    arrow(ax, 67.8, y1 + h / 2, 72.6, y1 + h / 2)
    ax.text(70.2, y1 + h + 0.9, "$\\rho=g_{adv}/g_{cd}$",
            ha="center", va="bottom", fontsize=7.8)
    box(ax, 72.6, y1, 17.6, h,
        "对抗权重自适应\n$w_{adv}\\!\\leftarrow\\!\\eta\\cdot g_{cd}/g_{adv}$\n"
        "（M1，$\\eta\\!=\\!0.1$）", fill=FILL_M1, fs=8.0, lw=1.7)

    ax.text(81.4, y1 - 3.0,
            "问题依据：实测初始 $\\rho=1/82.65$，对抗梯度被 CD 梯度淹没",
            ha="center", fontsize=7.9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888888", lw=0.7))

    # --- 第二行: 反传与监控 ---
    y2 = 27.0
    arrow(ax, 81.4, y1 - 4.6, 81.4, y2 + h)
    box(ax, 72.6, y2, 17.6, h,
        "加权反传\n$\\nabla_\\theta(\\mathcal{L}_{cd}\\!+\\!w_{unif}\\mathcal{L}_{unif}"
        "\\!+\\!w_{adv}\\mathcal{L}_{adv})$", fs=7.9)
    ax.text(81.4, y2 - 2.3,
            "观测量 $s_{cd}$ = cd_bwd_share，B-001 实测均值 0.5446",
            ha="center", fontsize=7.7, style="italic")

    arrow(ax, 72.6, y2 + h / 2, 68.2, y2 + h / 2)
    box(ax, 50.8, y2, 17.4, h,
        "训练期监控\n$monitor\\_cd/hd/nuc$\n"
        "注：非论文主表口径", fs=8.0, ls="--")
    arrow(ax, 50.8, y2 + h / 2, 46.4, y2 + h / 2)
    box(ax, 30.0, y2, 16.4, h,
        "落盘\nhistory.json\nmetrics.json", fill=FILL_IN, fs=8.0, ls="--")

    # 判别器更新: 接成闭环(损失计算 -> D 更新 -> 回到损失计算)
    dy = y2 + 1.2
    box(ax, 8.0, dy, 18.0, 4.6,
        "判别器 $D_\\phi$ 更新\n（真/假样本 + 梯度惩罚）", fs=7.9, ls="--")
    # 损失计算 下行到 D 更新
    ax.add_patch(FancyArrowPatch((37.0, y1), (26.0, dy + 2.3),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, linestyle="--",
                                 color=EDGE, zorder=2,
                                 connectionstyle="arc3,rad=0.18",
                                 shrinkA=0, shrinkB=0))
    ax.text(29.0, dy + 6.2, "$\\mathcal{Q}$ / $\\mathcal{Y}$", fontsize=7.6,
            ha="center", color="#333333")
    # D 更新 回到 损失计算 (提供 L_adv)
    ax.add_patch(FancyArrowPatch((17.0, dy + 4.6), (33.4 + 3.0, y1),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, linestyle="--",
                                 color=EDGE, zorder=1,
                                 connectionstyle="arc3,rad=-0.28",
                                 shrinkA=0, shrinkB=0))
    ax.text(19.6, y1 - 4.4, "提供 $\\mathcal{L}_{adv}$", fontsize=7.6,
            ha="center", color="#333333")

    # --- 第三行: 选点与评测 ---
    # y3 上移收紧中段留白(原 12.0 -> 17.0), 同时 ylim 由 60 收到 54
    y3 = 17.0
    arrow(ax, 38.2, y2, 38.2, y3 + h)
    box(ax, 29.0, y3, 18.6, h,
        "平台区选点\n综合评分 cd .5 / hd .2 / nuc .3\n影子 cd-only 对照",
        fill=FILL_KEY, fs=7.9)
    # 选点产物标签: 用正式术语, 不用 best ckpt 缩写
    arrow(ax, 47.6, y3 + h / 2, 52.4, y3 + h / 2)
    ax.text(50.0, y3 + h + 0.9, "选定权重", ha="center", va="bottom", fontsize=7.6)
    box(ax, 52.4, y3, 18.6, h,
        "官方口径评测\n$eval\\_cd\\_hd\\_official()$\n"
        "（论文主表唯一出口）", fill=FILL_KEY, fs=7.9, lw=1.6)
    arrow(ax, 71.0, y3 + h / 2, 75.4, y3 + h / 2)
    box(ax, 75.4, y3, 17.0, h,
        "显著性判定\n平台区 $\\bar{x}\\pm\\sigma$ / 2SE 门槛\n四档接受准则", fs=7.9)

    ax.text(83.9, y3 - 3.4,
            "门槛：CD 0.66% / HD 2.18% / NUC 0.93%\n"
            "判定：ACCEPT_FULL / ACCEPT_PART / REJECT_TRADE / REJECT_NULL",
            ha="center", fontsize=7.7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888888", lw=0.7))

    box(ax, 2.0, y3 - 0.6, 23.0, h + 1.2,
        "口径隔离规则\n训练期 monitor_* 仅用于选点与动态图；\n"
        "主表数字一律重新用官方协议计算；\n"
        "两套口径不得出现在同一张表内",
        fill="#fafafa", fs=7.8, ls=":")

    fig.savefig(os.path.join(OUTDIR, "F4_2_train_eval_pipeline.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure_id": "F4-2",
        "title": "训练与评价流程及梯度自适应对抗权重的介入点（流程图）",
        "depends_on_experiment": False,
        "note": ("图中引用的实测量(rho=1/82.65, cd_bwd_share=0.5446, "
                 "2SE 门槛)来自已完成实验, 见 EVIDENCE_LEDGER"),
        "m1_name": "梯度自适应对抗权重 (gradient-adaptive adversarial weighting, GAAW)",
        "thresholds_2se": {"cd_pct": 0.66, "hd_pct": 2.18, "nuc_pct": 0.93},
        "verdict_levels": ["ACCEPT_FULL", "ACCEPT_PART", "REJECT_TRADE", "REJECT_NULL"],
        "official_metric_entry": "eval_cd_hd_official()",
    }
    json.dump(meta, open(os.path.join(OUTDIR, "F4_2_train_eval_pipeline.data.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fig_forward()
    fig_pipeline()
    for fn in ("F4_1_network_forward.png", "F4_2_train_eval_pipeline.png"):
        p = os.path.join(OUTDIR, fn)
        print("[OK] %s (%d B)" % (fn, os.path.getsize(p)))
