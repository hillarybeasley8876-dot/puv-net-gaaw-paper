# -*- coding: utf-8 -*-
"""
生成第 4 章两张机制图 (讲清神经网络内部与主创新机制):

  F4_3_scmsa_mechanism.png   SC-MSA 通道偏移多头自注意力的窗口划分与铺满约束
  F4_4_m1_gradient_adaptive.png  M1 梯度自适应对抗权重的量纲分析与介入机制

所有结构参数直接来自源码实测:
  puvnet/models/pu_transformer.py SCMSA.__init__  (w = C'/psi, d = w/2, M = 2*psi-1)
  puvnet/models/pu_gan.py adaptive_adv_weight     (eta = target_ratio = 0.1)
不允许凭印象填数字。

图规范 (docs/STYLE_GUIDE.md §4):
  * 一图一论点, 图题即论点
  * 无渐变/阴影/3D, 黑白可读, 靠线型与灰阶区分
  * 中文字体下禁用 warning/check/cross 等符号字形, 用「注：」
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

FILL_IN = "#f2f2f2"
FILL_OP = "#ffffff"
FILL_KEY = "#d9d9d9"
FILL_M1 = "#bfbfbf"
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
# 图 4-3  SC-MSA 通道偏移窗口机制
# =====================================================================
# 源码实参 (pu_transformer.py:248-260), 以第 4/5 个 encoder 的 C'=256 为例
C_PRIME = 256
PSI = 4
W_SPLIT = C_PRIME // PSI          # 64
D_SHIFT = W_SPLIT // 2            # 32
M_HEADS = 2 * PSI - 1             # 7
LAST_END = (M_HEADS - 1) * D_SHIFT + W_SPLIT   # 必须 == 256
assert LAST_END == C_PRIME, "铺满约束不成立, 图中数字与源码不一致"

# 常规 MSA 对照
MSA_HEADS = PSI                   # 4
MSA_D = W_SPLIT                   # 64


def fig_scmsa():
    fig, ax = plt.subplots(figsize=(13.6, 8.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(50, 63.6,
            "图 4-3  通道偏移多头自注意力（SC-MSA）的窗口划分机制与跨头信息通路",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 61.0,
            "以第 4 层编码器 $C'\\!=\\!256$、$\\psi\\!=\\!4$ 为例；相邻窗口重叠 $w\\!-\\!d\\!=\\!32$ 个通道，"
            "构成常规 MSA 不具备的跨头通路",
            ha="center", fontsize=8, color="#444444")

    # ---------- 左上: Q/K/V 生成 ----------
    y_qkv = 58.0
    box(ax, 1.5, y_qkv, 15.0, 6.4,
        "输入特征 $\\mathbf{F}$\n$(B,N,C')=(B,256,256)$", fill=FILL_IN, fs=8.0)
    arrow(ax, 16.5, y_qkv + 3.2, 21.0, y_qkv + 3.2)
    box(ax, 21.0, y_qkv, 16.0, 6.4,
        "线性投影（无偏置）\n$\\mathbf{Q},\\mathbf{K},\\mathbf{V}"
        "\\!=\\!\\mathbf{F}W_q,\\mathbf{F}W_k,\\mathbf{F}W_v$", fs=8.0)
    ax.text(29.0, y_qkv - 2.0,
            "三个 $C'\\!\\times\\!C'$ 线性层，逐点共享；不做空间降采样",
            ha="center", fontsize=7.6, style="italic", color="#333333")

    # ---------- 通道轴与 7 个滑动窗口 ----------
    ax_x0, ax_x1 = 8.0, 88.0
    bar_y = 46.0
    bar_h = 2.4

    def cx(ch):
        return ax_x0 + (ax_x1 - ax_x0) * ch / C_PRIME

    ax.text(50, 53.4,
            "沿通道维施加步长 $d\\!=\\!w/2$ 的滑动窗口，共 $M\\!=\\!2\\psi\\!-\\!1\\!=\\!7$ 个注意力头",
            ha="center", fontsize=8.6, fontweight="bold")

    # 通道轴底条
    ax.add_patch(Rectangle((ax_x0, bar_y), ax_x1 - ax_x0, bar_h,
                           facecolor="#fafafa", edgecolor=EDGE,
                           linewidth=1.0, zorder=2))
    ax.text((ax_x0 + ax_x1) / 2, bar_y + bar_h / 2,
            "通道维 $C'=256$", ha="center", va="center", fontsize=8.0, zorder=3)
    for ch in (0, 64, 128, 192, 256):
        ax.plot([cx(ch), cx(ch)], [bar_y - 0.5, bar_y], color=EDGE, lw=0.8, zorder=3)
        ax.text(cx(ch), bar_y - 1.5, str(ch), ha="center", va="top", fontsize=7.0)

    # 7 个窗口, 逐行错开画出重叠
    row_h = 1.55
    row_gap = 0.42
    top = bar_y - 4.2
    for m in range(M_HEADS):
        s = m * D_SHIFT
        e = s + W_SPLIT
        yy = top - m * (row_h + row_gap)
        shade = FILL_KEY if m % 2 == 0 else "#ececec"
        ax.add_patch(Rectangle((cx(s), yy), cx(e) - cx(s), row_h,
                               facecolor=shade, edgecolor=EDGE,
                               linewidth=1.0, zorder=2))
        ax.text((cx(s) + cx(e)) / 2, yy + row_h / 2,
                "head %d：[%d, %d)" % (m + 1, s, e),
                ha="center", va="center", fontsize=7.2, zorder=3)
        ax.text(ax_x1 + 1.2, yy + row_h / 2,
                "$\\mathbf{A}_m\\!\\in\\!\\mathbb{R}^{256\\times256}$",
                ha="left", va="center", fontsize=6.9, color="#333333")

    bottom_row = top - (M_HEADS - 1) * (row_h + row_gap)

    # 重叠区标注: head1 与 head2 的重叠 [32,64)
    ax.annotate("", xy=(cx(32), top + row_h + 0.5), xytext=(cx(64), top + row_h + 0.5),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#333333"))
    ax.text(cx(48), top + row_h + 1.0,
            "重叠 $w\\!-\\!d\\!=\\!32$", ha="center", va="bottom", fontsize=7.2)

    # 铺满约束: 独占一行, 不与任何框重叠
    ax.text(50, bottom_row - 3.4,
            "铺满约束：$(M\\!-\\!1)d + w = 6\\times32 + 64 = 256 = C'$"
            "（源码 SCMSA.__init__ 断言，不成立则拒绝构图）",
            ha="center", va="center", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888888", lw=0.7))

    # ---------- 单头内部计算 ----------
    y_att = 12.4
    hh = 6.2

    # 窗口区 -> 单头计算 的连接: 取任一 head 的切片进入下方流程
    ax.text(50, bottom_row - 6.6,
            "下行：任取第 $m$ 个窗口，展示单头内部计算",
            ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.add_patch(FancyArrowPatch((50, bottom_row - 7.6), (10.75, y_att + hh),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, linestyle="--",
                                 color=EDGE, zorder=1,
                                 connectionstyle="arc3,rad=0.12",
                                 shrinkA=0, shrinkB=0))
    box(ax, 1.5, y_att, 17.0, hh,
        "窗口切片\n$\\mathbf{Q}_m,\\mathbf{K}_m,\\mathbf{V}_m"
        "\\!\\in\\!\\mathbb{R}^{N\\times w}$", fill=FILL_IN, fs=7.9)
    arrow(ax, 18.5, y_att + hh / 2, 22.2, y_att + hh / 2)
    box(ax, 22.2, y_att, 19.0, hh,
        "点积注意力\n$\\mathbf{A}_m\\!=\\!\\mathrm{softmax}"
        "(\\mathbf{Q}_m\\mathbf{K}_m^{\\top})$", fill=FILL_KEY, fs=7.9)
    arrow(ax, 41.2, y_att + hh / 2, 44.9, y_att + hh / 2)
    box(ax, 44.9, y_att, 17.5, hh,
        "加权聚合\n$\\mathbf{H}_m\\!=\\!\\mathbf{A}_m\\mathbf{V}_m$", fs=7.9)
    arrow(ax, 62.4, y_att + hh / 2, 66.1, y_att + hh / 2)
    box(ax, 66.1, y_att, 15.6, hh,
        "拼接\n$[\\mathbf{H}_1\\cdots\\mathbf{H}_7]$\n$N\\times(M\\!\\cdot\\!w)=N\\times448$",
        fill=FILL_IN, fs=7.6)
    arrow(ax, 81.7, y_att + hh / 2, 85.0, y_att + hh / 2)
    box(ax, 85.0, y_att, 13.5, hh,
        "输出投影\n$448\\!\\rightarrow\\!256$", fs=7.9)

    ax.text(50, y_att - 3.2,
            "注：原文算法未对 $\\mathbf{Q}_m\\mathbf{K}_m^{\\top}$ 做 $1/\\sqrt{w}$ 缩放，"
            "本文默认关闭以忠于原文，该项作为独立消融开关记录",
            ha="center", va="center", fontsize=7.6, style="italic", color="#333333")

    # ---------- 右上: 与常规 MSA 的对照 ----------
    cmp_x, cmp_y = 62.0, 56.6
    box(ax, cmp_x, cmp_y, 36.5, 8.0,
        "对照：常规 MSA（$d\\!=\\!w$，$M\\!=\\!\\psi\\!=\\!4$）\n"
        "窗口互不重叠，头间无共享通道；\n"
        "跨头信息只能经输出投影间接混合",
        fill=FILL_OP, ls="--", fs=7.8)

    fig.savefig(os.path.join(OUTDIR, "F4_3_scmsa_mechanism.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure_id": "F4-3",
        "title": "通道偏移多头自注意力（SC-MSA）的窗口划分机制与跨头信息通路",
        "depends_on_experiment": False,
        "source_of_truth": "puvnet/models/pu_transformer.py SCMSA.__init__ (L248-269)",
        "params": {
            "C_prime": C_PRIME, "psi": PSI, "w": W_SPLIT,
            "d": D_SHIFT, "M_heads": M_HEADS,
            "overlap": W_SPLIT - D_SHIFT,
            "tiling_check": "(M-1)*d + w = %d == C' = %d" % (LAST_END, C_PRIME),
            "concat_dim": M_HEADS * W_SPLIT,
        },
        "msa_baseline": {"d": MSA_D, "M_heads": MSA_HEADS, "overlap": 0},
        "note": "scale_qk 默认 False, 忠于原论文算法, 作为消融开关",
    }
    json.dump(meta, open(os.path.join(OUTDIR, "F4_3_scmsa_mechanism.data.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# =====================================================================
# 图 4-4  M1 梯度自适应对抗权重
# =====================================================================
ETA = 0.1                 # pu_gan.py adaptive_adv_weight target_ratio 默认
RHO_MEASURED = 1 / 82.65  # E-000 实测初始梯度比
W_AUTO_MEASURED = 8.27    # 实测自适应权重
S_CD_MEAN = 0.5446        # B-001 平台区 cd_bwd_share 均值


def fig_m1():
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    ax.text(50, 53.6,
            "图 4-4  梯度自适应对抗权重（GAAW）的量纲失配诊断与闭环调节机制",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 51.0,
            "固定权重方案在 CD 损失下降三个数量级的过程中必然失衡；"
            "GAAW 以梯度范数比为观测量，逐步重标定 $w_{adv}$",
            ha="center", fontsize=8, color="#444444")

    # ---------- 上半: 问题诊断 ----------
    ax.text(24.0, 46.8, "（a）问题：梯度尺度失配", ha="center",
            fontsize=9.4, fontweight="bold")

    y_p = 37.6
    hp = 7.4
    box(ax, 1.5, y_p, 20.0, hp,
        "重建项梯度\n$g_{cd}\\!=\\!\\|\\nabla_\\theta\\mathcal{L}_{cd}\\|_2$",
        fill=FILL_IN, fs=8.0)
    box(ax, 25.5, y_p, 20.0, hp,
        "对抗项梯度\n$g_{adv}\\!=\\!\\|\\nabla_\\theta\\mathcal{L}_{adv}\\|_2$",
        fill=FILL_IN, fs=8.0)
    # 两者之间的比值箭头
    ax.annotate("", xy=(24.6, y_p + hp / 2), xytext=(21.9, y_p + hp / 2),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=EDGE))
    ax.text(23.3, y_p + hp + 1.0,
            "$\\rho\\!=\\!g_{adv}/g_{cd}$", ha="center", va="bottom", fontsize=8.0)

    ax.text(23.5, y_p - 3.2,
            "实测（E-000，固定权重设定）：$\\rho\\!=\\!1/82.65$\n"
            "对抗梯度仅为重建梯度的 1.21%，判别器信号被淹没",
            ha="center", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888888", lw=0.7))

    # ---------- 右上: 为何固定权重必失衡 ----------
    ax.text(74.0, 46.8, "（b）固定权重为何必然失衡", ha="center",
            fontsize=9.4, fontweight="bold")
    box(ax, 52.0, y_p - 1.2, 46.5, hp + 2.4,
        "训练过程中 $\\mathcal{L}_{cd}$ 由 $10^{-3}$ 量级降至 $10^{-6}$ 量级，"
        "$g_{cd}$ 随之衰减；\n"
        "若 $w_{adv}$ 取常数，则 $w_{adv}g_{adv}/g_{cd}$ 单调上升，\n"
        "早期对抗欠驱动、后期对抗过驱动，二者不可能同时满足",
        fill=FILL_OP, ls="--", fs=7.9)

    # ---------- 下半: 解法闭环 ----------
    ax.text(50, 28.0, "（c）解法：以目标比 $\\eta$ 为约束的逐步重标定",
            ha="center", fontsize=9.4, fontweight="bold")

    y_s = 17.6
    hs = 7.6
    box(ax, 1.5, y_s, 18.5, hs,
        "反向传播两次\n（$\\mathcal{L}_{cd}$、$\\mathcal{L}_{adv}$\n"
        "对共享参数 $\\theta$）", fill=FILL_OP, fs=7.9)
    arrow(ax, 20.0, y_s + hs / 2, 23.6, y_s + hs / 2)
    box(ax, 23.6, y_s, 20.0, hs,
        "测量梯度范数\n$g_{cd}$、$g_{adv}$\n（$\\theta$ 为生成器共享权重）",
        fill=FILL_M1, fs=7.9, lw=1.7)
    arrow(ax, 43.6, y_s + hs / 2, 47.2, y_s + hs / 2)
    box(ax, 47.2, y_s, 22.5, hs,
        "重标定权重\n$w_{adv}\\!\\leftarrow\\!\\eta\\cdot g_{cd}/g_{adv}$\n"
        "$\\eta\\!=\\!0.1$（目标比）", fill=FILL_M1, fs=7.9, lw=1.7)
    arrow(ax, 69.7, y_s + hs / 2, 73.3, y_s + hs / 2)
    box(ax, 73.3, y_s, 25.2, hs,
        "加权反传\n$\\nabla_\\theta(\\mathcal{L}_{cd}"
        "\\!+\\!w_{unif}\\mathcal{L}_{unif}"
        "\\!+\\!w_{adv}\\mathcal{L}_{adv})$", fs=7.9)

    # 闭环回连: 加权反传 -> 反向传播两次(下一步)
    ax.add_patch(FancyArrowPatch((85.9, y_s), (10.75, y_s),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, linestyle="--",
                                 color=EDGE, zorder=1,
                                 connectionstyle="arc3,rad=0.14",
                                 shrinkA=0, shrinkB=0))
    ax.text(48.0, y_s - 6.0, "每步重新测量（$w_{adv}$ 以标量 detach 形式参与，不产生二阶梯度）",
            ha="center", fontsize=7.6, style="italic", color="#333333")

    # 实测量标注
    ax.text(58.5, y_s + hs + 1.4,
            "实测 $w_{adv}\\!=\\!8.27$",
            ha="center", va="bottom", fontsize=7.6)
    ax.text(85.9, y_s + hs + 1.4,
            "观测量 $s_{cd}\\!=\\!0.5446$（B-001 平台区）",
            ha="center", va="bottom", fontsize=7.6)

    ax.text(50, 4.2,
            "注：本机制只调节量级、不改变对抗损失形式，因此与判别器结构、"
            "梯度惩罚系数等设定相互独立，可单独消融（B2 组）",
            ha="center", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fafafa", ec="#888888", lw=0.7))

    fig.savefig(os.path.join(OUTDIR, "F4_4_m1_gradient_adaptive.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure_id": "F4-4",
        "title": "梯度自适应对抗权重（GAAW）的量纲失配诊断与闭环调节机制",
        "depends_on_experiment": True,
        "source_of_truth": "puvnet/models/pu_gan.py adaptive_adv_weight (L294-320)",
        "measured": {
            "eta_target_ratio": ETA,
            "rho_initial": "1/82.65",
            "rho_initial_pct": round(RHO_MEASURED * 100, 2),
            "w_adv_measured": W_AUTO_MEASURED,
            "s_cd_plateau_mean_b001": S_CD_MEAN,
        },
        "ablation_group": "B2",
        "note": ("warmup(adv_warmup_factor) 属 M3, 与 M1 为独立改进点, "
                 "本图不混合表述"),
    }
    json.dump(meta, open(os.path.join(OUTDIR, "F4_4_m1_gradient_adaptive.data.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fig_scmsa()
    fig_m1()
    for fn in ("F4_3_scmsa_mechanism.png", "F4_4_m1_gradient_adaptive.png"):
        p = os.path.join(OUTDIR, fn)
        print("[OK] %s (%d B)" % (fn, os.path.getsize(p)))
