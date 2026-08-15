# -*- coding: utf-8 -*-
"""
生成不依赖实验结果的论文示意图 / 结构图 / 路线图。
输出: docs/figures_schematic/*.png  + 同名 .data.json (可追溯)

红线:
  1. 每张图必须在图题中含「示意图」/「框图」等字样, 不得冒充实验结果。
  2. 图中出现的年份必须是真实首次发表年份, 禁止为排版方便篡改。
  3. 布局重叠由 scripts/selfcheck_schematic.py 量化校验, 不靠肉眼。
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


def save(fig, name, meta):
    p = os.path.join(OUTDIR, name + ".png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta["generated_at"] = TS
    meta["script"] = "scripts/make_schematic_figures.py"
    meta["depends_on_experiment"] = False
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


# ============================================================
# 图 1-2 技术路线图（六章大纲版；替代旧七章 M1/H1-H5 版本）
# ============================================================
def fig_1_2_roadmap():
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    stages = [
        (0.3, 5.6, "阶段一  问题界定\n· 任务定义、评价维度与研究边界\n· 方法谱系与能力边界梳理"),
        (0.3, 4.1, "阶段二  基线与诊断\n· 复现基线、核验评价协议\n· 实测瓶颈，产出命题约束"),
        (0.3, 2.6, "阶段三  命题与机制\n· 瓶颈转为可证伪命题与判据\n· 调节机制的形式化与复杂度"),
        (0.3, 1.1, "阶段四  验证与界定\n· 受控对照逐条裁定命题\n· 失效边界与不可外推范围"),
    ]
    for x, y, t in stages:
        box(ax, x, y, 3.6, 1.2, t, fs=9)
    for i in range(3):
        arrow(ax, (2.1, stages[i][1]), (2.1, stages[i + 1][1] + 1.2))

    outs = [
        (4.6, 5.6, "输出\n问题定位与\n方法谱系依据\n（第 1、2 章）"),
        (4.6, 4.1, "输出\n可复现基线与\n瓶颈证据\n（第 3 章）"),
        (4.6, 2.6, "输出\n预注册判据与\n机制形式化\n（第 4、5 章）"),
        (4.6, 1.1, "输出\n实测裁定与\n能力边界\n（第 6、7 章）"),
    ]
    for x, y, t in outs:
        box(ax, x, y, 2.4, 1.2, t, fc=C_GRAY, ec="#888", fs=8.6)
    for _, y, _ in stages:
        arrow(ax, (3.9, y + 0.6), (4.6, y + 0.6), color="#999", lw=1.2)

    box(ax, 7.6, 2.5, 2.1, 3.3,
        "论证约束\n\n方法设计\n须追溯到\n前一阶段的\n实测证据\n\n结论强度\n不超过\n后一阶段的\n对照实验",
        fc=C_HL, ec=C_HLE, fs=9.0, lw=1.8, bold=True)
    for _, y, _ in stages:
        arrow(ax, (7.0, y + 0.6), (7.6, 4.15), color="#C9A227", lw=1.0, ls="--", rad=0.12)

    ax.text(5.0, 6.92, "图 1.2  全文技术路线图", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.55,
            "注：实线箭头为阶段推进顺序，细线箭头为各阶段输出；虚线表示论证约束对各阶段的作用关系。"
            "任一环节缺失时，相应主张不进入结论。",
            ha="center", fontsize=8.3, color="#555")
    save(fig, "F1_2_technical_roadmap",
         {"figure_id": "1.2", "caption": "全文技术路线图", "type": "schematic",
          "chapter": "1.4.4", "n_stages": 4,
          "revision": "2026-08-15 对齐七章结构：阶段三由「结构模块」改为「命题与机制」，"
                      "输出章号改为第 4、5 章与第 6、7 章；"
                      "「核心纪律」改写为「论证约束」（去项目管理口吻）；"
                      "图题编号改为点号（FORMAT_TONGJI §2.2）"})


# ============================================================
# 图 2-1 点云上采样方法发展脉络
# ============================================================
def fig_2_1_timeline():
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.set_xlim(2001.5, 2026.5); ax.set_ylim(0, 12.5); ax.axis("off")

    AXIS_Y = 7.0
    ax.annotate("", xy=(2026.1, AXIS_Y), xytext=(2001.9, AXIS_Y),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.0))
    for yr in range(2003, 2026, 2):
        ax.plot([yr, yr], [AXIS_Y - 0.14, AXIS_Y + 0.14], color="#444", lw=1.2)
        ax.text(yr, AXIS_Y + 0.32, str(yr), ha="center", fontsize=8.5, color="#444")

    # 上半: 传统方法(稀疏, 单行足够)
    upper = [(2003, "MLS\n点集曲面"), (2007, "LOP"), (2009, "WLOP"),
             (2013, "EAR\n边缘感知"), (2015, "Deep Points\nConsolidation")]
    BW_U, BH_U = 1.75, 1.35
    for x, t in upper:
        ax.plot([x, x], [AXIS_Y + 0.14, 8.9], color="#7A9CC6", lw=1.2, zorder=0)
        box(ax, x - BW_U / 2, 8.95, BW_U, BH_U, t, fc="#EAF2FB", ec="#7A9CC6", fs=8.2)

    # 下半: 深度方法密集(2018-2024 共 9 项) -> 四层交错 + 竖直引导线
    # 年份为真实首次发表年份, 禁止为排版篡改; 重叠只能靠层分配 + 框宽解决。
    # 层分配约束: 同层相邻年份差 >= BW_L, 由 selfcheck_schematic.py 量化校验。
    lower = [
        (2018.0, "PU-Net\n首个深度方法", 0),
        (2018.8, "EC-Net\n边缘感知深度化", 1),
        (2019.2, "MPU\n渐进式上采样", 2),
        (2019.7, "PU-GAN\n引入对抗训练", 3),
        (2021.0, "PU-GCN\n图卷积聚合", 0),
        (2021.6, "Dis-PU\n解耦精化", 1),
        (2022.2, "PU-Transformer\n引入注意力", 2),
        (2023.2, "Grad-PU\n任意倍率", 3),
        (2024.0, "PUDM\n扩散式上采样", 0),
        (2024.6, "RepKPU\n核点表示与形变", 1),
    ]
    LAYER_TOP = [6.15, 4.85, 3.55, 2.25]
    BW_L, BH_L = 1.95, 1.12
    for x, t, lyr in lower:
        top = LAYER_TOP[lyr]
        y0 = top - BH_L
        hl = ("PU-GAN" in t) or ("PU-Transformer" in t)
        ax.plot([x, x], [AXIS_Y - 0.14, top], color="#C6867A", lw=1.0,
                ls=":" if lyr > 0 else "-", zorder=0)
        ax.plot(x, AXIS_Y - 0.14, "o", ms=3.5, color="#C6867A", zorder=1)
        box(ax, x - BW_L / 2, y0, BW_L, BH_L, t,
            fc=C_HL if hl else "#FBEEEA", ec=C_HLE if hl else "#C6867A",
            fs=7.2 if hl else 7.8, lw=1.9 if hl else 1.2, bold=hl)

    ax.text(2002.2, 10.75, "传统几何优化方法（先验驱动）", fontsize=11.5,
            fontweight="bold", color="#3B6FB6")
    ax.text(2002.2, 1.25, "深度学习方法（数据驱动，2018 年后密集涌现）",
            fontsize=11.5, fontweight="bold", color="#B4573F")

    ax.axvspan(2019.4, 2022.6, ymin=0.0, ymax=1.0, color=C_HL, alpha=0.35, zorder=-1)
    ax.text(2011.5, 11.55, "本文切入区间（图中高亮带）：以注意力主干的结构改进为主线，对抗训练作为可选辅助",
            ha="center", va="center", fontsize=9.2, fontweight="bold",
            bbox=dict(fc=C_HL, ec=C_HLE, lw=1.6, boxstyle="round,pad=0.35"))

    ax.text(2014.0, 12.28, "图 2-1  点云上采样方法发展脉络示意图",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(2014.0, 0.4,
            "注：下半部分采用四层交错排布以避免 2018—2024 年密集条目重叠；竖直引导线标示各方法的真实首次发表年份。",
            ha="center", fontsize=8.4, color="#555")
    save(fig, "F2_1_method_timeline",
         {"figure_id": "2-1", "caption": "点云上采样方法发展脉络示意图", "type": "schematic",
          "chapter": "2.1", "n_upper": len(upper), "n_lower": len(lower),
          "layout_fix": "下半密集区改四层交错排布，修复单行重叠缺陷；年份未做任何调整",
          "n_layers_lower": 4})


# ============================================================
# 图 2-2 方法分类树
# ============================================================
def fig_2_2_taxonomy():
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 7); ax.axis("off")

    box(ax, 5.0, 6.0, 2.5, 0.7, "点云上采样方法", fc=C_HL, ec=C_HLE, fs=11, bold=True, lw=1.8)

    l1 = [(0.5, 4.6, 2.4, "传统几何优化\n（先验驱动）"),
          (3.3, 4.6, 2.4, "监督深度学习\n（固定倍率）"),
          (6.1, 4.6, 2.4, "隐式/任意倍率\n（连续表示）"),
          (8.9, 4.6, 2.4, "生成式方法\n（对抗/扩散）")]
    for x, y, w, t in l1:
        box(ax, x, y, w, 0.85, t, fs=9.2, bold=True)
        arrow(ax, (6.25, 6.0), (x + w / 2, y + 0.85), color="#888", lw=1.2)

    leaves = {
        0: ["MLS (2003)", "LOP (2007)", "WLOP (2009)", "EAR (2013)", "DPC (2015)"],
        1: ["PU-Net (2018)", "EC-Net (2018)", "MPU (2019)", "PU-GCN (2021)", "Dis-PU (2021)",
            "PU-Transformer (2022)", "PU-CRN (2022)"],
        2: ["Meta-PU (2021)", "Neural Points (2022)", "SAPCU (2022)", "Grad-PU (2023)",
            "iPUNet (2023)"],
        3: ["l-GAN (2018)", "TreeGAN (2019)", "PU-GAN (2019)", "PUDM (2024)"],
    }
    for i, (x, y, w, _) in enumerate(l1):
        items = leaves[i]
        for j, it in enumerate(items):
            yy = 3.85 - j * 0.5
            hl = ("PU-Transformer" in it) or ("PU-GAN (2019)" in it)
            box(ax, x + 0.1, yy, w - 0.2, 0.4, it,
                fc=C_HL if hl else C_GRAY, ec=C_HLE if hl else "#AAA",
                fs=8.2, lw=1.6 if hl else 1.0, bold=hl)
        ax.plot([x + w / 2, x + w / 2], [y, 4.25], color="#BBB", lw=1.0, zorder=0)

    ax.text(6.25, 6.85, "图 2-2  点云上采样方法分类框图",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6.25, 0.35, "注：橙色高亮为本文直接构建于其上的两项工作（PU-GAN 的对抗框架与 PU-Transformer 的注意力主干）。",
            ha="center", fontsize=8.5, color="#555")
    save(fig, "F2_2_method_taxonomy",
         {"figure_id": "2-2", "caption": "点云上采样方法分类框图", "type": "schematic",
          "chapter": "2.1.4", "n_categories": 4,
          "n_leaves": sum(len(v) for v in leaves.values())})


# ============================================================
# 图 3-2 上采样质量四维分解与冲突
# ============================================================
def fig_3_2_quality_dims():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 3.4, 4.4, 3.2, 1.2, "上采样质量", fc=C_HL, ec=C_HLE, fs=11, bold=True, lw=1.8)
    dims = [(0.4, 7.6, "分布保真性\n(Chamfer 距离)"),
            (6.2, 7.6, "空间均匀性\n(NUC / nn-cv)"),
            (0.4, 1.4, "表面贴合性\n(P2F 距离)"),
            (6.2, 1.4, "细节锐度\n(Hausdorff 距离)")]
    for x, y, t in dims:
        box(ax, x, y, 3.4, 1.2, t, fs=9.2)
        arrow(ax, (x + 1.7, y + (0 if y > 5 else 1.2)), (5.0, 5.6 if y > 5 else 4.4),
              color="#888", lw=1.3)
    for (x1, y1), (x2, y2), lab in [((2.1, 7.6), (2.1, 2.6), "①"),
                                    ((7.9, 7.6), (7.9, 2.6), "②"),
                                    ((3.8, 8.2), (6.2, 8.2), "③")]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<|-|>", color=C_HLE, lw=1.6, ls="--"))
        ax.text((x1 + x2) / 2 + 0.25, (y1 + y2) / 2, lab + " 冲突",
                fontsize=8.8, color=C_HLE, fontweight="bold",
                rotation=90 if abs(x1 - x2) < 0.5 else 0,
                ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_title("(a) 质量维度分解与冲突关系", fontsize=11, pad=8)

    ax = axes[1]
    t = np.linspace(0, 1, 200)
    fid = 1.0 - 0.55 * t ** 1.4
    uni = 0.35 + 0.60 * t ** 0.75
    ax.plot(t, fid, lw=2.4, color="#3B6FB6", label="分布保真性（越高越好）")
    ax.plot(t, uni, lw=2.4, color="#E07B39", label="空间均匀性（越高越好）")
    idx = int(np.argmin(np.abs(fid - uni)))
    ax.plot(t[idx], fid[idx], "o", ms=10, color="#C62828", zorder=5)
    ax.annotate("权衡拐点\n（本文关注区间）", xy=(t[idx], fid[idx]),
                xytext=(t[idx] + 0.13, fid[idx] + 0.22), fontsize=9.2,
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
                bbox=dict(fc=C_HL, ec=C_HLE, lw=1.2, boxstyle="round,pad=0.3"))
    ax.axvspan(t[idx] - 0.12, t[idx] + 0.12, color=C_HL, alpha=0.55, zorder=0)
    ax.set_xlabel("均匀性约束强度（对抗 / 均匀性损失权重）", fontsize=10)
    ax.set_ylabel("相对质量（归一化示意值）", fontsize=10)
    ax.set_ylim(0.2, 1.12); ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="lower center")
    ax.grid(alpha=0.3, ls=":")
    ax.set_title("(b) 保真性—均匀性权衡示意（非实验数据）", fontsize=11, pad=8)

    fig.suptitle("图 3-2  点云上采样质量的四维分解与内在冲突示意图",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "注：子图 (b) 为定性趋势示意，纵轴为归一化相对值，不代表任何实测数据。",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    save(fig, "F3_2_quality_dimensions",
         {"figure_id": "3-2", "caption": "点云上采样质量的四维分解与内在冲突示意图",
          "type": "schematic", "chapter": "3.1.2/3.1.3", "n_dims": 4,
          "subplot_b_is_illustrative": True})


# ============================================================
# 图 3-4 评价协议流程
# ============================================================
def fig_3_4_eval_protocol():
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    steps = [
        (0.3, "测试网格模型\n(127 个 .off)"),
        (2.35, "泊松采样\n生成稀疏输入\nN = 2048"),
        (4.4, "分块推理\npatch 划分\n+ 归一化"),
        (6.45, "结果融合\n+ 反归一化\n回原坐标系"),
        (8.5, "指标计算\nCD / HD / P2F\n/ NUC"),
        (10.55, "论文口径数字\neval_cd_hd_\nofficial()"),
    ]
    for i, (x, t) in enumerate(steps):
        last = (i == len(steps) - 1)
        box(ax, x, 3.2, 1.75, 1.5, t,
            fc=C_HL if last else C_BOX, ec=C_HLE if last else C_EDGE,
            fs=8.4, bold=last, lw=1.8 if last else 1.4)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.75, 3.95), (steps[i + 1][0], 3.95))

    box(ax, 2.35, 1.2, 3.8, 1.3,
        "口径一致性约束\n· 归一化半径与原实现对齐\n· 距离度量为平方距离均值",
        fc=C_GRAY, ec="#888", fs=8.4)
    arrow(ax, (4.25, 2.5), (4.25, 3.2), color="#888", ls="--", lw=1.2)
    arrow(ax, (5.3, 2.5), (7.3, 3.2), color="#888", ls="--", lw=1.2, rad=-0.15)

    box(ax, 6.9, 1.2, 4.4, 1.3,
        "权威来源锚定\n协议实现以 PU-GCN 官方源码为准\n（已存 sha256 校验与出处说明）",
        fc="#EAF7EE", ec="#4C9A5E", fs=8.4)
    arrow(ax, (9.1, 2.5), (9.375, 3.2), color="#4C9A5E", ls="--", lw=1.2)

    box(ax, 0.3, 5.15, 5.2, 0.75,
        "训练期 monitor_* 指标 —— 仅用于选点与曲线，禁止进入论文主表",
        fc="#FDECEA", ec="#C62828", fs=8.6, bold=True)
    arrow(ax, (2.9, 5.15), (9.375, 4.7), color="#C62828", ls=":", lw=1.3, rad=0.1)

    ax.text(6.0, 6.0, "图 3-4  上采样质量评价协议流程框图",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6.0, 0.55,
            "注：红色框标明本文严格执行的口径隔离规则，训练期监控指标与论文报告指标来自不同计算通道。",
            ha="center", fontsize=8.5, color="#555")
    save(fig, "F3_4_eval_protocol",
         {"figure_id": "3-4", "caption": "上采样质量评价协议流程框图", "type": "schematic",
          "chapter": "3.3.4", "n_steps": len(steps),
          "notes": "口径隔离规则与 PU-GCN 官方协议锚定"})


# ============================================================
# 图 3-5 瓶颈归因图
# ============================================================
def fig_3_5_bottleneck():
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.6); ax.axis("off")

    box(ax, 4.0, 5.3, 3.5, 0.85, "均匀性—保真性权衡困境", fc="#FDECEA", ec="#C62828",
        fs=11, bold=True, lw=1.8)

    mid = [(0.4, 3.4, "成因一\nChamfer 距离的\n点对点最近邻匹配\n对聚簇不敏感"),
           (4.0, 3.4, "成因二\n对抗损失梯度尺度\n远小于重建损失\n（固定权重下被压制）"),
           (7.6, 3.4, "成因三\n均匀性约束缺乏\n显式监督信号\n或权重量级失配")]
    for x, y, t in mid:
        hl = "成因二" in t
        box(ax, x, y, 3.5, 1.4, t, fc=C_HL if hl else C_BOX,
            ec=C_HLE if hl else C_EDGE, fs=8.8, lw=1.8 if hl else 1.4, bold=hl)
        arrow(ax, (x + 1.75, y + 1.4), (5.75, 5.3), color="#888", lw=1.2)

    low = [(0.4, 1.5, "对策方向 A\n双向 CD 权重再分配\n(本文 A1/A2 组)"),
           (4.0, 1.5, "对策方向 B\n梯度自适应对抗权重\n(本文 M1，主创新)"),
           (7.6, 1.5, "对策方向 C\n均匀性损失权重标定\n(本文 C1 组)")]
    for i, (x, y, t) in enumerate(low):
        hl = (i == 1)
        box(ax, x, y, 3.5, 1.25, t, fc=C_HL if hl else C_GRAY,
            ec=C_HLE if hl else "#888", fs=8.8, lw=1.8 if hl else 1.2, bold=hl)
        arrow(ax, (mid[i][0] + 1.75, mid[i][1]), (x + 1.75, y + 1.25),
              color=C_HLE if hl else "#999", lw=1.6 if hl else 1.2)

    ax.text(5.75, 6.35, "图 3-5  均匀性—保真性权衡的成因归因与对策框图",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.75, 0.75,
            "注：成因二的梯度尺度失衡由本文实测确认，是主创新机制 M1 的直接依据；具体测量值见第 4.3.1 节。",
            ha="center", fontsize=8.5, color="#555")
    save(fig, "F3_5_bottleneck_attribution",
         {"figure_id": "3-5", "caption": "均匀性—保真性权衡的成因归因与对策框图",
          "type": "schematic", "chapter": "3.2.2", "n_causes": 3, "n_remedies": 3})


# ============================================================
# 图 1-1 点云稀疏性及上采样任务示意图
#   说明: 点位由解析曲面(环面片)采样得到, 属几何示意, 非任何模型的实验输出。
# ============================================================
def fig_1_1_task_illustration():
    rng = np.random.default_rng(20260811)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))

    # 解析曲面: 环面片投影到 2D 观察面, 用于示意"沿表面采样"
    def sample_surface(n):
        u = rng.uniform(0.0, 2.0 * np.pi, n)
        v = rng.uniform(0.0, 2.0 * np.pi, n)
        R, r = 1.0, 0.42
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        # 固定视角正交投影
        return x * 0.94 + z * 0.34, y * 0.90 + z * 0.42

    xs_d, ys_d = sample_surface(1024)
    idx = rng.choice(1024, 256, replace=False)

    panels = [
        ("(a) 稀疏输入 $\\mathcal{P}$，$N=256$", xs_d[idx], ys_d[idx], 9.0, "#C6867A"),
        ("(b) 上采样输出 $\\mathcal{Q}$，$rN=1024$", xs_d, ys_d, 4.2, "#3B6FB6"),
        ("(c) 真值稠密点集 $\\mathcal{Y}$，$rN=1024$", *sample_surface(1024), 4.2, "#4C8C57"),
    ]
    for ax, p in zip(axes, panels):
        title, px, py, ms, c = p
        ax.scatter(px, py, s=ms, c=c, linewidths=0, alpha=0.85)
        ax.set_title(title, fontsize=10.5, pad=9)
        ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal"); ax.axis("off")

    fig.text(0.352, 0.50, "$G_\\theta$", fontsize=15, ha="center", va="center",
             bbox=dict(fc=C_HL, ec=C_HLE, lw=1.5, boxstyle="round,pad=0.32"))
    fig.text(0.352, 0.40, "上采样映射", fontsize=8.6, ha="center", color="#555")
    fig.text(0.655, 0.50, "监督比较", fontsize=9.0, ha="center", va="center",
             bbox=dict(fc=C_GRAY, ec="#888", lw=1.2, boxstyle="round,pad=0.28"))
    fig.text(0.655, 0.40, "集合距离", fontsize=8.6, ha="center", color="#555")

    fig.suptitle("图 1-1  点云稀疏性及上采样任务示意图", fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.035,
             "注：三组点位均由同一解析曲面（环面）随机采样得到，仅用于示意稀疏输入、上采样输出与真值稠密点集的关系，"
             "不代表任何模型的实验输出。$\\mathcal{Q}$ 与 $\\mathcal{Y}$ 采样实例不同，体现任务的采样实例不唯一性。",
             ha="center", fontsize=8.4, color="#555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.13, wspace=0.32)
    save(fig, "F1_1_task_illustration",
         {"figure_id": "1-1", "caption": "点云稀疏性及上采样任务示意图", "type": "schematic",
          "chapter": "1.2", "N_input": 256, "N_output": 1024, "r": 4,
          "point_source": "解析环面曲面随机采样（rng seed=20260811），非实验输出"})


# ============================================================
# 图 1.3 研究内容与章节关系图
# ============================================================
def fig_1_3_chapter_map():
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")

    chaps = [
        (0.35, 5.70, 3.1, 1.10, "第 1 章  绪论\n问题定位与研究边界", C_BOX, C_EDGE),
        (0.35, 4.20, 3.1, 1.10, "第 2 章  文献综述\n方法谱系与研究缺口", C_BOX, C_EDGE),
        (0.35, 2.70, 3.1, 1.10, "第 3 章  基线复现与瓶颈诊断\n协议核验与实测瓶颈", C_BOX, C_EDGE),
        (4.45, 5.00, 3.1, 1.10, "第 4 章  模型假设与研究设计\n可证伪命题与预注册判据", C_HL, C_HLE),
        (4.45, 3.20, 3.1, 1.10, "第 5 章  梯度自适应对抗权重机制\n形式化、算法与复杂度", C_HL, C_HLE),
        (8.55, 4.10, 3.1, 1.10, "第 6 章  实验结果与边界分析\n逐条裁定、失效边界", C_BOX, C_EDGE),
        (8.55, 1.90, 3.1, 1.10, "第 7 章  结论与展望\n获支持的结论与局限", C_GRAY, "#888"),
    ]
    for x, y, w, h, t, fc, ec in chaps:
        box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=9.0, lw=1.8 if fc == C_HL else 1.4,
            bold=(fc == C_HL))

    # 主链：1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
    arrow(ax, (1.90, 5.70), (1.90, 5.30), lw=1.6)
    arrow(ax, (1.90, 4.20), (1.90, 3.80), lw=1.6)
    arrow(ax, (3.45, 3.25), (4.45, 5.20), lw=1.6, rad=0.10)
    arrow(ax, (6.00, 5.00), (6.00, 4.30), lw=1.6)
    arrow(ax, (7.55, 3.75), (8.55, 4.45), lw=1.6)
    arrow(ax, (10.10, 4.10), (10.10, 3.00), lw=1.6)

    # 支撑与约束关系
    arrow(ax, (3.45, 6.25), (8.55, 4.85), color="#999", lw=1.1, ls="--", rad=-0.18)
    arrow(ax, (7.55, 5.55), (8.55, 4.90), color="#C9A227", lw=1.3, ls="--")

    ax.text(6.0, 7.10, "图 1.3  研究内容与章节关系图", ha="center", fontsize=12.5,
            fontweight="bold")
    ax.text(0.35, 1.75,
            "实线：论证的推进顺序\n虚线（灰）：理论依据的支撑关系\n虚线（黄）：判据对裁定的约束关系",
            fontsize=9.0, color="#555", va="top", linespacing=1.9)
    ax.text(6.0, 0.42,
            "注：高亮框为本文的方法主体。第 4 章在观察结果之前固定判据，第 5 章给出被检验的机制本体，"
            "第 6 章按该判据原样裁定，三者共同构成不可事后调整的论证链条。",
            ha="center", fontsize=8.4, color="#555")
    save(fig, "F1_3_chapter_map",
         {"figure_id": "1.3", "caption": "研究内容与章节关系图", "type": "schematic",
          "chapter": "1.6", "n_chapters": 7,
          "revision": "2026-08-15 由六章改为七章结构：拆出第 4 章（命题与判据）与"
                      "第 5 章（机制形式化），实验章顺延为第 6 章、结论章为第 7 章；"
                      "图题编号改为点号（FORMAT_TONGJI §2.2）"})


# ============================================================
# 图 2-3 点云特征学习骨干演进图
# ============================================================
def fig_2_3_backbone_evolution():
    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.6); ax.axis("off")

    gens = [
        (0.30, "第一代  逐点特征 + 对称聚合",
         "PointNet\n共享 MLP + 最大池化",
         "缺陷：全局池化仅保留通道极值，\n邻域内无信息交换，缺失局部几何建模"),
        (3.42, "第二代  分层邻域建模",
         "PointNet++ / PointCNN / KPConv\n采样—分组—局部聚合",
         "改进：显式建模邻域相对位置\n代价：邻域尺度与分组策略需调节"),
        (6.54, "第三代  图卷积与动态邻域",
         "DGCNN / PU-GCN\n特征空间 kNN 图 + 边卷积",
         "改进：跨空间距离的语义邻域交换\n代价：逐层建图的时间与显存开销"),
        (9.66, "第四代  点云 Transformer",
         "PCT / Point Transformer /\nPU-Transformer\n自注意力实现全对全或邻域交互",
         "张力：全局注意力代价 $O(N^2)$，\n局部注意力感受野受限"),
    ]
    BW = 2.72
    for x, title, method, comment in gens:
        hl = "PU-Transformer" in method
        ax.text(x + BW / 2, 6.98, title, ha="center", fontsize=9.8, fontweight="bold",
                color="#B4573F" if hl else "#3B6FB6")
        box(ax, x, 5.05, BW, 1.62, method, fc=C_HL if hl else C_BOX,
            ec=C_HLE if hl else C_EDGE, fs=8.0 if hl else 8.6,
            lw=1.9 if hl else 1.4, bold=hl)
        box(ax, x, 3.20, BW, 1.42, comment, fc=C_GRAY, ec="#999", fs=8.0)
        arrow(ax, (x + BW / 2, 5.05), (x + BW / 2, 4.62), color="#999", lw=1.2)

    for i in range(3):
        x0 = gens[i][0] + BW
        x1 = gens[i + 1][0]
        arrow(ax, (x0, 5.86), (x1, 5.86), lw=1.8)

    box(ax, 1.60, 1.35, 9.40, 1.35,
        "贯穿主线：特征交互范围与计算代价之间的折中\n"
        "PU-Transformer 的通道偏移多头自注意力即针对该折中提出；本文第 4 章的结构改进以此为出发点",
        fc=C_HL, ec=C_HLE, fs=9.2, lw=1.8, bold=True)
    for x, _, _, _ in gens:
        arrow(ax, (x + BW / 2, 3.20), (x + BW / 2, 2.70), color="#C9A227", lw=1.0, ls="--")

    ax.text(6.3, 7.42, "图 2-3  点云特征学习骨干演进图", ha="center", fontsize=12.5,
            fontweight="bold")
    ax.text(6.3, 0.72,
            "注：横向箭头表示后一代方法针对前一代已知缺陷的改进方向；灰框为该代方法的已知缺陷或代价。"
            "各方法的机制描述与文献出处见 2.2 节正文。",
            ha="center", fontsize=8.4, color="#555")
    save(fig, "F2_3_backbone_evolution",
         {"figure_id": "2-3", "caption": "点云特征学习骨干演进图", "type": "schematic",
          "chapter": "2.2", "n_generations": 4})


# ============================================================
# 图 2-4 损失函数与质量维度关系图
# ============================================================
def fig_2_4_loss_quality_map():
    fig, ax = plt.subplots(figsize=(12.4, 7.8))
    ax.set_xlim(-0.55, 12.4); ax.set_ylim(0, 7.8); ax.axis("off")

    ax.text(1.55, 7.06, "质量维度", ha="center", fontsize=10.6, fontweight="bold", color="#3B6FB6")
    ax.text(6.20, 7.06, "度量 / 监督工具", ha="center", fontsize=10.6, fontweight="bold", color="#3B6FB6")
    ax.text(10.85, 7.06, "已知盲区", ha="center", fontsize=10.6, fontweight="bold", color="#3B6FB6")

    dims = [
        (5.42, "分布保真性\n$\\mathcal{Q}$ 与 $\\mathcal{Y}$ 的\n点集整体接近程度"),
        (3.94, "空间均匀性\n$\\mathcal{Q}$ 在曲面上\n分布是否规则"),
        (2.46, "表面贴合性\n$\\mathcal{Q}$ 到真实曲面\n的距离"),
        (0.98, "细节锐度\n锐利边缘与高曲率\n区域是否保留"),
    ]
    tools = [
        (5.42, "Chamfer 距离（训练与评价）"),
        (3.94, "均匀性损失 / NUC / $\\mathrm{cv}_{\\text{nn}}$"),
        (2.46, "点到面距离 P2F（需真值网格）"),
        (0.98, "局部误差分解 + 边缘监督 / 可视化"),
    ]
    blinds = [
        (5.42, "均值形式对局部聚簇\n惩罚被稀释"),
        (3.94, "依赖预设邻域半径；\n对偏离曲面不敏感"),
        (2.46, "对点的分布不敏感；\n仅在含网格测试集可算"),
        (0.98, "缺乏公认的独立\n标量指标"),
    ]
    for (y, t), (_, tl), (_, bl) in zip(dims, tools, blinds):
        box(ax, 0.30, y, 2.50, 1.20, t, fs=8.6)
        box(ax, 3.35, y, 5.70, 1.20, tl, fc="#F2F7FF", ec="#7A9CC6", fs=9.0)
        box(ax, 9.60, y, 2.50, 1.20, bl, fc=C_GRAY, ec="#999", fs=8.2)
        arrow(ax, (2.80, y + 0.60), (3.35, y + 0.60), color="#999", lw=1.2)
        arrow(ax, (9.05, y + 0.60), (9.60, y + 0.60), color="#C6867A", lw=1.2)

    # 维度间冲突关系（标注置于维度列左侧空白区，避免压住框内文字）
    ax.annotate("", xy=(0.14, 3.94 + 0.60), xytext=(0.14, 5.42 + 0.60),
                arrowprops=dict(arrowstyle="<|-|>", color=C_HLE, lw=1.7))
    ax.text(0.02, 5.06, "度量盲区冲突", fontsize=8.2, color=C_HLE, rotation=90,
            va="center", ha="center")
    ax.annotate("", xy=(0.14, 0.98 + 0.60), xytext=(0.14, 3.94 + 0.10),
                arrowprops=dict(arrowstyle="<|-|>", color=C_HLE, lw=1.7,
                                connectionstyle="arc3,rad=0.30"))
    ax.text(0.02, 2.60, "尺度矛盾", fontsize=8.2, color=C_HLE, rotation=90,
            va="center", ha="center")

    box(ax, 2.10, 0.16, 8.20, 0.62,
        "报告纪律：每组对比与消融同时报告 CD、HD、NUC 与 $\\mathrm{cv}_{\\text{nn}}$，"
        "并标注是否超过显著性门槛（$2\\mathrm{SE}$）",
        fc=C_HL, ec=C_HLE, fs=9.0, lw=1.7, bold=True)

    ax.text(6.2, 7.56, "图 2-4  损失函数与质量维度关系图", ha="center", fontsize=12.5,
            fontweight="bold")
    save(fig, "F2_4_loss_quality_map",
         {"figure_id": "2-4", "caption": "损失函数与质量维度关系图", "type": "schematic",
          "chapter": "2.5.4", "n_dimensions": 4,
          "note": "维度—工具—盲区三列对应关系与 2.1.3 / 2.5.4 节正文一致"})


def main():
    print("生成不依赖实验的示意图 -> %s" % OUTDIR)
    fig_1_1_task_illustration()
    fig_1_2_roadmap()
    fig_1_3_chapter_map()
    fig_2_1_timeline()
    fig_2_2_taxonomy()
    fig_2_3_backbone_evolution()
    fig_2_4_loss_quality_map()
    fig_3_2_quality_dims()
    fig_3_4_eval_protocol()
    fig_3_5_bottleneck()
    n = len([f for f in os.listdir(OUTDIR) if f.endswith(".png")])
    print("\n完成: %d 张 PNG" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
