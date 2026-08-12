"""论文级过程图与可视化产出。

设计原则
--------
1. **所有图都从真实落盘数据重绘**：函数只接受 npz/json 路径或数组，
   不接受「手填数字」。防止图与实验记录脱节。
2. **每张图旁边必须落一份同名 .json 数据源**，论文审稿要溯源时可直接给。
3. **点云图统一视角与配色**，跨方法对比才有意义（视角随机会误导读者）。
4. 中文字体：Windows 用 Microsoft YaHei；缺字体时自动退回英文标签，
   不静默出乱码方框。

产出的图类别（对应 PAPER_REMAKE_PLAN 图清单）
-------------------------------------------
- F-loss    : 训练曲线（loss / 各分项 / 学习率）
- F-metric  : 验证指标随 epoch 变化（CD/HD/NUC）
- F-cloud   : 点云定性对比（input / 各方法 / GT），支持局部放大
- F-ablation: 消融柱状图（带数值标注）
- F-noise   : 噪声鲁棒性折线（beta 扫描）
- F-hist    : 最近邻距离分布直方图（用于论证密度均匀性）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")          # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402

# ---- 中文字体 ----
_CJK_OK = False
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        _CJK_OK = True
        break
    except Exception:
        continue

DPI = 300                       # 顶刊要求 300+ dpi
CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def _L(zh: str, en: str) -> str:
    """有中文字体用中文，否则退回英文，不出乱码方框。"""
    return zh if _CJK_OK else en


def _save(fig, out_path: Path, data: dict) -> Path:
    """保存图 + 同名 json 数据源（论文溯源用）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    src = out_path.with_suffix(".data.json")
    src.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                             default=float), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# 训练曲线
# --------------------------------------------------------------------------
def plot_training_curves(history: dict, out_path: Path,
                         title: str = "") -> Path:
    """训练曲线：损失分项（对数轴）+ 比例类量（线性轴）+ 学习率。

    history 需含 'epoch' 与若干形如 'train_*' / 'val_*' / 'lr' 的等长序列。

    ⚠️ 实测教训（2026-08-11，R-001 冒烟）—— 两个必须避免的图表缺陷：
      1. **量纲混画**：`cd_bwd_share` 是比例（0~1），与 loss 值（1e-3 量级）
         塞进同一对数轴，会把 loss 曲线压到纵轴下方 1/3，严重压缩可读范围。
         故此处按后缀把「比例类」量自动分流到独立线性轴子图。
      2. **完全重合的曲线看不见**：B-001 纯 CD 配置下 `train_total ≡ train_cd`，
         后画的把先画的完全盖掉，图例 5 条但只看得到 4 条，读者会以为漏了数据。
         故对与已画曲线数值完全重合的序列改用虚线 + 加粗，并在图例标注 `(≡…)`。
    """
    ep = history.get("epoch") or list(range(len(next(
        v for k, v in history.items() if isinstance(v, list)))))
    all_keys = [k for k in history
                if k.startswith(("train_", "val_")) and isinstance(history[k], list)]

    # 比例/系数类：与 loss 不同量纲，必须分轴
    def _is_ratio(k: str) -> bool:
        return k.endswith(("_share", "_factor", "_ratio"))

    loss_keys = sorted(k for k in all_keys if not _is_ratio(k))
    ratio_keys = sorted(k for k in all_keys if _is_ratio(k))

    has_lr = "lr" in history
    ncol = 1 + (1 if ratio_keys else 0) + (1 if has_lr else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(6.0 * ncol, 4.2))
    axes = np.atleast_1d(axes)
    ai = 0

    # --- 损失分项 ---
    ax = axes[ai]
    ai += 1
    drawn: list[tuple[str, list]] = []
    for i, k in enumerate(loss_keys):
        vals = history[k]
        # 检测与已画曲线完全重合（否则会被静默盖掉）
        dup_of = None
        for pk, pv in drawn:
            if len(pv) == len(vals) and all(
                    abs(a - b) <= 1e-12 * max(1.0, abs(a)) for a, b in zip(pv, vals)):
                dup_of = pk
                break
        style = "--" if (dup_of or k.startswith("val_")) else "-"
        lw = 2.4 if dup_of else 1.6
        label = f"{k} (≡{dup_of})" if dup_of else k
        ax.plot(ep, vals, style, color=CB_COLORS[i % len(CB_COLORS)],
                label=label, linewidth=lw)
        drawn.append((k, vals))
    ax.set_xlabel(_L("训练轮次 (epoch)", "Epoch"))
    ax.set_ylabel(_L("损失值", "Loss"))
    ax.set_yscale("log")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, ncol=1)
    if title:
        ax.set_title(title)

    # --- 比例类（线性轴，0~1）---
    if ratio_keys:
        axr = axes[ai]
        ai += 1
        for i, k in enumerate(ratio_keys):
            axr.plot(ep, history[k], "-", color=CB_COLORS[i % len(CB_COLORS)],
                     label=k, linewidth=1.6)
        axr.set_xlabel(_L("训练轮次 (epoch)", "Epoch"))
        axr.set_ylabel(_L("比例", "Ratio"))
        axr.set_ylim(0, 1)
        axr.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axr.grid(alpha=0.3, linestyle=":")
        axr.legend(fontsize=8)
        axr.set_title(_L("双向 CD 占比（覆盖 vs 精度）",
                         "Bidirectional CD share"))

    if has_lr:
        ax2 = axes[ai]
        ax2.plot(ep, history["lr"], color=CB_COLORS[0], linewidth=1.6)
        ax2.set_xlabel(_L("训练轮次 (epoch)", "Epoch"))
        ax2.set_ylabel(_L("学习率", "Learning rate"))
        ax2.set_yscale("log")
        ax2.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    return _save(fig, out_path, history)


# --------------------------------------------------------------------------
# 验证指标曲线
# --------------------------------------------------------------------------
def plot_metric_curves(epochs: list[int], metrics: dict[str, list[float]],
                       out_path: Path, scale: float = 1e3) -> Path:
    """验证指标随 epoch 变化，**每个指标一个子图、各自独立标轴**。

    为什么不能画在同一根轴上（2026-08-11 实测发现的缺陷）：
      B-001 的 CD≈0.0021 / HD≈0.0080 / NUC≈0.55 跨两个数量级。
      单轴叠加时 HD/CD 量级比 3.95×，会把 CD 曲线压成一条视觉直线 ——
      CD 实际降了 23.7%，图上却看不出任何变化，严重误导读者。
      这与之前修掉的「比例类字段与 loss 混轴」是同一类错误。

    缩放规则：只有**距离类**指标按论文惯例 ×1e3；
      比例/无量纲类（NUC、share、ratio、acc 等）保持原值，
      否则 0.55 会被显示成 550，读者无法与文献对照。
    """
    ratio_like = ("nuc", "share", "ratio", "acc", "cv", "std")

    def _is_ratio(name: str) -> bool:
        return any(t in name.lower() for t in ratio_like)

    items = sorted(metrics.items())
    n = len(items)
    if n == 0:
        raise ValueError("metrics 为空，无可绘制指标")

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
    axes = axes[0]

    used_scales: dict[str, float] = {}
    for i, (name, vals) in enumerate(items):
        ax = axes[i]
        sc = 1.0 if _is_ratio(name) else scale
        used_scales[name] = sc
        ax.plot(epochs, [v * sc for v in vals], "-o", markersize=3,
                color=CB_COLORS[i % len(CB_COLORS)], linewidth=1.6, label=name)
        # 标出最优点（距离/误差类越小越好；比例类此处同样取最小，
        # 因为 NUC 与 cv 都是越小越均匀）
        best_i = min(range(len(vals)), key=lambda j: vals[j])
        ax.plot(epochs[best_i], vals[best_i] * sc, "*", markersize=15,
                color="#CC0000", zorder=5,
                label=f"best ep{epochs[best_i]}={vals[best_i]:.6g}")
        unit = "" if sc == 1.0 else f" (×{sc:.0e})"
        ax.set_title(f"{name}{unit}")
        ax.set_xlabel(_L("训练轮次 (epoch)", "Epoch"))
        ax.set_ylabel(f"{name}{unit}")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, out_path,
                 {"epochs": epochs, "metrics": metrics,
                  "scale": scale, "per_metric_scale": used_scales,
                  "note": "每个指标独立子图与独立标轴；比例类指标不缩放"})


# --------------------------------------------------------------------------
# 点云定性对比
# --------------------------------------------------------------------------
def plot_point_clouds(clouds: dict[str, np.ndarray], out_path: Path,
                      elev: float = 20.0, azim: float = 45.0,
                      point_size: float = 1.2,
                      zoom_box: tuple | None = None,
                      depth_shade: bool = True,
                      cull_backface: bool = True,
                      shared_axes: bool = True) -> Path:
    """点云并排对比。视角固定，保证跨方法可比。

    clouds : 有序字典 {标签: (N,3)}，建议顺序 input / 各方法 / GT
    zoom_box : (cx, cy, cz, half) 指定局部放大区域，则额外出一行放大图
    depth_shade : 按视线深度着色。3D 散点正投影会把前后表面重叠成「实心圆盘」，
        必须靠深度信息才能看出曲面结构 —— 否则论文图无法体现上采样质量。
    cull_backface : 剔除背面点（仅保留朝向观察者的半侧）。
        对闭合曲面尤其重要，否则背面点透过来使密度看起来虚高一倍。
    shared_axes : **所有子图共用同一坐标轴范围**（按本行全部点云的联合包围盒）。

    ⚠️ 经验教训 1：自检只能验「文件存在且非零」，验不出「图是否有效表达信息」。
       新增图类型后必须肉眼检查一次。
    ⚠️ 经验教训 2（实测，2026-08-11）：**独立坐标轴会系统性掩盖尺度误差**。
       R-001 管线验证的 1-epoch 输出，pred 实际最大半径仅 0.449 而 gt 为 1.000
       （收缩到 45%），但因每个子图各自 autoscale，图上三者看起来一样大 ——
       55% 的形变被完全隐藏。这类图会误导审稿人，故 `shared_axes` 默认为 True。
       仅在明确只看局部纹理、不看尺度时才允许关闭，且须在图注说明。
    """
    names = list(clouds.keys())
    n = len(names)
    nrow = 2 if zoom_box else 1
    fig = plt.figure(figsize=(3.3 * n, 3.5 * nrow))

    # 视线方向（用于背面剔除与深度排序）
    er, ar = np.deg2rad(elev), np.deg2rad(azim)
    view_dir = np.array([np.cos(er) * np.cos(ar),
                         np.cos(er) * np.sin(ar),
                         np.sin(er)])

    # --- 共享坐标轴范围：按「全部点云的联合包围盒」取等边立方体 ---
    # 等边是必须的：xyz 轴比例不一致会把球形误差画成椭球，同样是误导。
    lims = None
    if shared_axes:
        allpts = np.concatenate(
            [np.asarray(v, dtype=np.float64).reshape(-1, 3)
             for v in clouds.values() if len(v)], axis=0)
        ctr = (allpts.max(axis=0) + allpts.min(axis=0)) / 2
        half = float((allpts.max(axis=0) - allpts.min(axis=0)).max()) / 2
        half = max(half, 1e-6) * 1.05          # 留 5% 边距
        lims = [(ctr[d] - half, ctr[d] + half) for d in range(3)]

    meta: dict = {}
    for r in range(nrow):
        for i, name in enumerate(names):
            ax = fig.add_subplot(nrow, n, r * n + i + 1, projection="3d")
            pc = np.asarray(clouds[name], dtype=np.float64)
            n_raw = len(pc)

            if r == 1:
                cx, cy, cz, half_z = zoom_box
                m = ((np.abs(pc[:, 0] - cx) < half_z) &
                     (np.abs(pc[:, 1] - cy) < half_z) &
                     (np.abs(pc[:, 2] - cz) < half_z))
                pc = pc[m]

            if cull_backface and len(pc) > 0:
                centroid = pc.mean(axis=0)
                # 法向近似用「质心 -> 点」方向；朝向观察者的保留
                outward = pc - centroid
                nrm = np.linalg.norm(outward, axis=1, keepdims=True)
                nrm[nrm < 1e-12] = 1e-12
                facing = (outward / nrm) @ view_dir
                keep = facing > -0.05
                if keep.sum() >= 10:      # 防止把点全剔掉
                    pc = pc[keep]

            depth = pc @ view_dir if len(pc) else np.zeros(0)
            if depth_shade and len(pc):
                order = np.argsort(depth)       # 远的先画，近的覆盖在上
                pc, depth = pc[order], depth[order]
                sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size,
                                c=depth, cmap="viridis", alpha=0.9,
                                linewidths=0, depthshade=False)
            else:
                sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size,
                                c=CB_COLORS[i % len(CB_COLORS)], alpha=0.8,
                                linewidths=0)
            del sc

            # 尺度信息进标题：即使有人误关 shared_axes，数字仍能暴露收缩
            r_max = (float(np.linalg.norm(pc - pc.mean(axis=0), axis=1).max())
                     if len(pc) else 0.0)
            tag = _L("可见", "vis")
            ax.set_title(f"{name}\n({n_raw} pts, {len(pc)} {tag}, "
                         f"r={r_max:.3f})", fontsize=9)
            ax.view_init(elev=elev, azim=azim)
            if lims is not None and r == 0:
                ax.set_xlim(*lims[0])
                ax.set_ylim(*lims[1])
                ax.set_zlim(*lims[2])
            ax.set_axis_off()
            ax.set_box_aspect((1, 1, 1))
            meta.setdefault(name, {})[f"row{r}"] = {
                "n_raw": int(n_raw), "n_drawn": int(len(pc)),
                "max_radius_drawn": r_max}

    fig.tight_layout()
    meta["_view"] = {"elev": elev, "azim": azim, "zoom_box": zoom_box,
                     "depth_shade": depth_shade,
                     "cull_backface": cull_backface,
                     "shared_axes": shared_axes,
                     "shared_lims": lims}
    return _save(fig, out_path, meta)


# --------------------------------------------------------------------------
# 消融柱状图
# --------------------------------------------------------------------------
def plot_ablation_bars(labels: list[str], values: dict[str, list[float]],
                       out_path: Path, ylabel: str = "",
                       scale: float = 1e3) -> Path:
    """消融对比柱状图，柱顶标注数值（审稿人要看具体数）。"""
    n_grp = len(labels)
    n_ser = len(values)
    width = 0.8 / max(n_ser, 1)
    x = np.arange(n_grp)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * n_grp), 4.2))
    for i, (name, vals) in enumerate(values.items()):
        pos = x + i * width - 0.4 + width / 2
        v = [a * scale for a in vals]
        bars = ax.bar(pos, v, width * 0.92,
                      color=CB_COLORS[i % len(CB_COLORS)], label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel or _L(f"指标值 (×{scale:.0e})",
                               f"Metric (×{scale:.0e})"))
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    if n_ser > 1:
        ax.legend(fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    return _save(fig, out_path,
                 {"labels": labels, "values": values, "scale": scale})


# --------------------------------------------------------------------------
# 噪声鲁棒性
# --------------------------------------------------------------------------
def plot_noise_robustness(betas: list[float], series: dict[str, list[float]],
                          out_path: Path, ylabel: str = "CD",
                          scale: float = 1e3) -> Path:
    """噪声水平 beta 扫描下各方法的指标折线。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, vals) in enumerate(series.items()):
        ax.plot([b * 100 for b in betas], [v * scale for v in vals], "-o",
                markersize=4, color=CB_COLORS[i % len(CB_COLORS)],
                label=name, linewidth=1.8)
    ax.set_xlabel(_L("噪声水平 β (%)", "Noise level β (%)"))
    ax.set_ylabel(f"{ylabel} (×{scale:.0e})")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path,
                 {"betas": betas, "series": series, "scale": scale})


# --------------------------------------------------------------------------
# 最近邻距离分布
# --------------------------------------------------------------------------
def plot_nn_histogram(clouds: dict[str, np.ndarray], out_path: Path,
                      bins: int = 60) -> Path:
    """各方法输出的「最近邻距离」分布。

    均匀分布 -> 窄峰；聚集 -> 左侧长尾（大量极小间距）。
    这是比 NUC 单个数字更能说服审稿人的可视化证据。
    """
    from scipy.spatial import cKDTree

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    stats = {}
    for i, (name, pc) in enumerate(clouds.items()):
        pc = np.asarray(pc, dtype=np.float64)
        tree = cKDTree(pc)
        d, _ = tree.query(pc, k=2)
        nn = d[:, 1]
        ax.hist(nn, bins=bins, histtype="step", density=True,
                color=CB_COLORS[i % len(CB_COLORS)], label=name, linewidth=1.6)
        stats[name] = {"mean": float(nn.mean()), "std": float(nn.std()),
                       "cv": float(nn.std() / max(nn.mean(), 1e-12)),
                       "min": float(nn.min()), "max": float(nn.max())}
    ax.set_xlabel(_L("最近邻距离", "Nearest-neighbour distance"))
    ax.set_ylabel(_L("概率密度", "Density"))
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path, stats)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def self_check() -> bool:
    import tempfile
    ok = True
    print("=" * 74)
    print("visualize 自检")
    print("=" * 74)
    print(f"中文字体可用: {_CJK_OK}"
          f"{'' if _CJK_OK else '  -> 图内标签自动退回英文'}")

    tmp = Path(tempfile.mkdtemp(prefix="viz_check_"))
    rng = np.random.default_rng(0)

    # 1. 训练曲线
    # 刻意构造两个曾在 R-001 暴露的缺陷场景：
    #   - train_total 与 train_cd 完全重合（纯 CD 配置的真实情形）
    #   - train_cd_bwd_share 是比例量，不能与 loss 同轴
    cd_series = list(np.linspace(0.05, 0.01, 10))
    hist = {"epoch": list(range(10)),
            "train_cd": cd_series,
            "train_total": list(cd_series),          # 完全重合
            "train_cd_bwd_share": list(np.linspace(0.45, 0.62, 10)),
            "val_cd": list(np.linspace(0.06, 0.02, 10)),
            "lr": list(np.geomspace(1e-3, 1e-4, 10))}
    p = plot_training_curves(hist, tmp / "f_loss.png", title="self-check")
    ok &= p.exists() and p.with_suffix(".data.json").exists()
    print(f"[1] 训练曲线 {p.name} {p.stat().st_size} B  json={p.with_suffix('.data.json').exists()}")

    # 1b. 量纲分离 + 重合曲线可见（防回归）
    # 判据：比例类字段必须被分流到独立子图，不与 loss 同轴
    _ratio = sorted(k for k in hist
                    if k.startswith(("train_", "val_"))
                    and k.endswith(("_share", "_factor", "_ratio")))
    _loss = sorted(k for k in hist
                   if k.startswith(("train_", "val_"))
                   and not k.endswith(("_share", "_factor", "_ratio")))
    c1b = (_ratio == ["train_cd_bwd_share"]
           and set(_loss) == {"train_cd", "train_total", "val_cd"})
    ok &= c1b
    print(f"[1b] 量纲分流 {'PASS' if c1b else 'FAIL'}: "
          f"比例轴={_ratio} 损失轴={_loss}")

    # 2. 指标曲线
    p = plot_metric_curves(list(range(10)),
                           {"CD": list(np.linspace(6e-4, 4e-4, 10)),
                            "HD": list(np.linspace(5e-3, 3e-3, 10))},
                           tmp / "f_metric.png")
    ok &= p.exists()
    print(f"[2] 指标曲线 {p.name} {p.stat().st_size} B")

    # 3. 点云对比（含局部放大）
    sph = rng.standard_normal((2048, 3))
    sph /= np.linalg.norm(sph, axis=1, keepdims=True)
    # 放大框取球冠一小块：half 太大会把整个球框进来，放大图失去意义
    p = plot_point_clouds({"input": sph[:512], "ours": sph, "GT": sph},
                          tmp / "f_cloud.png", point_size=4.0,
                          zoom_box=(0.0, 0.0, 0.92, 0.30))
    ok &= p.exists()
    meta = json.loads(p.with_suffix(".data.json").read_text(encoding="utf-8"))
    zoom_n = meta["ours"]["row1"]["n_drawn"]
    print(f"[3] 点云对比 {p.name} {p.stat().st_size} B  放大区可见点={zoom_n}")
    # 放大图至少要有一定点数才有展示价值
    ok &= zoom_n >= 30

    # 3b. 共享坐标轴必须真生效（防回归）
    # 背景：R-001 管线验证时 pred 收缩到 gt 的 45%，但独立 autoscale 让三图看起来
    #      一样大，55% 的形变被图表隐藏。这条断言锁住修复。
    # 判据：① 三子图共用同一 lims；② 标题里的 r 值必须反映真实收缩（比值 < 0.6）
    shrunk = sph * 0.45
    p = plot_point_clouds({"input": sph[:512], "pred": shrunk, "GT": sph},
                          tmp / "f_cloud_scale.png", point_size=4.0)
    meta = json.loads(p.with_suffix(".data.json").read_text(encoding="utf-8"))
    lims = meta["_view"]["shared_lims"]
    r_pred = meta["pred"]["row0"]["max_radius_drawn"]
    r_gt = meta["GT"]["row0"]["max_radius_drawn"]
    ratio = r_pred / max(r_gt, 1e-12)
    c3b = (meta["_view"]["shared_axes"] is True and lims is not None
           and ratio < 0.6)
    ok &= c3b
    print(f"[3b] 共享坐标轴 {'PASS' if c3b else 'FAIL'}: "
          f"r_pred={r_pred:.3f} r_gt={r_gt:.3f} ratio={ratio:.3f} "
          f"(期望 <0.6，真实缩放 0.45) lims={lims[0] if lims else None}")

    # 4. 消融柱状图
    p = plot_ablation_bars(["B1 CD", "B2 +adv", "B3 +uni", "B4 full"],
                           {"CD": [6.0e-4, 5.5e-4, 5.2e-4, 4.8e-4]},
                           tmp / "f_abl.png")
    ok &= p.exists()
    print(f"[4] 消融柱图 {p.name} {p.stat().st_size} B")

    # 5. 噪声折线
    p = plot_noise_robustness([0.0, 0.005, 0.01, 0.02],
                              {"ours": [4.5e-4, 4.6e-4, 6.1e-4, 1.05e-3],
                               "PU-Trans": [4.5e-4, 4.5e-4, 6.1e-4, 1.06e-3]},
                              tmp / "f_noise.png")
    ok &= p.exists()
    print(f"[5] 噪声折线 {p.name} {p.stat().st_size} B")

    # 6. 最近邻直方图 —— 均匀 vs 聚集应有明显区别
    clumped = sph.copy()
    clumped[:600] = sph[600:1200] + rng.standard_normal((600, 3)) * 1e-3
    p = plot_nn_histogram({"uniform": sph, "clumped": clumped},
                          tmp / "f_hist.png")
    ok &= p.exists()
    st = json.loads(p.with_suffix(".data.json").read_text(encoding="utf-8"))
    print(f"[6] 最近邻直方图 {p.name} {p.stat().st_size} B")
    print(f"    uniform  cv={st['uniform']['cv']:.4f}")
    print(f"    clumped  cv={st['clumped']['cv']:.4f}  (聚集应更大)")
    ok &= st["clumped"]["cv"] > st["uniform"]["cv"]

    print()
    print(f"临时产物目录: {tmp}")
    print(f"自检结果: {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
