# -*- coding: utf-8 -*-
"""跨组点云对比图 —— 论文里的视觉证据。

同一样本、同一 epoch 阶段，把 baseline 与各消融组的 pred 并排渲染，
并叠加最近邻距离着色（把 NUC/均匀性的差异变成看得见的疏密）。

设计约束:
  * 只读各 run 落盘的 clouds/*.npz，不重新推理（不抢 GPU、不引入随机性）
  * 每张图落同名 .data.json，记录 run/epoch/样本 id 与实测统计量
  * 未完成的组自动跳过

用法:
    python scripts/build_cloud_comparison.py
    python scripts/build_cloud_comparison.py --epoch 149 --base B002_baseline150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from puvnet.viz.visualize import _save  # noqa: E402

DESIGN = ROOT / "runs" / "ablation_design" / "ablation_matrix.json"


# --------------------------------------------------------------------------
def nn_dist(p: np.ndarray) -> np.ndarray:
    """每点到最近邻的距离（不含自身）。"""
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def uniformity(p: np.ndarray) -> dict:
    """均匀性统计：最近邻距离的变异系数 cv 比 NUC 更灵敏（本项目实测）。"""
    nn = nn_dist(p)
    return {"nn_mean": float(nn.mean()), "nn_std": float(nn.std()),
            "nn_cv": float(nn.std() / nn.mean()) if nn.mean() > 0 else 0.0}


def pick_snapshot(run_dir: Path, epoch: int | None,
                  sample_idx: int | None) -> tuple[Path, int, int] | None:
    """挑一个快照文件。epoch=None 取最大 epoch；sample=None 取最小 index。"""
    mf = run_dir / "clouds_manifest.json"
    if not mf.exists():
        return None
    man = json.loads(mf.read_text(encoding="utf-8"))
    if not man:
        return None
    if epoch is None:
        entry = max(man, key=lambda e: e["epoch"])
    else:
        cand = [e for e in man if e["epoch"] == epoch]
        if not cand:                       # 该 epoch 没存，退化到最接近的
            entry = min(man, key=lambda e: abs(e["epoch"] - epoch))
        else:
            entry = cand[0]
    samples = entry["samples"]
    if sample_idx is None:
        s = min(samples, key=lambda x: x["index"])
    else:
        cand = [x for x in samples if x["index"] == sample_idx]
        if not cand:
            return None
        s = cand[0]
    f = run_dir / "clouds" / s["file"]
    return (f, entry["epoch"], s["index"]) if f.exists() else None


# --------------------------------------------------------------------------
def fig_compare(entries: list[dict], out_path: Path,
                view: tuple[float, float] = (18, -60)) -> Path:
    """并排渲染各组 pred（按最近邻距离着色）。第一列为 GT 参考。"""
    n = len(entries) + 1
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.9 * nrow))

    gt = entries[0]["gt"]
    panels = [{"title": "Ground Truth", "pts": gt, "tag": "GT"}]
    for e in entries:
        panels.append({"title": e["label"], "pts": e["pred"],
                       "tag": e["label"]})

    # 统一色标范围，否则各子图颜色不可比
    all_nn = [nn_dist(p["pts"]) for p in panels]
    vmin = float(min(a.min() for a in all_nn))
    vmax = float(np.percentile(np.concatenate(all_nn), 97))

    stats = {}
    for i, (p, nn) in enumerate(zip(panels, all_nn), start=1):
        ax = fig.add_subplot(nrow, ncol, i, projection="3d")
        pts = p["pts"]
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=nn, s=4.5,
                        cmap="viridis", vmin=vmin, vmax=vmax,
                        linewidths=0, alpha=0.95)
        u = uniformity(pts)
        stats[p["tag"]] = u
        ax.set_title(f"{p['title']}\nnn_cv={u['nn_cv']:.4f}  n={len(pts)}",
                     fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=view[0], azim=view[1])
        # 等比例，避免形状被拉伸误导
        rng = np.ptp(pts, axis=0).max() / 2
        ctr = pts.mean(axis=0)
        for setlim, c in ((ax.set_xlim, ctr[0]), (ax.set_ylim, ctr[1]),
                          (ax.set_zlim, ctr[2])):
            setlim(c - rng, c + rng)

    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.55, pad=0.02,
                      location="right")
    # 注意方向：viridis 深紫=小值=最近邻很近=局部聚簇；黄=大值=稀疏/空洞
    cb.set_label("最近邻距离  深紫=聚簇  黄=稀疏", fontsize=8)
    fig.suptitle("上采样结果对比（颜色=最近邻距离，nn_cv 越小越均匀）",
                 fontsize=10.5)
    return _save(fig, out_path,
                 {"panels": [p["tag"] for p in panels],
                  "uniformity": stats, "vmin": vmin, "vmax": vmax,
                  "colorbar_direction": ("viridis: 深紫=小值=聚簇, "
                                         "黄=大值=稀疏"),
                  "note": "pred 直接取训练期落盘快照，未重新推理"})


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="B002_baseline150")
    ap.add_argument("--epoch", type=int, default=None,
                    help="取哪个 epoch 的快照（默认各 run 的最大 epoch）")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--outdir", default="paper_assets")
    args = ap.parse_args()

    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    runs = [("baseline", ROOT / "runs" / args.base)]
    for name, g in design["groups"].items():
        runs.append((name, ROOT / g["out_dir"]))

    entries, missing = [], []
    ref_idx = args.sample
    for label, rd in runs:
        got = pick_snapshot(rd, args.epoch, ref_idx)
        if got is None:
            missing.append(label)
            continue
        f, ep, idx = got
        if ref_idx is None:
            ref_idx = idx           # 后续组锁定同一样本，保证可比
        d = np.load(f)
        entries.append({"label": f"{label}\n(ep{ep})", "pred": d["pred"],
                        "gt": d["gt"], "input": d["input"],
                        "run": rd.name, "epoch": ep, "index": idx})

    print("=" * 70)
    print("跨组点云对比")
    print("=" * 70)
    print(f"可用 : {[e['run'] for e in entries]}")
    print(f"缺失 : {missing if missing else '(无)'}")
    if not entries:
        print("[WAIT] 没有任何可用快照，先把 baseline 跑完。")
        return 1
    print(f"样本 : idx={ref_idx}")

    outdir = ROOT / args.outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    p = fig_compare(entries, outdir / f"T4_cloud_cmp_idx{ref_idx}.png")
    print(f"\n{p.name}  {p.stat().st_size} B")
    meta = json.loads(p.with_suffix(".data.json").read_text(encoding="utf-8"))
    print("\n均匀性 nn_cv（越小越均匀）:")
    for k, v in meta["uniformity"].items():
        print(f"  {k.splitlines()[0]:<20} {v['nn_cv']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
