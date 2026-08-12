# -*- coding: utf-8 -*-
"""用 B-001 真实 metrics.json 验证新选点器 —— 不能只靠合成数据自检。

输出：runs/B001_reproduce/selection_replay.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.metrics.selection import (CompositeSelector, convergence_check,
                                      plateau_stats)

RUN = ROOT / "runs" / "B001_reproduce"


def main() -> int:
    mpath = RUN / "metrics.json"
    if not mpath.exists():
        print(f"[FAIL] 找不到 {mpath}")
        return 1
    blob = json.loads(mpath.read_text(encoding="utf-8"))
    recs = blob["records"] if isinstance(blob, dict) else blob
    print(f"载入 {len(recs)} 条 epoch 记录  <- {mpath.name}")

    # ---- 1. 原 cd-only 选点会选谁 ----
    cd_best = min((r for r in recs if r.get("monitor_cd") is not None),
                  key=lambda r: r["monitor_cd"])
    print(f"\n[cd-only 选点]  ep{cd_best['epoch']}  "
          f"cd={cd_best['monitor_cd']:.6f}  hd={cd_best['monitor_hd']:.6f}  "
          f"nuc={cd_best['monitor_nuc']:.6f}")

    # ---- 2. 新综合选点会选谁 ----
    sel = CompositeSelector()
    for r in recs:
        sel.update(epoch=r["epoch"], cd=r.get("monitor_cd"),
                   hd=r.get("monitor_hd"), nuc=r.get("monitor_nuc"))
    be = sel.best_epoch
    br = next(r for r in recs if r["epoch"] == be)
    print(f"[综合选点]      ep{be}  cd={br['monitor_cd']:.6f}  "
          f"hd={br['monitor_hd']:.6f}  nuc={br['monitor_nuc']:.6f}")
    print(f"                权重 {sel.weights}  综合分 {sel.best_score:.6f}")

    # ---- 3. 三项各自变化了多少 ----
    print("\n[两种选点的逐项对比]  负=综合选点更好")
    deltas = {}
    for k in ("cd", "hd", "nuc"):
        a = cd_best[f"monitor_{k}"]
        b = br[f"monitor_{k}"]
        d = (b - a) / a * 100.0
        deltas[k] = d
        print(f"  {k:4s}  cd-only={a:.6f}  综合={b:.6f}  {d:+.2f}%")

    # ---- 4. 平台区统计（论文主表数字）----
    st = plateau_stats(recs, frac=0.5)
    print(f"\n[平台区 ep{st['epoch_range'][0]}-{st['epoch_range'][1]}"
          f"  n={st['plateau_n']}]  论文主表报这个")
    for k in ("cd", "hd", "nuc"):
        s = st[k]
        sig = s["best_sigma_from_mean"]
        print(f"  {k:4s}  {s['plateau_mean']:.6f} ± {s['plateau_std']:.6f}"
              f"   最优 {s['best']:.6f} @ep{s['best_epoch']}"
              + (f"  ({sig:+.2f}σ)" if sig is not None else ""))

    # ---- 5. 收敛判据（150 epoch 的依据）----
    cv = convergence_check(recs, window=10)
    print(f"\n[收敛检查]  ep{cv['range_a'][0]}-{cv['range_a'][1]} vs "
          f"ep{cv['range_b'][0]}-{cv['range_b'][1]}")
    for k in ("cd", "hd", "nuc"):
        c = cv[k]
        print(f"  {k:4s}  {c['mean_a']:.6f} -> {c['mean_b']:.6f}  "
              f"{c['change_pct']:+.2f}%   收敛={c['converged']}")

    out = {"source": str(mpath), "n_records": len(recs),
           "cd_only_pick": {"epoch": cd_best["epoch"],
                            "cd": cd_best["monitor_cd"],
                            "hd": cd_best["monitor_hd"],
                            "nuc": cd_best["monitor_nuc"]},
           "composite_pick": {"epoch": be, "cd": br["monitor_cd"],
                              "hd": br["monitor_hd"], "nuc": br["monitor_nuc"],
                              "score": sel.best_score,
                              "weights": sel.weights},
           "delta_pct": deltas,
           "plateau": st, "convergence": cv}
    dst = RUN / "selection_replay.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
