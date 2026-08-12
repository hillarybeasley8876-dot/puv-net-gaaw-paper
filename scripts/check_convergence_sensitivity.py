# -*- coding: utf-8 -*-
"""窗口敏感性检查 —— 收敛结论到底稳不稳。

动机（诚实记录）：助手先前口头报的「ep70-79 -> ep90-99：cd -0.82% / hd -5.27%
/ nuc -3.84%」是手挑窗口算的。改用标准 window=10（ep80-89 -> ep90-99）后
得到 cd -1.55% / hd -6.40% / nuc -0.19%，nuc 的收敛判定直接翻转。

结论若随窗口翻转，就不能拿它当「100 -> 150 epoch」的依据。本脚本把所有
合理窗口都算一遍，看哪些结论稳定、哪些是窗口伪影。

输出：runs/B001_reproduce/convergence_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.metrics.selection import _mean_std, convergence_check

RUN = ROOT / "runs" / "B001_reproduce"
KEYS = ("cd", "hd", "nuc")


def seg_mean(recs, key, lo, hi):
    m, _ = _mean_std([r.get("monitor_" + key) for r in recs
                      if lo <= r["epoch"] <= hi])
    return m


def main() -> int:
    blob = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    recs = blob["records"] if isinstance(blob, dict) else blob
    print(f"载入 {len(recs)} 条记录\n")

    out = {"n": len(recs), "windows": {}, "pairs": {}}

    # ---- A. 标准 convergence_check 扫窗口宽度 ----
    print("[A] 相邻等宽窗口（window=W：最后 2W 个 epoch 两两比较）")
    print(f"{'W':>4} | " + " | ".join(f"{k:>22}" for k in KEYS))
    for w in (5, 10, 15, 20, 25):
        cv = convergence_check(recs, keys=KEYS, window=w)
        cells = []
        rec = {}
        for k in KEYS:
            c = cv[k]
            rec[k] = {"change_pct": c["change_pct"], "converged": c["converged"]}
            cells.append(f"{c['change_pct']:+7.2f}% {'收敛' if c['converged'] else '未收敛'}"
                         .rjust(22))
        out["windows"][f"w{w}"] = {"range_a": cv["range_a"],
                                   "range_b": cv["range_b"], **rec}
        print(f"{w:>4} | " + " | ".join(cells))

    # ---- B. 助手先前口头报的那个窗口，复算 ----
    print("\n[B] 复算助手先前口头报的窗口 ep70-79 -> ep90-99（非相邻，跨 10 epoch 空隙）")
    rec = {}
    for k in KEYS:
        a = seg_mean(recs, k, 70, 79)
        b = seg_mean(recs, k, 90, 99)
        d = (b - a) / a * 100.0
        rec[k] = {"mean_a": a, "mean_b": b, "change_pct": d}
        print(f"  {k:4s} {a:.6f} -> {b:.6f}  {d:+.2f}%")
    out["pairs"]["ep70_79_vs_ep90_99"] = rec
    print("  ^ 与助手先前口述（cd -0.82% / hd -5.27% / nuc -3.84%）对照，"
          "确认口述数字是否可复现")

    # ---- C. 末段整体趋势：后 30 epoch 线性斜率符号 ----
    print("\n[C] 后 30 epoch (ep70-99) 线性拟合斜率（每 epoch 相对变化）")
    tail = [r for r in recs if r["epoch"] >= 70]
    rec = {}
    for k in KEYS:
        xs = [r["epoch"] for r in tail]
        ys = [r.get("monitor_" + k) for r in tail]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den
        # 折算成「每 10 epoch 相对变化百分比」，便于与 A/B 对照
        per10 = slope * 10 / my * 100.0
        rec[k] = {"slope": slope, "pct_per_10ep": per10, "mean": my}
        print(f"  {k:4s} 斜率={slope:+.3e}/ep  = {per10:+.2f}% /10ep  "
              f"{'仍在下降' if per10 < -0.5 else ('基本持平' if abs(per10) <= 0.5 else '在上升')}")
    out["tail_slope_ep70_99"] = rec

    # ---- D. 结论稳定性判定 ----
    print("\n[D] 结论稳定性")
    verdict = {}
    for k in KEYS:
        flags = [out["windows"][f"w{w}"][k]["converged"] for w in (5, 10, 15, 20, 25)]
        stable = len(set(flags)) == 1
        slope_down = out["tail_slope_ep70_99"][k]["pct_per_10ep"] < -0.5
        verdict[k] = {"converged_flags": flags, "flag_stable": stable,
                      "tail_still_decreasing": slope_down}
        print(f"  {k:4s} 各窗口收敛标记={flags}  "
              f"{'一致' if stable else '★随窗口翻转（不可作为依据）'}  |  "
              f"末段斜率{'仍降' if slope_down else '持平'}")
    out["verdict"] = verdict

    dst = RUN / "convergence_sensitivity.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
