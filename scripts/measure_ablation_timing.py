# -*- coding: utf-8 -*-
"""8 组消融的耗时实测 + 排产重算。

动机：B2 冒烟实测 3.89 s/epoch vs baseline 0.5 s/epoch = 7.8 倍。
      原先"8 组 x 150 epoch ≈ 37 h"的估算完全作废（它假设各组同速）。
      必须按组实测每步耗时，重算总排产。

方法：每组用相同的小数据量跑 3 epoch，取后 2 个 epoch 的均值
      （第 1 个含 CUDA 预热与 cudnn autotune，必须丢弃）。
      再按"全量 patch 数 / 小样本 patch 数"线性外推到 150 epoch。

⚠️ 线性外推的前提：单 epoch 耗时随 patch 数线性增长。
   本脚本用两个不同 limit 实测同一组来验证该前提，不假设成立。

输出：runs/ablation_design/timing.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ablation_design"

GROUPS = ["b002_baseline150", "abl_A1_cd_balance", "abl_A2_cd_boost_bwd",
          "abl_B1_adv_fixed", "abl_B2_adv_adaptive", "abl_C1_uniform",
          "abl_D1_scale_qk", "abl_AC_combo", "abl_BD_combo"]

EP_RE = re.compile(r"ep(\d+).*?\((\d+\.?\d*)s")


def run_one(cfg_name: str, limit: int, n_ep: int = 3) -> list[float]:
    """跑 n_ep 个 epoch，返回每 epoch 秒数。"""
    tag = f"runs/TIMING_{cfg_name}_{limit}"
    cmd = [sys.executable, "scripts/train_pu.py",
           "--config", f"configs/{cfg_name}.yaml",
           "--override", f"epochs={n_ep}", "lr_step=2",
           f"data.limit={limit}", f"out_dir={tag}",
           "select_warmup=2", f"dump_cloud_every={n_ep}"]
    env = {"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8",
           "PYTHONUTF8": "1"}
    import os
    e = dict(os.environ); e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=e)
    if p.returncode != 0:
        print(f"    [FAIL] {cfg_name} rc={p.returncode}")
        print("    " + (p.stderr or "")[-400:])
        return []
    secs = [float(m.group(2)) for m in EP_RE.finditer(p.stdout or "")]
    return secs


def main() -> int:
    print("=" * 74)
    print("消融组耗时实测")
    print("=" * 74)

    # --- 先验证线性外推前提：同一组两个 limit ---
    print("\n[前提验证] 单 epoch 耗时是否随 patch 数线性增长")
    probe = {}
    for lim in (300, 600):
        s = run_one("b002_baseline150", lim, 3)
        if len(s) < 2:
            print("  探测失败，中止")
            return 1
        probe[lim] = sum(s[1:]) / len(s[1:])
        print(f"  limit={lim:<5} 稳态 {probe[lim]:.3f} s/ep")
    ratio_data = 600 / 300
    ratio_time = probe[600] / probe[300]
    lin_err = abs(ratio_time - ratio_data) / ratio_data * 100
    print(f"  数据量比 {ratio_data:.2f}x  耗时比 {ratio_time:.2f}x  "
          f"偏差 {lin_err:.1f}%")
    linear_ok = lin_err < 25
    print(f"  → {'✅ 近似线性，可外推（偏差已记录）' if linear_ok else '★非线性，外推不可靠'}")

    # --- 各组实测 ---
    LIM = 600
    print(f"\n[各组实测] limit={LIM}, 3 epoch, 取后 2 个均值")
    res = {}
    base_s = None
    for g in GROUPS:
        t0 = time.time()
        s = run_one(g, LIM, 3)
        if len(s) < 2:
            res[g] = {"error": "run failed"}
            continue
        steady = sum(s[1:]) / len(s[1:])
        if g == "b002_baseline150":
            base_s = steady
        res[g] = {"per_epoch_sec_at_limit": steady, "raw": s,
                  "wall_probe_sec": round(time.time() - t0, 1)}
        print(f"  {g:<24} {steady:>7.3f} s/ep"
              + (f"  ({steady/base_s:>5.2f}x baseline)" if base_s else ""))

    # --- 外推到全量 150 epoch ---
    # 全量 patch 数从 B-001 记录取（train=65550 @ val_ratio=0.05 of 69000）
    FULL_TRAIN = 65550
    print(f"\n[外推] 全量 train patch = {FULL_TRAIN}, 目标 150 epoch")
    scale = FULL_TRAIN / (LIM * 0.95)      # limit 后再切 5% val
    print(f"  外推系数 = {FULL_TRAIN} / ({LIM} x 0.95) = {scale:.1f}x")
    total_h = 0.0
    print(f"\n  {'组':<24} {'s/ep(全量)':>12} {'150ep 小时':>11}")
    for g, r in res.items():
        if "error" in r:
            print(f"  {g:<24} {'FAILED':>12}")
            continue
        full_ep = r["per_epoch_sec_at_limit"] * scale
        hours = full_ep * 150 / 3600
        r["extrapolated_sec_per_epoch_full"] = full_ep
        r["hours_150ep"] = hours
        if g != "b002_baseline150":
            total_h += hours
        print(f"  {g:<24} {full_ep:>12.1f} {hours:>11.2f}")

    print(f"\n  8 组消融合计（不含 baseline）: {total_h:.1f} h "
          f"= {total_h/24:.1f} 天（3090 串行）")
    b = res.get("b002_baseline150", {}).get("hours_150ep")
    if b:
        print(f"  参照：B-002 baseline 本身 {b:.2f} h")

    # 与 B-001 实测校准（3090 干净基准 104.25 ms/步，已知 100ep=3.12h）
    print("\n[校准] B-001 实测 100 epoch 用了约 3.87 h（02:09->05:54 含验证与落盘）")
    print("       故 baseline 150 epoch 预计约 5.8 h")
    if b:
        print(f"       本次外推给出 {b:.2f} h，"
              f"{'✅ 量级吻合' if 3 < b < 9 else '★与实测偏差过大，外推不可信'}")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "timing.json"
    dst.write_text(json.dumps(
        {"linearity_probe": {"limits": probe, "data_ratio": ratio_data,
                             "time_ratio": ratio_time,
                             "deviation_pct": lin_err, "linear_ok": linear_ok},
         "measure_limit": LIM, "full_train_patches": FULL_TRAIN,
         "extrapolation_scale": scale,
         "groups": res, "total_ablation_hours": total_h},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
