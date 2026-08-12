# -*- coding: utf-8 -*-
"""消融实验功效分析 —— 跑之前就判断"能不能测出来"。

动机：改进 A 定标结果显示其权重扰动仅 ±7.9%（w_fwd 1.079 / w_bwd 0.921），
      而 B-001 平台区 CD 的相对噪声是 ±2.4%（σ=0.000051 / 均值 0.002166）。
      若预期效应量小于噪声，则跑完必然得到"无显著差异"，
      3 小时机时换一个注定测不出的结论。

本脚本用 B-001 的真实噪声水平，算出：
  1. 单次 run 的可检测最小效应量（MDE, minimum detectable effect）
  2. 各改进的预期效应量（能估的估，不能估的标注"未知"）
  3. 哪些改进需要多种子重复才有统计功效

⚠️ 诚实边界：改进 B/C/D 的效应量**无法在跑之前估计**（它们是开关型改动，
   不像 A 有可解析的扰动幅度）。本脚本只对 A 给出定量预判，
   对 B/C/D 只给出"能测出多大效应"的门槛，不编造其预期效果。

输出：runs/ablation_design/power_analysis.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.metrics.selection import _mean_std

B001 = ROOT / "runs" / "B001_reproduce"
OUT = ROOT / "runs" / "ablation_design"
KEYS = ("cd", "hd", "nuc")


def main() -> int:
    blob = json.loads((B001 / "metrics.json").read_text(encoding="utf-8"))
    recs = blob["records"]
    n = len(recs)
    tail = recs[n // 2:]                    # 平台区 ep50-99

    print("=" * 72)
    print("消融实验功效分析（噪声基准：B-001 平台区 ep50-99, n=%d）" % len(tail))
    print("=" * 72)

    noise = {}
    print(f"\n{'指标':<6} {'平台均值':>12} {'σ':>12} {'相对σ':>8} "
          f"{'MDE(2σ)':>9} {'MDE(3σ)':>9}")
    for k in KEYS:
        vals = [r["monitor_" + k] for r in tail]
        m, sd = _mean_std(vals)
        rel = sd / m * 100
        noise[k] = {"mean": m, "std": sd, "rel_std_pct": rel,
                    # 单次 run、单点比较时，要 2σ/3σ 才敢说有差异
                    "mde_2sigma_pct": 2 * rel, "mde_3sigma_pct": 3 * rel}
        print(f"{k:<6} {m:>12.6f} {sd:>12.6f} {rel:>7.2f}% "
              f"{2*rel:>8.2f}% {3*rel:>8.2f}%")

    print("\n  MDE = 单次 run 用平台区均值比较时，能可靠检出的最小相对变化。")
    print("  注：用平台区均值（n=50）而非单点比较时，均值标准误 = σ/√50，")
    print("      门槛显著下降 —— 这正是『平台区报数』定案的第二个好处。")

    for k in KEYS:
        se = noise[k]["std"] / math.sqrt(len(tail))
        noise[k]["stderr_of_mean"] = se
        noise[k]["mde_mean_2se_pct"] = 2 * se / noise[k]["mean"] * 100
    print(f"\n{'指标':<6} {'均值标准误':>12} {'MDE(2SE)':>10}  <- 平台区均值比较的真实门槛")
    for k in KEYS:
        print(f"{k:<6} {noise[k]['stderr_of_mean']:>12.8f} "
              f"{noise[k]['mde_mean_2se_pct']:>9.2f}%")

    # ---- 改进 A 的预期效应量 ----
    cal_path = OUT / "improve_a_calibration.json"
    a_pred = None
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        a1 = cal["candidates"]["A1_balance"]
        # A 组直接改变的是 loss 加权，其对 CD 指标的效应量上界可粗估为
        # 权重扰动幅度 × 该项贡献占比（这是乐观上界，实际通常更小）
        pert = abs(a1["w_cd_fwd"] - 1.0) * 100
        a_pred = {"weight_perturb_pct": pert,
                  "scale_drift_pct": a1["scale_drift_pct"],
                  "note": ("权重扰动 ±%.1f%%。注意：这是**输入端**扰动，"
                           "不等于输出端 CD 的变化量；后者通常更小，"
                           "因为优化过程会部分吸收权重变化。" % pert)}
        print("\n" + "-" * 72)
        print("改进 A（双向 CD 加权）预判")
        print("-" * 72)
        print(f"  权重扰动幅度      : ±{pert:.2f}%  (w_fwd={a1['w_cd_fwd']})")
        print(f"  CD 项尺度漂移     : {a1['scale_drift_pct']:+.2f}%")
        print(f"  平台区均值 MDE    : {noise['cd']['mde_mean_2se_pct']:.2f}% (2SE)")
        print(f"  单点比较 MDE      : {noise['cd']['mde_2sigma_pct']:.2f}% (2σ)")
        if pert < noise["cd"]["mde_2sigma_pct"]:
            print("  → ★若按单点比较，输入扰动已小于噪声门槛，几乎必然测不出")
        if pert > noise["cd"]["mde_mean_2se_pct"]:
            print(f"  → ✅ 但按平台区均值比较，门槛降到 "
                  f"{noise['cd']['mde_mean_2se_pct']:.2f}%，"
                  f"输入扰动 {pert:.1f}% 高于门槛，**有机会检出**")
            print("     前提：效应传导到输出端的衰减不超过 "
                  f"{pert / noise['cd']['mde_mean_2se_pct']:.1f} 倍")
        else:
            print("  → ★即使用平台区均值也低于门槛，建议改用多种子或放大权重")

    # ---- B/C/D：只给门槛，不编造预期 ----
    print("\n" + "-" * 72)
    print("改进 B / C / D 预判")
    print("-" * 72)
    print("  这三项是**开关型**改动（对抗项开关 / uniform 项开关 / scale_qk 开关），")
    print("  其效应量无解析表达式，跑前无法估计 —— 本脚本不编造数字。")
    print("  只给出判据：跑完后若平台区均值变化 <")
    print(f"    CD {noise['cd']['mde_mean_2se_pct']:.2f}% / "
          f"HD {noise['hd']['mde_mean_2se_pct']:.2f}% / "
          f"NUC {noise['nuc']['mde_mean_2se_pct']:.2f}%")
    print("  则判为『无显著差异』，不得在论文中声称有改进。")
    print("\n  ⚠️ 注意 NUC 的相对噪声达 %.2f%%，是三项里最吵的 ——"
          % noise["nuc"]["rel_std_pct"])
    print("     改进 C（uniform 项）恰好主打 NUC，其结论对噪声最敏感，")
    print("     必要时该组需要多种子重复。")

    # ---- 单种子够不够 ----
    print("\n" + "-" * 72)
    print("种子数建议")
    print("-" * 72)
    print("  本轮 8 组 × 150 epoch 单种子 ≈ 37 h（3090 串行）。")
    print("  若某组结论落在门槛附近（±1.5 倍 MDE 内），必须补种子，")
    print("  否则该组结论不可写入论文。届时再按需追加，不预先花 3 倍机时。")

    verdict = {
        "noise_baseline": noise,
        "improve_a_prediction": a_pred,
        "decision_rule": {
            "significant_if_plateau_mean_change_exceeds": {
                k: noise[k]["mde_mean_2se_pct"] for k in KEYS},
            "unit": "percent",
            "basis": "2 x standard error of plateau mean (n=%d)" % len(tail),
        },
        "honest_limits": [
            "改进 B/C/D 为开关型改动，效应量跑前无法估计，本分析不给预期值",
            "改进 A 的『权重扰动 ±7.9%』是输入端扰动，非输出端 CD 变化量",
            "所有门槛基于 B-001 单次 run 的 epoch 间噪声，"
            "未包含种子间噪声（后者通常更大）",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "power_analysis.json"
    dst.write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
