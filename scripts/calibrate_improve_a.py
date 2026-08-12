# -*- coding: utf-8 -*-
"""改进 A（双向 CD 加权）的权重取值定标 —— 用 B-001 实测数据定，不拍脑袋。

问题：A 组的 w_cd_fwd / w_cd_bwd 该取多少？

依据：B-001 满 100 epoch 实测 cd_bwd_share mean = 0.5446。
      即当前 fwd:bwd 的贡献比约为 0.4554 : 0.5446。

三种候选方案，本脚本把它们的实际效果算出来再选：
  A1「均衡化」：把两项的**加权贡献**拉平到 50:50 → 需提高 fwd 权重
  A2「强化覆盖」：顺着 bwd 已占优的方向再加强 → 反向放大
  A3「等和约束」：w_fwd + w_bwd = 2（保持 CD 总尺度与 baseline 同量级）

关键约束：w_fwd/w_bwd 不做归一化，会同时改变 CD 项总尺度。
若总尺度变了，lr 的有效步长也变了 → A 组与 baseline 的差异会混入
「等效学习率不同」这个混杂因子，无法归因到「加权本身」。
故必须施加等和约束，且本脚本要把尺度漂移量算出来核对。

输出：runs/ablation_design/improve_a_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

B001 = ROOT / "runs" / "B001_reproduce"
OUT = ROOT / "runs" / "ablation_design"


def main() -> int:
    hist = json.loads((B001 / "history.json").read_text(encoding="utf-8"))
    shares = [r["train_cd_bwd_share"] for r in hist
              if r.get("train_cd_bwd_share") is not None]
    fwds = [r["train_cd_fwd"] for r in hist if r.get("train_cd_fwd") is not None]
    bwds = [r["train_cd_bwd"] for r in hist if r.get("train_cd_bwd") is not None]
    n = len(shares)
    share_mean = sum(shares) / n
    # 用平台区（后 50 epoch）的绝对值定标，避免早期剧烈变化污染
    tail = slice(n // 2, n)
    fwd_m = sum(fwds[tail]) / len(fwds[tail])
    bwd_m = sum(bwds[tail]) / len(bwds[tail])

    print("=" * 68)
    print("改进 A 权重定标（数据源：B-001 history.json）")
    print("=" * 68)
    print(f"cd_bwd_share  全程均值 = {share_mean:.4f}  (n={n})")
    print(f"平台区(后 {len(fwds[tail])} ep) fwd 均值 = {fwd_m:.8f}")
    print(f"平台区(后 {len(bwds[tail])} ep) bwd 均值 = {bwd_m:.8f}")
    print(f"实际贡献比 fwd:bwd = {fwd_m/(fwd_m+bwd_m):.4f} : {bwd_m/(fwd_m+bwd_m):.4f}")
    base_cd = fwd_m + bwd_m
    print(f"baseline CD 平台均值 (w=1,1) = {base_cd:.8f}")

    cands = {}

    # --- A1 均衡化：令 w_f*fwd == w_b*bwd，且 w_f + w_b = 2 ---
    # w_f*fwd = w_b*bwd  且 w_f+w_b=2  =>  w_f = 2*bwd/(fwd+bwd)
    w_f = 2 * bwd_m / (fwd_m + bwd_m)
    w_b = 2 - w_f
    cands["A1_balance"] = (w_f, w_b,
                           "把两项加权贡献拉平到 50:50（提高 fwd 权重）")

    # --- A2 强化覆盖：顺 bwd 优势方向加强，等和约束 ---
    # 取与 A1 关于 1.0 对称的一组，形成"反向"对照
    cands["A2_boost_bwd"] = (2 - w_f, w_f,
                             "反向：进一步强化 bwd（与 A1 对称，作方向对照）")

    # --- A3 温和均衡：A1 的一半强度，等和约束 ---
    cands["A3_mild"] = (1 + (w_f - 1) * 0.5, 1 + (w_b - 1) * 0.5,
                        "A1 的一半强度（若 A1 过冲则用它）")

    print("\n候选方案（全部满足 w_fwd + w_bwd = 2 等和约束）")
    print(f"{'方案':<14} {'w_fwd':>7} {'w_bwd':>7} {'加权CD':>12} "
          f"{'尺度漂移':>9} {'新贡献比fwd':>11}")
    res = {}
    for name, (wf, wb, desc) in cands.items():
        new_cd = wf * fwd_m + wb * bwd_m
        drift = (new_cd - base_cd) / base_cd * 100
        new_share_f = (wf * fwd_m) / new_cd
        res[name] = {"w_cd_fwd": round(wf, 6), "w_cd_bwd": round(wb, 6),
                     "weighted_cd": new_cd, "scale_drift_pct": drift,
                     "new_fwd_share": new_share_f, "desc": desc}
        print(f"{name:<14} {wf:>7.4f} {wb:>7.4f} {new_cd:>12.8f} "
              f"{drift:>+8.2f}% {new_share_f:>11.4f}")

    print("\n[判读]")
    a1 = res["A1_balance"]
    print(f"  A1 尺度漂移 {a1['scale_drift_pct']:+.2f}%，新 fwd 贡献比 "
          f"{a1['new_fwd_share']:.4f}（目标 0.5）")
    if abs(a1["scale_drift_pct"]) < 2.0:
        print("  → 等和约束下尺度漂移 < 2%，可视为与 baseline 同量级，"
              "不引入等效学习率混杂因子")
    else:
        print("  → ★尺度漂移偏大，需在论文中报告并考虑同步调 lr")

    # 定案：A 组主用 A1，若结果异常再用 A3 复核
    verdict = {"chosen": "A1_balance",
               "reason": ("等和约束 w_fwd+w_bwd=2 保证 CD 总尺度不变，"
                          "同时把两项加权贡献拉平到 50:50，"
                          "直接检验『bwd 长期占优是否应被纠正』这一假设"),
               "fallback": "A3_mild（若 A1 过冲）",
               "control": "A2_boost_bwd（反方向对照，用于排除『任何加权都变好』）"}
    print(f"\n[定案] A 组采用 {verdict['chosen']}: "
          f"w_cd_fwd={res['A1_balance']['w_cd_fwd']} / "
          f"w_cd_bwd={res['A1_balance']['w_cd_bwd']}")
    print(f"       反向对照 A2 必须也跑 —— 否则无法排除『改了就变好』的伪相关")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "improve_a_calibration.json"
    dst.write_text(json.dumps(
        {"source": str(B001 / "history.json"),
         "n_epochs": n, "cd_bwd_share_mean": share_mean,
         "plateau_fwd_mean": fwd_m, "plateau_bwd_mean": bwd_m,
         "baseline_cd_plateau": base_cd,
         "candidates": res, "verdict": verdict},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
