# -*- coding: utf-8 -*-
"""baseline 与消融 run 的同口径对比。

用法: python scripts/compare_runs.py ABL_A1_cd_balance [其他run...]

纪律:
  - 只对比同一台机器上的 run; 机器不同直接拒绝(避免跨机数字并列)。
  - 差异按平台区标准差折算为"多少倍 SD", 并与 2SE 门槛比较, 不看百分比拍脑袋。
  - 2SE 由平台区样本量 n 计算: SE = SD / sqrt(n)。
"""
import io
import json
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "docs/_ch3_stats.json"


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/compare_runs.py <run_name> [...]")
        return 2
    b = load(BASE)
    print("baseline: %s  (%d epoch, %s)" % (b["source_run"], b["n_epochs"],
                                            b["env"]["gpu_name"]))
    rc = 0
    for name in sys.argv[1:]:
        rel = "docs/_stats_%s.json" % name
        if not os.path.isfile(os.path.join(ROOT, rel)):
            print("\n!! 缺少统计产物 %s, 先跑 python scripts/ch3_stats.py %s" % (rel, name))
            rc = 1
            continue
        a = load(rel)
        print("\n" + "=" * 74)
        print("对比 run: %s  (%d epoch, %s)" % (a["source_run"], a["n_epochs"],
                                             a["env"]["gpu_name"]))
        if a["env"]["gpu_name"] != b["env"]["gpu_name"]:
            print("!! 硬件不同(%s vs %s), 按纪律不并列比较数字"
                  % (a["env"]["gpu_name"], b["env"]["gpu_name"]))
            rc = 1
            continue
        # 配置差异
        ce_a, ce_b = a["config_echo"], b["config_echo"]
        keys = sorted(set(ce_a) | set(ce_b))
        diff = [(k, ce_b.get(k), ce_a.get(k)) for k in keys
                if k != "select_weights" and ce_a.get(k) != ce_b.get(k)]
        print("\n配置差异 (baseline -> %s):" % name)
        for k, vb, va in diff:
            print("  %-18s %-12s -> %s" % (k, vb, va))
        if not diff:
            print("  (无差异 — 注意: 若这是消融组, 说明配置未生效)")
            rc = 1
        # 双向硬检: 防"回显漏键"把真实消融误判成未生效, 也防拿两个 baseline 互比
        if not b.get("is_clean_baseline"):
            print("!! baseline 侧 is_clean_baseline=False, 不能充当对照基准")
            rc = 1
        if not a.get("ablation_keys"):
            print("!! %s 未检出任何消融键(ablation_keys 为空), 拒绝当作消融组解读" % name)
            rc = 1
        else:
            print("  消融键: %s" % a["ablation_keys"])

        # 平台区三指标
        print("\n平台区对比 (指标越低越好; SD/SE 取 baseline 平台区):")
        print("  警示: 此处 SD 为单 seed 平台区的 *时间* 波动, 非 run 间波动。")
        print("        2SE 由此得到会偏小, 易把噪声判成'过门槛'。")
        print("        最终结论须以多 seed run 间 SD 重算, 本表仅作趋势观察。")
        print("  %-5s %-13s %-13s %-9s %-9s %-8s %s"
              % ("指标", "baseline", name, "绝对差", "倍SD", "门槛2SE", "判定"))
        for k in ("cd", "hd", "nuc"):
            pb, pa = b["plateau"][k], a["plateau"][k]
            mb, ma = pb["plateau_mean"], pa["plateau_mean"]
            sd = pb["plateau_std"]
            n = b["plateau"]["plateau_n"]
            se = sd / math.sqrt(n)
            d = ma - mb
            nsd = d / sd if sd else float("nan")
            verdict = "改善(过门槛)" if d < -2 * se else \
                      ("变差(过门槛)" if d > 2 * se else "REJECT_NULL")
            print("  %-5s %-13.6f %-13.6f %+9.6f %+9.2f %8.6f %s"
                  % (k.upper(), mb, ma, d, nsd, 2 * se, verdict))

        # 收敛与选点
        print("\n多窗口收敛计数 (5 窗口):")
        for k in ("cd", "hd", "nuc"):
            print("  %-5s baseline %d/5   %s %d/5" % (
                k.upper(), b["multi_window_summary"][k]["n_converged"],
                name, a["multi_window_summary"][k]["n_converged"]))

        print("\n选点:")
        for tag, d in (("baseline", b), (name, a)):
            s = d["selection"]
            print("  %-22s 加权=%s  仅CD=%s  一致=%s"
                  % (tag, s["best_epoch_weighted"], s["best_epoch_cd_only"], s["agree"]))

        print("\n后向 Chamfer 占比 (训练集 / 监控):")
        for tag, d in (("baseline", b), (name, a)):
            t, m = d["train_cd_bwd_share"], d["monitor_loss_cd_bwd_share"]
            print("  %-22s train %.4f [%.4f,%.4f] n<0.5=%d | monitor %.4f n<0.5=%d"
                  % (tag, t["mean"], t["min"], t["max"], t["n_below_0.5"],
                     m["mean"], m["n_below_0.5"]))

        print("\n开销:")
        for tag, d in (("baseline", b), (name, a)):
            c = d["cost"]
            print("  %-22s %.1f s/ep  总 %.2f h  峰值显存 %.3f GB"
                  % (tag, c["sec_per_epoch_mean"], c["total_hours"], c["gpu_peak_gb"]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
