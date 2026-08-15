# -*- coding: utf-8 -*-
"""
第 6.6.1 节「训练时间、显存和额外反向传播开销」的数据生成。

为什么需要这一节：GAAW 每步需对两个损失分支分别反传以取梯度范数，
这是**真实的额外计算开销**。若只报指标改善而不报代价，读者无法判断
该方法是否值得采用。本脚本从 history.json 的 `sec` 与 `gpu_peak_gb`
字段直接统计，不做任何估算。

口径（跑前定死，写死在本文件，不从结果反推）：
  1. 时间口径取**中位数**而非均值：训练首个 epoch 含 CUDA 上下文初始化与
     cuDNN autotune，耗时系统性偏高（实测 B2 首 epoch 164.51s vs 中位数约
     130s），均值会被首 epoch 拉偏。中位数对该类单点异常不敏感。
  2. 显存取全程**最大值**（峰值即上界，取中位数无意义）。
  3. 跨机器不并列：3090 组与 5090 组分表报告。同机器内才计算相对开销，
     因为不同 GPU 的绝对耗时不可比（这正是跨机器红线的适用场景）。
  4. 额外开销的归因边界：B2 vs B1 的时间差同时包含
     (a) GAAW 的两次反传、(b) 两者 best epoch 不同导致的 checkpoint I/O 差异。
     本脚本只报总时间差，不声称全部归因于 (a)。

产物：docs/_train_cost.json
"""
import json, io, os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "3090": {
        "gpu_label": "NVIDIA GeForce RTX 3090",
        "runs": ["B002_baseline150", "ABL_A1_cd_balance", "ABL_A2_cd_boost_bwd",
                 "ABL_C1_uniform", "ABL_AC_combo", "ABL_D1_scale_qk"],
        "baseline": "B002_baseline150",
    },
    "5090": {
        "gpu_label": "NVIDIA GeForce RTX 5090",
        "runs": ["B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"],
        "baseline": "B002_baseline150_5090",
    },
}

# 判别器参数量（第 5 章表 5.2 已报，此处复用同一数字）
DISC_PARAMS = 255426
GEN_PARAMS = 1152803


def load(run):
    h = json.load(io.open(os.path.join(ROOT, "runs", run, "history.json"), encoding="utf-8"))
    e = json.load(io.open(os.path.join(ROOT, "runs", run, "env.json"), encoding="utf-8"))
    secs = [r["sec"] for r in h if r.get("sec") is not None]
    mems = [r["gpu_peak_gb"] for r in h if r.get("gpu_peak_gb") is not None]
    return {
        "n_epoch": len(h),
        "sec_median": st.median(secs),
        "sec_mean": st.mean(secs),
        "sec_first": secs[0],
        "sec_min": min(secs),
        "sec_max": max(secs),
        "sec_total": sum(secs),
        "gpu_peak_gb": max(mems),
        "gpu_name": e.get("gpu_name"),
        "torch": e.get("torch"),
        "cuda": e.get("cuda_version"),
    }


out = {
    "schema": "puvnet.train_cost/v1",
    "purpose": "第 6.6.1 节训练代价数据。时间取 epoch 中位数（避免首 epoch 初始化开销污染），显存取全程峰值。",
    "caliber": {
        "time_statistic": "median over per-epoch `sec` field",
        "time_rationale": "首 epoch 含 CUDA 上下文初始化与 cuDNN autotune，系统性偏高；中位数对该单点异常不敏感",
        "mem_statistic": "max over per-epoch `gpu_peak_gb` field",
        "cross_host_rule": "3090 组与 5090 组分表；相对开销只在同机器内计算",
        "attribution_caveat": "B2 vs B1 的时间差同时包含 GAAW 两次反传与 best epoch 不同带来的 I/O 差异，本文只报总差不作单一归因",
    },
    "params": {"generator": GEN_PARAMS, "discriminator": DISC_PARAMS,
               "note": "判别器仅训练期使用，推理期不参与计算，故推理开销与基线一致"},
    "groups": {},
}

for gname, g in GROUPS.items():
    base = load(g["baseline"])
    runs = {}
    for r in g["runs"]:
        d = load(r)
        d["sec_median_rel_pct"] = (d["sec_median"] - base["sec_median"]) / base["sec_median"] * 100
        d["gpu_peak_rel_pct"] = (d["gpu_peak_gb"] - base["gpu_peak_gb"]) / base["gpu_peak_gb"] * 100
        d["is_baseline"] = (r == g["baseline"])
        runs[r] = d
    out["groups"][gname] = {
        "gpu_label": g["gpu_label"], "baseline": g["baseline"], "runs": runs,
    }

p = os.path.join(ROOT, "docs", "_train_cost.json")
json.dump(out, io.open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote", p)
print()

for gname, g in out["groups"].items():
    print("=" * 78)
    print(f"【{gname}】 {g['gpu_label']}   baseline={g['baseline']}")
    print("=" * 78)
    print(f"  {'run':26s} {'ep':>4s} {'中位s':>8s} {'相对%':>8s} {'总时h':>7s} {'峰值GB':>7s} {'相对%':>7s}")
    for r, d in g["runs"].items():
        print(f"  {r:26s} {d['n_epoch']:>4d} {d['sec_median']:>8.2f} "
              f"{d['sec_median_rel_pct']:>+7.2f}% {d['sec_total']/3600:>7.2f} "
              f"{d['gpu_peak_gb']:>7.3f} {d['gpu_peak_rel_pct']:>+6.2f}%")
    print(f"  环境: torch={list(g['runs'].values())[0]['torch']}  "
          f"cuda={list(g['runs'].values())[0]['cuda']}")
    print()

# 主对比的额外开销
b1 = out["groups"]["5090"]["runs"]["ABL_B1_adv_fixed"]
b2 = out["groups"]["5090"]["runs"]["ABL_B2_adv_adaptive"]
b0 = out["groups"]["5090"]["runs"]["B002_baseline150_5090"]
print("=" * 78)
print("主对比的训练代价（5090 同机）")
print("=" * 78)
print(f"  B0 无对抗   : {b0['sec_median']:7.2f} s/epoch   峰值 {b0['gpu_peak_gb']:.3f} GB")
print(f"  B1 固定权重 : {b1['sec_median']:7.2f} s/epoch ({b1['sec_median_rel_pct']:+.2f}% vs B0)   峰值 {b1['gpu_peak_gb']:.3f} GB ({b1['gpu_peak_rel_pct']:+.2f}%)")
print(f"  B2 GAAW     : {b2['sec_median']:7.2f} s/epoch ({b2['sec_median_rel_pct']:+.2f}% vs B0)   峰值 {b2['gpu_peak_gb']:.3f} GB ({b2['gpu_peak_rel_pct']:+.2f}%)")
d_pct = (b2['sec_median'] - b1['sec_median']) / b1['sec_median'] * 100
d_mem = (b2['gpu_peak_gb'] - b1['gpu_peak_gb']) / b1['gpu_peak_gb'] * 100
print(f"  → GAAW 相对固定权重的额外开销: 时间 {d_pct:+.2f}%   显存 {d_mem:+.2f}%")
print(f"  → 判别器参数 {DISC_PARAMS:,}（占生成器 {DISC_PARAMS/GEN_PARAMS*100:.1f}%），仅训练期使用")
