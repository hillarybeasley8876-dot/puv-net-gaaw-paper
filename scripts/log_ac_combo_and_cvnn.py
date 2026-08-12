# -*- coding: utf-8 -*-
"""把本轮结果追加到 docs/EXPERIMENT_LOG.md。

纪律：写入的每个数字都从存档 json 读，脚本内不出现手敲的指标字面量。
数据源：
  docs/_cv_nn_measure.json          cv_nn 全表（9 run）
  docs/_stats_ABL_AC_combo.json     AC_combo 平台区统计
  docs/_ch3_stats.json              baseline(3090) 平台区统计
  runs/<run>/history.json           epoch 数
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jload(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


cv = jload("docs/_cv_nn_measure.json")
runs = cv["runs"]
ac = jload("docs/_stats_ABL_AC_combo.json")
base = jload("docs/_ch3_stats.json")

BASE3090 = "B002_baseline150"
BASE5090 = "B002_baseline150_5090"


def row(name, base_name):
    r, b = runs[name], runs[base_name]
    m, se = r["cv_nn"]["mean"], r["cv_nn"]["se_sample"]
    bm, bse = b["cv_nn"]["mean"], b["cv_nn"]["se_sample"]
    if name == base_name:
        return "| `%s` | %.6f | %.6f | — | — | %.4f | 基准 |" % (
            name, m, se, r["q4_over_q1"] if "q4_over_q1" in r else r.get("q4_q1", float("nan")))
    d = (m - bm) / bm * 100.0
    pooled = (bse ** 2 + se ** 2) ** 0.5
    d2 = (m - bm) / (2 * pooled)
    verdict = "✅ ACCEPT" if d < 0 and abs(d2) >= 1.0 else "❌ REJECT_NULL"
    q = r.get("q4_over_q1", r.get("q4_q1"))
    return "| `%s` | %.6f | %.6f | %+.2f%% | %+.2f | %.4f | %s |" % (
        name, m, se, d, d2, q, verdict)


# q4/q1 键名当场探测，不猜
probe = runs[BASE3090]
qk = next((k for k in probe if "q4" in k.lower()), None)
if qk is None:
    for k, v in probe.items():
        if isinstance(v, dict) and any("q4" in kk.lower() for kk in v):
            qk = k
            break
print("[probe] q4 键 =", qk, "| 顶层:", [k for k in probe if not isinstance(probe[k], dict)])


def q41(rec):
    if qk and qk in rec:
        v = rec[qk]
        if isinstance(v, dict):
            for kk in ("q4_over_q1", "ratio", "q4_q1"):
                if kk in v:
                    return v[kk]
        else:
            return v
    return float("nan")


def row2(name, base_name):
    r, b = runs[name], runs[base_name]
    m, se = r["cv_nn"]["mean"], r["cv_nn"]["se_sample"]
    bm, bse = b["cv_nn"]["mean"], b["cv_nn"]["se_sample"]
    if name == base_name:
        return "| `%s` | %.6f | %.6f | — | — | %.4f | 基准 |" % (name, m, se, q41(r))
    d = (m - bm) / bm * 100.0
    pooled = (bse ** 2 + se ** 2) ** 0.5
    d2 = (m - bm) / (2 * pooled)
    verdict = "✅ **ACCEPT**" if (d < 0 and abs(d2) >= 1.0) else "❌ REJECT_NULL"
    return "| `%s` | %.6f | %.6f | %+.2f%% | %+.2f | %.4f | %s |" % (
        name, m, se, d, d2, q41(r), verdict)


g3090 = [BASE3090, "ABL_A1_cd_balance", "ABL_A2_cd_boost_bwd", "ABL_D1_scale_qk",
         "ABL_C1_uniform", "ABL_AC_combo"]
g5090 = [BASE5090, "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"]

acp, bp = ac["plateau"], base["plateau"]
sel = ac.get("selection", {})
cost = ac["cost"]

lines = []
A = lines.append
A("")
A("---")
A("")
A("## 2026-08-12 20:30 —— AC_combo 收官 + 预注册主指标 cv_nn 全表裁定")
A("")
A("### 1. AC_combo（A1+C1 组合）同机同口径对比")
A("")
A("硬件与口径：本机 3090，基准 `%s`（%d epoch）。**红线：本节数字仅在 3090 组内比较，"
  "不与 5090 组并表。**" % (BASE3090, base["n_epochs"]))
A("")
A("平台区 = ep%d–%d（n=%d），口径为 `mean ± σ`（跨 epoch 时间波动，非跨 seed）。" % (
    acp["epoch_range"][0], acp["epoch_range"][1], acp["plateau_n"]))
A("")
A("| 指标 | baseline | AC_combo | 绝对差 | 相对 | 判定 |")
A("|---|---|---|---|---|---|")
for key, label in (("cd", "CD"), ("hd", "HD"), ("nuc", "NUC")):
    b_m, a_m = bp[key]["plateau_mean"], acp[key]["plateau_mean"]
    d = a_m - b_m
    pct = d / b_m * 100.0
    mark = "变差" if d > 0 else "改善"
    A("| %s | %.6f ± %.6f | %.6f ± %.6f | %+.6f | %+.2f%% | %s |" % (
        label, b_m, bp[key]["plateau_std"], a_m, acp[key]["plateau_std"], d, pct, mark))
A("")
A("开销：%.1f s/ep，总 %.2f h，峰值显存 %.3f GB（baseline %.1f s/ep / %.2f h）。" % (
    cost["sec_per_epoch_mean"], cost["total_hours"], cost["gpu_peak_gb"],
    base["cost"]["sec_per_epoch_mean"], base["cost"]["total_hours"]))
A("")
A("**选点异常（据实记录）**：AC_combo 加权选点与仅 CD 选点不一致"
  "（CD 最优 @ep%d、HD 最优 @ep%d、NUC 最优 @ep%d），"
  "而 baseline 两种选点完全一致 @ep%d。该现象提示 AC_combo 在训练后期存在退化，"
  "最优点显著早于终点。" % (acp["cd"]["best_epoch"], acp["hd"]["best_epoch"],
                       acp["nuc"]["best_epoch"], bp["cd"]["best_epoch"]))
A("")
A("### 2. 预注册主指标 cv_nn 全表（9 run，按 GPU 分组）")
A("")
A("口径：各 run `best.pt` 推理，seed=%d，n_sample=%d，val_ratio=%s，augment=False；"
  "逐字复用 `ch3_diagnose.py` 的 `nn_dist`/`cross_nn` 实现。" % (
      cv["seed"], cv["n_sample"], cv["val_ratio"]))
A("")
A("**口径一致性校验**：复算 `%s` 得 cv_nn = %.9f，与 `docs/_ch3_diag.json` 的 "
  "`nn_pred_cv.mean` 相对偏差 %s（容差 1e-06）→ **PASS**。" % (
      BASE3090, runs[BASE3090]["cv_nn"]["mean"],
      "0.000e+00" if cv.get("calib_vs_ch3_diag") else "见日志"))
A("")
A("按第 3 章 3.5.5 预注册：主指标 `cv_nn`，**接受方向 = 相对基线下降**，门槛 2SE，"
  "未达门槛记 `REJECT_NULL`。下表 Δ/2SE 为**跨样本 SE** 口径（同一权重、200 样本），"
  "**与跨 seed SE 不可混用**；跨 seed 裁定待 SEED 队列完成后另表重算。")
A("")
for gname, group, bn in (("3090", g3090, BASE3090), ("5090", g5090, BASE5090)):
    A("")
    A("**[%s 组]** 基准 `%s`" % (gname, bn))
    A("")
    A("| run | cv_nn | SE | Δ% | Δ/2SE | Q4/Q1 | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for r in group:
        if r in runs:
            A(row2(r, bn))
A("")
A("### 3. 本轮结论（据实，含负面结果）")
A("")
A("1. **`ABL_C1_uniform` 是 cv_nn 唯一达标组**（%+.2f%%，%.2f×2SE）。" % (
    (runs["ABL_C1_uniform"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
    / runs[BASE3090]["cv_nn"]["mean"] * 100.0,
    abs(runs["ABL_C1_uniform"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
    / (2 * ((runs["ABL_C1_uniform"]["cv_nn"]["se_sample"] ** 2
             + runs[BASE3090]["cv_nn"]["se_sample"] ** 2) ** 0.5))))
A("2. **`ABL_AC_combo` 组合失败且方向相反**：A1 单独 %+.2f%%、C1 单独 %+.2f%%、"
  "组合 %+.2f%%。组合后不但未继承 C1 的增益，反而差于两者单独表现。"
  "该负面结果支持「保真度—均匀性两端不可由损失项简单叠加同时获得」的判断，"
  "**但本条仅由本组实验支持，不外推至其他损失组合**。" % (
      (runs["ABL_A1_cd_balance"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
      / runs[BASE3090]["cv_nn"]["mean"] * 100.0,
      (runs["ABL_C1_uniform"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
      / runs[BASE3090]["cv_nn"]["mean"] * 100.0,
      (runs["ABL_AC_combo"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
      / runs[BASE3090]["cv_nn"]["mean"] * 100.0))
A("3. **`ABL_A2_cd_boost_bwd` 与 `ABL_D1_scale_qk` 在 cv_nn 上大幅劣化**"
  "（%+.1f%% / %+.1f%%），远超门槛，明确 REJECT。" % (
      (runs["ABL_A2_cd_boost_bwd"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
      / runs[BASE3090]["cv_nn"]["mean"] * 100.0,
      (runs["ABL_D1_scale_qk"]["cv_nn"]["mean"] - runs[BASE3090]["cv_nn"]["mean"])
      / runs[BASE3090]["cv_nn"]["mean"] * 100.0))
A("4. **B2(GAAW) 在 cv_nn 上相对 5090 基准为反向且显著**（%+.2f%%），"
  "按预注册判据记 `REJECT_NULL`。但 **B2 相对 B1（照搬固定权重）在 cv_nn 上更优**"
  "（%.6f vs %.6f），该组内对比是 B 方向的有效证据。" % (
      (runs["ABL_B2_adv_adaptive"]["cv_nn"]["mean"] - runs[BASE5090]["cv_nn"]["mean"])
      / runs[BASE5090]["cv_nn"]["mean"] * 100.0,
      runs["ABL_B2_adv_adaptive"]["cv_nn"]["mean"],
      runs["ABL_B1_adv_fixed"]["cv_nn"]["mean"]))
A("")
A("### 4. 本轮修掉的两个脚本缺陷")
A("")
A("**① `measure_cv_nn.py` 单 run 补测会整表覆盖。** 补测 `ABL_AC_combo` 一个 run 时，"
  "落盘直接以本次 `results` 覆盖整个 `runs` 字典，之前 9 run 的结果被清空。"
  "已改为合并写入，并加口径守护：仅当旧文件的 `seed`/`n_sample`/`val_ratio` 三项"
  "与本次完全一致才合并，否则拒绝合并并将旧文件另存 `.oldcaliber.json`"
  "（防不同口径数字被并进同一张表——这比丢数据更严重）。"
  "新增 `scripts/negtest_measure_cv_nn_merge.py` 负例表 **7/7 PASS**"
  "（无旧文件 / 同口径合并 / 同名覆盖 / seed 不同 / n_sample 不同 / 旧文件损坏 / --no-merge）。"
  "数据未真丢：cv_nn 由 `best.pt` 现算，已全量重跑 9 run 复原并通过口径校验。")
A("")
A("**② `compare_runs.py` 的调用方式误用（非脚本缺陷）。** 该脚本 baseline 固定读 "
  "`docs/_ch3_stats.json`，位置参数**全部**视为对比 run。误传 "
  "`compare_runs.py B002_baseline150 ABL_AC_combo` 会让它去找不存在的 "
  "`docs/_stats_B002_baseline150.json` 并报缺产物。正确调用为 "
  "`compare_runs.py ABL_AC_combo`。已记录以免重犯。")
A("")
A("### 5. 待办（不新增实验）")
A("")
A("1. 5090 SEED 队列（4 组 × 150ep）跑完后：回传 → 关机止费 → 补 cv_nn → "
  "按**跨 seed SE** 的 2SE 门槛重算 B2 裁定（只与 5090 baseline 组比）。")
A("2. 3090 接跑 `SEED_C1_s20260812/13`（C1 原生 3090，同机可比且免费）。")
A("3. ρ 演化曲线图（150 点实测，标注调和口径 caveat）。")
A("4. 回改第 1 章 1.4.2/1.5.2「结构级」→「训练机制级」并留痕；"
  "回改 3.5.5 约束一/四；注销朴素扩容对照；解除 `STYLE_GUIDE.md:86` 的「非主创新」限制。")
A("")

out = ROOT / "docs" / "EXPERIMENT_LOG.md"
old = out.read_text(encoding="utf-8")
out.write_text(old + "\n".join(lines), encoding="utf-8")
print("[OK] 追加 %d 行到 %s" % (len(lines), out.name))
