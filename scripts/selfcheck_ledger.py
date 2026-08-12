"""证据台账一致性自检。

目的：EVIDENCE_LEDGER.md 是本项目的诚信红线，里面每个数字必须能对上落盘产物。
本脚本把台账里的关键数字与真实 JSON 逐个比对，防止手写文档时抄错/抄旧。

用法：python scripts/selfcheck_ledger.py
"""
import io
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(BASE, "docs", "EVIDENCE_LEDGER.md")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def load(rel):
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


with io.open(LEDGER, encoding="utf-8") as f:
    text = f.read()

# ---------- 1. 结构性检查 ----------
for sec in ["## 1.", "### 1.7", "### 1.8", "### 1.9", "### 2.1", "### 2.2",
            "### 2.3", "### 2.4", "## 3.", "## 4.", "## 5."]:
    check("章节存在 %s" % sec, sec in text)

# 过期表述必须已清除
stale = [
    "最大空洞：没有任何真实数据集\n",
    "## 2.2 所有性能数字：全部为空",
    "没有数据 = 没有一个实验数字。这是当前唯一的硬阻塞。",
    "最后更新：2026-08-10",
]
for s in stale:
    check("过期表述已清除: %r" % s[:28], s not in text)

# ---------- 2. B-001 数字对账 ----------
rep = load("runs/B001_reproduce/selection_replay.json")
check("selection_replay.json 存在", rep is not None)
if rep:
    pl = rep["plateau"]
    pairs = [
        ("cd 平台均值", "0.002166", pl["cd"]["plateau_mean"], 1e-6),
        ("cd 平台σ", "0.000051", pl["cd"]["plateau_std"], 1e-6),
        ("hd 平台均值", "0.008091", pl["hd"]["plateau_mean"], 1e-6),
        ("hd 平台σ", "0.000623", pl["hd"]["plateau_std"], 1e-6),
        ("nuc 平台均值", "0.568401", pl["nuc"]["plateau_mean"], 1e-6),
        ("nuc 平台σ", "0.018593", pl["nuc"]["plateau_std"], 1e-6),
        ("hd 平台最优", "0.007019", pl["hd"]["best"], 1e-6),
        ("nuc 平台最优", "0.539498", pl["nuc"]["best"], 1e-6),
    ]
    for label, written, actual, tol in pairs:
        in_doc = written in text
        matches = abs(float(written) - actual) < tol
        check("台账数字对账 %s (%s)" % (label, written), in_doc and matches,
              "in_doc=%s actual=%.9f" % (in_doc, actual))

    # 选点一致性：反转 1 必须如实记录为"一致"
    same = (rep["cd_only_pick"]["epoch"] == rep["composite_pick"]["epoch"])
    check("B-001 两准则选点确实一致(反转1)", same,
          "cd_only=%s composite=%s" % (rep["cd_only_pick"]["epoch"],
                                       rep["composite_pick"]["epoch"]))
    check("台账写明 ep041 且标注逐位一致",
          "ep041" in text and "逐位一致" in text)

# ---------- 3. 显著性门槛对账 ----------
pw = load("runs/ablation_design/power_analysis.json")
check("power_analysis.json 存在", pw is not None)
if pw:
    rule = pw["decision_rule"]["significant_if_plateau_mean_change_exceeds"]
    for k, written in [("cd", "0.66%"), ("hd", "2.18%"), ("nuc", "0.93%")]:
        in_doc = written in text
        matches = abs(round(rule[k], 2) - float(written.rstrip("%"))) < 0.005
        check("门槛对账 %s=%s" % (k, written), in_doc and matches,
              "actual=%.4f" % rule[k])
    # honest_limits 必须在台账里体现
    check("台账保留 honest_limits(种子间噪声)",
          "种子间噪声" in text and "honest_limits" in text)

# ---------- 4. 改进 A 标定值对账 ----------
cal = load("runs/ablation_design/improve_a_calibration.json")
check("improve_a_calibration.json 存在", cal is not None)
if cal:
    blob = json.dumps(cal, ensure_ascii=False)
    for v in ["1.078971", "0.921029"]:
        check("A1 标定值 %s 同时在存档与台账" % v,
              (v in blob) and (v in text))

# ---------- 5. 收敛敏感性对账 ----------
cs = load("runs/B001_reproduce/convergence_sensitivity.json")
check("convergence_sensitivity.json 存在", cs is not None)
if cs:
    blob = json.dumps(cs, ensure_ascii=False)
    check("台账写明 HD 稳定判未收敛(唯一稳健)",
          "HD 稳定判 F" in text or "只有 HD 稳定判 F" in text,
          "台账须如实记录反转2")
    check("台账写明 CD 判定翻转 T,F,T,T,F", "T,F,T,T,F" in text)
    check("台账写明 NUC 判定翻转 T,T,F,F,F", "T,T,F,F,F" in text)

# ---------- 6. 数据资产字节数对账 ----------
for rel, written in [("data/raw/PU1K.zip", "972,264,385"),
                     ("data/raw/PUGAN_poisson_256_poisson_1024.h5", "339,607,493")]:
    p = os.path.join(BASE, rel)
    if os.path.exists(p):
        real = os.path.getsize(p)
        check("数据字节数对账 %s" % os.path.basename(rel),
              (written in text) and (int(written.replace(",", "")) == real),
              "real=%d" % real)
    else:
        # 允许放在别处，但台账里写了数字就必须能找到文件
        check("数据文件可定位 %s" % os.path.basename(rel), False,
              "台账写了 %s 但 %s 不存在" % (written, rel))

# ---------- 7. 口径声明必须存在（最重要的诚信项） ----------
check("台账声明 monitor_* 非论文数字",
      "非论文主表数字" in text or "不是论文主表数字" in text)
check("台账声明论文唯一出口 eval_cd_hd_official",
      "eval_cd_hd_official" in text)
check("台账保留 M1 不足以支撑顶刊的诚实边界",
      "不足以支撑顶刊" in text)
check("台账保留 B1 不可省的归因口径",
      "B1 不可省" in text)
check("台账指出尚无论文口径数字落盘",
      "尚未有任何一个论文主表口径" in text)

# ---------- 8. 禁止出现占位/编造痕迹 ----------
for bad in ["约提升", "预计提升", "大幅优于", "XX%"]:
    check("无编造/占位表述: %s" % bad, bad not in text)

# ---------- 汇总 ----------
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print("=" * 66)
for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    line = "[%s] %s" % (tag, name)
    if detail and not ok:
        line += "  <-- " + detail
    print(line)
print("=" * 66)
print("ledger selfcheck: %d PASS / %d FAIL" % (n_pass, n_fail))
print("ledger bytes: %d" % len(text.encode("utf-8")))
sys.exit(1 if n_fail else 0)
