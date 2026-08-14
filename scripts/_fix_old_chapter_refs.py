# -*- coding: utf-8 -*-
"""
拆七章后的旧章号修正：把指向"实验/消融/主表"的「第 5 章」改为「第 6 章」。

背景：拆章前实验章是第 5 章，拆章后变为第 6 章（第 5 章改为 GAAW 机制章）。
第 1、2、3 章中写于拆章前的交叉引用仍指向旧编号。

安全设计（防误改）：
  1. 只改**逐条列出的精确字符串**，不做正则批量替换——正则会误伤指向新第 5 章
     （机制章）的正确引用，例如第 1.6 节的章节导言。
  2. 每条替换前后打印，且校验替换后文件中该旧串数量归零。
  3. 若某条旧串未找到，报 MISS 而不静默跳过（可能已被手工改过或文本有差异）。
"""
import io, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")

# (文件, 旧串, 新串)  —— 每条都经人工确认语境指向实验章
EDITS = [
    # 第 2 章
    ("ch2_related_work.md",
     "但不纳入第 5 章的主线对",
     "但不纳入第 6 章的主线对"),
    ("ch2_related_work.md",
     "为第 3.4.3 节的协议对齐与第 5 章的多指标同报提供依据。",
     "为第 3.4.3 节的协议对齐与第 6 章的多指标同报提供依据。"),
    # 第 3 章
    ("ch3_baseline.md",
     "相应地也不在第 5 章与任意倍率方法并列对比。",
     "相应地也不在第 6 章与任意倍率方法并列对比。"),
    ("ch3_baseline.md",
     "其作用是在第 5 章中量化两类局部证据各自的必要性。",
     "其作用是在第 6 章中量化两类局部证据各自的必要性。"),
    ("ch3_baseline.md",
     "并在第 5 章作为独立消融项测量其影响。",
     "并在第 6 章作为独立消融项测量其影响。"),
    ("ch3_baseline.md",
     "**官方评价口径。** 第 5 章主表数字",
     "**官方评价口径。** 第 6 章主表数字"),
    ("ch3_baseline.md",
     "这为第 5 章的多指标同报提供了本项目内的实证依据",
     "这为第 6 章的多指标同报提供了本项目内的实证依据"),
    ("ch3_baseline.md",
     "由第 4 章的设计与第 5 章的消融共同完成",
     "由第 4 章的设计与第 6 章的消融共同完成"),
    ("ch3_baseline.md",
     "需由第 4 章的结构改进配合第 5 章的受控消融判定",
     "需由第 4 章的研究设计配合第 6 章的受控消融判定"),
    ("ch3_baseline.md",
     "且在第 5 章中与\"同等参数量的朴素扩容\"作对照",
     "且在第 6 章中与\"同等参数量的朴素扩容\"作对照"),
    ("ch3_baseline.md",
     "实测中该方向为不利（第 5 章据实报告）",
     "实测中该方向为不利（第 6 章据实报告）"),
    ("ch3_baseline.md",
     "故第 4 章与第 5 章中一切以 `ABL_B1_adv_fixed` 为对照的结论",
     "故第 4 章与第 6 章中一切以 `ABL_B1_adv_fixed` 为对照的结论"),
]

print("=" * 74)
print("旧章号修正（第 5 章 -> 第 6 章，仅实验语境）")
print("=" * 74)

by_file = {}
for f, old, new in EDITS:
    by_file.setdefault(f, []).append((old, new))

fixed = missed = 0
for f, pairs in by_file.items():
    p = os.path.join(CH, f)
    if not os.path.exists(p):
        print(f"  !! 文件不存在: {f}")
        sys.exit(1)
    t = io.open(p, encoding="utf-8").read()
    orig = t
    for old, new in pairs:
        if old in t:
            n = t.count(old)
            t = t.replace(old, new)
            print(f"  FIX  {f}  x{n}  {old[:46]}")
            fixed += n
        else:
            print(f"  MISS {f}        {old[:46]}")
            missed += 1
    if t != orig:
        io.open(p, "w", encoding="utf-8").write(t)

print()
print(f"  已修 {fixed} 处，未命中 {missed} 处")

# 校验：三章中不应再有"第 N 章"指向实验语境的旧引用
print()
print("=" * 74)
print("校验：残留检查")
print("=" * 74)
import re
KEY = r"(消融|主表|多指标同报|实验|据实报告|受控消融)"
left = 0
for f in ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md"]:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    for m in re.finditer(r"第\s*5\s*章", t):
        s = max(0, m.start() - 45); e = min(len(t), m.end() + 45)
        ctx = t[s:e].replace("\n", " ")
        if re.search(KEY, ctx):
            line = t[:m.start()].count("\n") + 1
            print(f"  ?  {f}:{line}  …{ctx.strip()[:96]}…")
            left += 1
if left == 0:
    print("  OK  三章中不再有指向实验语境的「第 5 章」")
else:
    print(f"  !! 仍有 {left} 处待人工确认（可能是指向新第 5 章机制章的正确引用）")
