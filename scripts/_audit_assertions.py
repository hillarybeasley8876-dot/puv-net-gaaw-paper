# -*- coding: utf-8 -*-
"""
断言体检（七章）：找出"既无 cite 占位符、又无「本文实测」标记"的陈述句。

依据 docs/STYLE_GUIDE.md §2.9 断言纪律：
  A 类 文献结论 -> 必须挂 {{cite:KEY}}
  B 类 本文实测 -> 必须标 run / 口径 / 存档
  C 类 本文推断 -> 必须紧邻依据，且不得写成既成事实

本脚本抓的是**疑似 A 类却未挂 cite** 的句子：
含"外部知识断言"信号词，但整句无 cite 占位符、也无本文实测标记。

信号词分两类：
  强信号（几乎必然需要 cite）：文献/研究表明/已有工作/该方法/原文/被提出/广泛
  弱信号（视语境）：通常/一般/常见/标准做法/惯例/领域

本文实测标记（有其一即视为 B 类，免 cite）：
  本文/实测/第N章/第N.M节/run 名/表 N-M/图 N-M/存档路径/docs\\_*.json

输出按章分组，供人工逐条判断。不自动改稿。
"""
import os, io, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")
FILES = ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md",
         "ch4_design.md", "ch5_mechanism.md", "ch6_experiments.md", "ch7_conclusion.md"]

STRONG = r"(文献|研究表明|已有工作|该方法|原文|被提出|广泛|学界|前人|既有方法|经典|奠基)"
WEAK = r"(通常|一般而言|常见的|标准做法|惯例|领域内|普遍的做法)"
# B 类标记：本文自己的证据
BMARK = r"(本文|本章|本节|实测|第\s*\d+\s*[章节]|第\s*\d+\.\d+|表\s*\d+-\d+|图\s*\d+-\d+|"\
        r"runs/|docs/_|B002_|ABL_|B-001|`[a-z_]+\.py`|epoch|GAAW|算法\s*\d+-\d+)"
CITE = r"\{\{cite:"

def sentences(text):
    """按中文句末标点切句，保留行号。"""
    out = []
    for li, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if not s or s.startswith(("#", ">", "|", "```", "$$")):
            continue
        # 表格行、公式行跳过
        for seg in re.split(r"(?<=[。；！？])", s):
            seg = seg.strip()
            if len(seg) >= 18:
                out.append((li, seg))
    return out

print("=" * 78)
print("断言体检：疑似 A 类（文献结论）却未挂 cite 的句子")
print("=" * 78)

report = defaultdict(list)
stat = {}
for f in FILES:
    p = os.path.join(CH, f)
    t = io.open(p, encoding="utf-8").read()
    sents = sentences(t)
    n_total = len(sents)
    n_hit = 0
    for li, s in sents:
        if re.search(CITE, s):
            continue                    # 已挂 cite
        if re.search(BMARK, s):
            continue                    # 属 B 类（本文证据）
        strong = re.search(STRONG, s)
        weak = re.search(WEAK, s)
        if strong or weak:
            lvl = "强" if strong else "弱"
            report[f].append((li, lvl, s))
            n_hit += 1
    stat[f] = (n_total, n_hit)

for f in FILES:
    n_total, n_hit = stat[f]
    print()
    print(f"--- {f}   受检句 {n_total}，疑似 {n_hit} ---")
    for li, lvl, s in report[f][:12]:
        print(f"  [{lvl}] L{li}: {s[:104]}")
    if len(report[f]) > 12:
        print(f"  …（另有 {len(report[f])-12} 条）")

print()
print("=" * 78)
print("汇总")
print("=" * 78)
tot_s = sum(v[0] for v in stat.values())
tot_h = sum(v[1] for v in stat.values())
print(f"  受检句总数 {tot_s}，疑似未挂 cite 的断言 {tot_h} 条")
print(f"  {'file':26s} {'受检句':>7s} {'疑似':>6s} {'占比':>7s}")
for f in FILES:
    n, h = stat[f]
    print(f"  {f:26s} {n:>7d} {h:>6d} {h/max(n,1)*100:>6.1f}%")
if tot_s < 300:
    print("  !! 受检句数偏低，切句逻辑可能失效")
