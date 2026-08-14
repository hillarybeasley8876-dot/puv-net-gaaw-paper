# -*- coding: utf-8 -*-
"""
引用替换前的匹配核查（七章）。

三项检查：
  ① 全文用到的 cite key 是否都在 REFERENCES.json 中（缺失=无法替换）
  ② 库中有哪些 key 全文未用（冗余，可能是历史遗留）
  ③ 各章的引用密度（用于判断第 4/5/6/7 章新写内容是否引用过少）

不做替换，只报告。替换由 migrate_cite_numbers.py 执行。
"""
import json, os, io, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")
FILES = ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md",
         "ch4_design.md", "ch5_mechanism.md", "ch6_experiments.md", "ch7_conclusion.md"]

refs = json.load(io.open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
lib = {r["key"]: r["number"] for r in refs["references"]}
lib_lower = {k.lower(): k for k in lib}

used = Counter()
per_file = defaultdict(Counter)
for f in FILES:
    p = os.path.join(CH, f)
    if not os.path.exists(p):
        print(f"  !! 缺文件 {f}")
        continue
    t = io.open(p, encoding="utf-8").read()
    # 排除说明性写法 {{cite:⟨key⟩}}
    for m in re.finditer(r"\{\{cite:([^}⟨]+)\}\}", t):
        k = m.group(1).strip()
        used[k] += 1
        per_file[f][k] += 1

print("=" * 74)
print("① 全文 cite key 与库的匹配")
print("=" * 74)
print(f"  库中文献数    : {len(lib)}")
print(f"  全文用到的 key: {len(used)}  （引用次数合计 {sum(used.values())}）")

missing = [k for k in used if k not in lib]
case_fix = []
if missing:
    print()
    print(f"  ❌ 库中缺失 {len(missing)} 个 key（无法替换编号）：")
    for k in sorted(missing):
        hint = lib_lower.get(k.lower())
        if hint:
            case_fix.append((k, hint))
            print(f"     {k:26s} 用量 {used[k]:>3d}   ← 库中有大小写不同的 '{hint}'")
        else:
            print(f"     {k:26s} 用量 {used[k]:>3d}   ← 库中无任何近似项")
else:
    print("  ✅ 全部 key 均在库中")

print()
print("=" * 74)
print("② 库中未被引用的 key（冗余）")
print("=" * 74)
unused = sorted(set(lib) - set(used))
print(f"  未引用: {len(unused)} / {len(lib)}")
if unused:
    for i in range(0, min(len(unused), 40), 5):
        print("     " + "  ".join(f"{k:22s}" for k in unused[i:i+5]))
    if len(unused) > 40:
        print(f"     …（另有 {len(unused)-40} 个）")

print()
print("=" * 74)
print("③ 各章引用密度")
print("=" * 74)
print(f"  {'file':26s} {'唯一key':>8s} {'总次数':>8s}")
for f in FILES:
    c = per_file[f]
    print(f"  {f:26s} {len(c):>8d} {sum(c.values()):>8d}")

print()
print("=" * 74)
print("④ 高频引用 Top 15")
print("=" * 74)
for k, v in used.most_common(15):
    num = lib.get(k, "?")
    print(f"  [{num}] {k:26s} x{v}")

print()
print("=" * 74)
print("汇总")
print("=" * 74)
if missing:
    print(f"BLOCKED  {len(missing)} 个 key 不在库中，替换前须补齐或改名")
    if case_fix:
        print(f"         其中 {len(case_fix)} 个疑似大小写不一致，可直接改正文")
else:
    print("READY    可执行编号替换")
