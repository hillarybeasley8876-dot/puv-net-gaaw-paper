# -*- coding: utf-8 -*-
"""全文交叉引用（节号 / 图号 / 表号）存在性核对。

动机：ch2 曾引用「第 5.5.4 节」，而 ch5 实际只到 §5.5.2 —— 悬空引用。
这类错误数字审计器查不出（它只管数值回溯），且引用编号替换后更难发现，
故在替换前单独核一遍。

规则：
  · 节号 第 X.Y[.Z] 节  -> chX 必须存在对应 ## X.Y 或 ### X.Y.Z 标题
  · 图号 图 X-N        -> 必须在某章里有「| 图 X-N |」图题行或定义
  · 表号 表 X-N        -> 必须有 **表 X-N  ...** 标题
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 本脚本位于 scripts/ 下（由 scripts/_tmp/ 转正），ROOT 为 2 层 dirname。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_DIR = os.path.join(ROOT, 'docs', 'chapters')

FILES = {}
for f in sorted(os.listdir(CH_DIR)):
    if not f.endswith('.md') or f.startswith('_') or 'pre_outline' in f:
        continue
    m = re.match(r'ch(\d)_', f)
    FILES[int(m.group(1)) if m else 0] = f

# --- 收集实际存在的锚点 ---
sections = set()
figures = set()
tables = set()
for num, f in FILES.items():
    t = io.open(os.path.join(CH_DIR, f), encoding='utf-8').read()
    for m in re.finditer(r'^##\s+(\d+\.\d+)\s', t, re.M):
        sections.add(m.group(1))
    for m in re.finditer(r'^###\s+(\d+\.\d+\.\d+)\s', t, re.M):
        sections.add(m.group(1))
    for m in re.finditer(r'^#\s+第\s*(\d+)\s*章', t, re.M):
        sections.add(m.group(1))
    # 图题定义：表格行 | 图 X-N | ... |
    for m in re.finditer(r'\|\s*图\s*(\d+\.\d+)\s*\|', t):
        figures.add(m.group(1))
    for m in re.finditer(r'^\*\*图\s*(\d+\.\d+)\s', t, re.M):
        figures.add(m.group(1))
    for m in re.finditer(r'^\*\*表\s*(\d+\.\d+)\s', t, re.M):
        tables.add(m.group(1))

print(f'锚点：节 {len(sections)}  图 {len(figures)}  表 {len(tables)}')
print(f'  图: {sorted(figures)}')
print(f'  表: {sorted(tables)}')
print()

bad = []
for num, f in sorted(FILES.items()):
    t = io.open(os.path.join(CH_DIR, f), encoding='utf-8').read()
    lines = t.split('\n')
    for i, ln in enumerate(lines, 1):
        for m in re.finditer(r'第\s*(\d+\.\d+(?:\.\d+)?)\s*节', ln):
            if m.group(1) not in sections:
                bad.append(('节', f, i, m.group(1), ln.strip()[:96]))
        for m in re.finditer(r'图\s*(\d+\.\d+)', ln):
            if m.group(1) not in figures:
                bad.append(('图', f, i, m.group(1), ln.strip()[:96]))
        for m in re.finditer(r'表\s*(\d+\.\d+)', ln):
            if m.group(1) not in tables:
                bad.append(('表', f, i, m.group(1), ln.strip()[:96]))

# ---- 防假绿守卫 ----
# 锚点集合为空时，「悬空引用」判定失去意义：脚本仍会报「0 处」。
# 实测教训（2026-08-15）：图表编号由连字符改为点号后，本脚本的锚点正则
# 仍写 `\d+-\d+`，导致锚点与引用两侧同时匹配为空，全程静默通过。
# 因此这里显式断言锚点数量非零，且与预期规模相符。
guard_fail = False
if not sections:
    print('  !! 节锚点为空 —— 正则失配，本次判定不可信')
    guard_fail = True
if not figures:
    print('  !! 图锚点为空 —— 正则失配，本次判定不可信')
    guard_fail = True
if not tables:
    print('  !! 表锚点为空 —— 正则失配，本次判定不可信')
    guard_fail = True

print('=== 悬空交叉引用 ===')
if not bad:
    print('  无')
else:
    for kind, f, i, ref, ctx in bad:
        print(f'  [{kind}] {f}:{i}  引用「{ref}」不存在')
        print(f'        {ctx}')
print()
print(f'合计 {len(bad)} 处')
sys.exit(2 if guard_fail else (1 if bad else 0))
