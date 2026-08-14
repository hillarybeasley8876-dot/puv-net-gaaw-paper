# -*- coding: utf-8 -*-
"""参考文献与正文引用的双向一致性终检（交付前必跑）。

四项检查，任一失败即不可交付：
  1. 正文每个 [N] 都能在 GB/T 7714 表中找到条目 → 否则悬空引用
  2. 表中每个条目都被正文引用             → 否则幽灵条目
  3. 表内编号无重复、且与 REFERENCES.json 一致
  4. 正文不残留 {{cite:...}} 真实占位符（说明性 ⟨key⟩ 除外）
"""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 本脚本位于 scripts/ 下，ROOT 为 2 层 dirname。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, 'docs', 'chapters')

refs = json.load(io.open(os.path.join(ROOT, 'docs', 'REFERENCES.json'),
                         encoding='utf-8'))['references']
json_nums = [int(r['number']) for r in refs]

md = io.open(os.path.join(ROOT, 'docs', 'REFERENCES_GB7714.md'),
             encoding='utf-8').read()
md_nums = [int(m.group(1)) for m in re.finditer(r'^\[(\d+)\]\s', md, re.M)]

FILES = [f for f in sorted(os.listdir(CH))
         if f.endswith('.md') and not f.startswith('_')
         and 'pre_outline' not in f and not f.endswith('.bak_cite')]

cited = {}
leftover = 0
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding='utf-8').read()
    for m in re.finditer(r'\[(\d+(?:,\d+)*)\]', t):
        for x in m.group(1).split(','):
            cited.setdefault(int(x), set()).add(f)
    leftover += len(re.findall(r'\{\{cite:[^}⟨⟩]+\}\}', t))

fails = []

print('=== 1. 正文 [N] -> 表中条目 ===')
dangling = sorted(n for n in cited if n not in set(md_nums))
print(f'  正文引用 {len(cited)} 个编号；悬空: '
      f'{dangling if dangling else "无"}')
if dangling:
    for n in dangling:
        print(f'    [{n}] 出现于 {sorted(cited[n])}')
    fails.append('悬空引用')

print()
print('=== 2. 表中条目 -> 正文引用 ===')
ghost = sorted(n for n in md_nums if n not in cited)
print(f'  表内条目 {len(md_nums)} 条；未被引用: '
      f'{ghost if ghost else "无"}')
if ghost:
    fails.append('幽灵条目')

print()
print('=== 3. 编号唯一性与库一致性 ===')
dup = sorted({n for n in md_nums if md_nums.count(n) > 1})
print(f'  表内重号: {dup if dup else "无"}')
same = sorted(md_nums) == sorted(json_nums)
print(f'  表 vs REFERENCES.json 编号集合一致: {same}')
if dup:
    fails.append('表内重号')
if not same:
    fails.append('表与库不一致')
    print(f'    仅表: {sorted(set(md_nums) - set(json_nums))}')
    print(f'    仅库: {sorted(set(json_nums) - set(md_nums))}')

print()
print('=== 4. 占位符残留 ===')
print(f'  真实 {{{{cite:KEY}}}} 残留: {leftover}（应为 0）')
if leftover:
    fails.append('占位符残留')

print()
nums = sorted(md_nums)
gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
print(f'编号范围 {nums[0]}–{nums[-1]}，条目 {len(nums)}，'
      f'空档 {len(gaps)} 个（已在表头说明）')
print()
print('结论:', '全部通过 —— 参考文献与正文双向一一对应'
      if not fails else f'失败 {fails}')
sys.exit(1 if fails else 0)
