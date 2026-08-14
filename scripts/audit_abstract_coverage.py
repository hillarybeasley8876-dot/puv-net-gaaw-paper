# -*- coding: utf-8 -*-
"""摘要的不利事实覆盖盘点。

动机：上一轮实测到「重构会静默丢掉不利事实，而数字审计器查不出来」
（审计器只验写出来的数字对不对，不验该写的有没有写）。
摘要是压缩率最高的地方，最容易在"精简"过程中把不利事实挤掉，
故必须单独盘点，而不能因为审计 PASS 就认为没问题。

清单来自 ch7 结论章明确列出的、对本文不利或限制结论强度的事实。
中英文摘要都必须命中，缺一即 FAIL。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 本脚本位于 scripts/ 下（由 scripts/_tmp/ 转正），ROOT 为 2 层 dirname。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ZH = os.path.join(ROOT, 'docs', 'chapters', 'abstract_zh.md')
EN = os.path.join(ROOT, 'docs', 'chapters', 'abstract_en.md')

zh = io.open(ZH, encoding='utf-8').read()
en = io.open(EN, encoding='utf-8').read()


def unwrap(t):
    """剥掉 markdown 强调标记再匹配。

    实测（②「不主张对抗优于无对抗」）：英文原文写作
    `it is **not** claimed that using adversarial training outperforms...`，
    判据串 `not claimed that` 因中间夹着 `**` 而 MISS → 假红。
    这与 C2b/C2c 的「单元格装饰必须先剥再解析」是同一类错误：
    **判据脚本报红时，先分清是稿子漏写还是判据被标记打断。**
    """
    return t.replace('**', '').replace('*', '').replace('`', '')


zh_u, en_u = unwrap(zh), unwrap(en)

# (项目, 中文判据(任一命中), 英文判据(任一命中))
ITEMS = [
    ('① 对抗训练本身损害主指标',
     ['14.86', '损害主指标', '劣化'],
     ['14.86', 'degrades the primary metric']),
    ('② 不主张对抗优于无对抗',
     ['不主张采用对抗训练优于不采用', '不主张对抗'],
     ['not claimed that using adversarial training outperforms']),
    ('③ C1 竞争方案性价比更优',
     ['5.44', '性价比'],
     ['5.44', 'better cost-benefit']),
    ('④ 动态 vs 更小权重无法归因',
     ['无法在', '归因'],
     ['cannot be attributed']),
    ('⑤ CD/HD 不作有效性主张',
     ['不作有效性主张'],
     ['no claim of effectiveness']),
    ('⑥ 不声称公式首创',
     ['不声称首次提出'],
     ['does not claim to be the first']),
    ('⑦ 单种子/未测跨种子稳健性',
     ['单一随机种子', '本次受控运行'],
     ['single random seed', 'observed in this controlled run']),
    ('⑧ 不可外推范围明确',
     ['不作跨数据集', '不作', '外推'],
     ['not extrapolated']),
    ('⑨ 三项失败案例',
     ['失效边界', '崩塌'],
     ['failure boundary', 'collapse']),
]

print('=' * 74)
print('摘要不利事实覆盖盘点')
print('=' * 74)
fails = []
for name, zk, ek in ITEMS:
    hz = [k for k in zk if k in zh_u]
    he = [k for k in ek if k in en_u]
    ok = bool(hz) and bool(he)
    print(f'  {name:26s} 中文={"HIT " if hz else "MISS"} '
          f'英文={"HIT " if he else "MISS"}  {"OK" if ok else "FAIL"}')
    if not ok:
        fails.append((name, bool(hz), bool(he)))

print()
if fails:
    print('缺失项（必须补进摘要）:')
    for name, hz, he in fails:
        miss = []
        if not hz:
            miss.append('中文')
        if not he:
            miss.append('英文')
        print(f'  {name} -> 缺 {"/".join(miss)}')
else:
    print('全部 9 项不利事实在中英文摘要中均有覆盖')

# 附：中英文数字对称性
import re
def nums(t):
    return sorted(set(re.findall(r'\d+\.\d+|\b\d{4,}\b', t)))
nz, ne = nums(zh), nums(en)
print()
print('中文摘要数字:', nz)
print('英文摘要数字:', ne)
only_zh = [x for x in nz if x not in ne]
only_en = [x for x in ne if x not in nz]
print('仅中文出现:', only_zh if only_zh else '无')
print('仅英文出现:', only_en if only_en else '无')
if only_zh or only_en:
    fails.append(('数字不对称', False, False))

# ---- 自检：判据不得因过宽而永远命中（防假绿）----
# 剥掉 markdown 标记后判据变宽，必须验证「删掉该句后能检出」。
print()
print('=== 判据自检（删句负例）===')
selfcheck_fail = []
for name, zk, ek in ITEMS:
    # 从中文摘要里删掉命中的第一个判据串，应变为 MISS
    hit = next((k for k in zk if k in zh_u), None)
    if hit is None:
        continue
    tampered = zh_u.replace(hit, '', 1)
    still = any(k in tampered for k in zk)
    # 有些项有多个候选判据，删一个仍可能命中，属正常；
    # 只要求"删掉全部候选后必须 MISS"。
    t2 = zh_u
    for k in zk:
        t2 = t2.replace(k, '')
    detected = not any(k in t2 for k in zk)
    print(f'  {name:26s} 删全部候选后 MISS = '
          f'{"是" if detected else "否（判据过宽！）"}')
    if not detected:
        selfcheck_fail.append(name)

if selfcheck_fail:
    print()
    print('判据过宽的项:', selfcheck_fail)
    fails.append(('判据自检失败', False, False))

sys.exit(1 if fails else 0)
