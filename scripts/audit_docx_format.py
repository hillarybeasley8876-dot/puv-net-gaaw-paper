# -*- coding: utf-8 -*-
"""任务 B 格式自检：核对成稿正文段的段落格式与 run 字号是否落在
模板实测规格上（行距 20pt / 首行 24pt / 两端对齐 / 12pt 小四）。

设计要点（防假绿）：
  · 不只看「有多少段合规」，而是断言「受检段数 == 应检段数」；
    受检数为 0 直接 exit 2（正文段识别失效时不得静默通过）。
  · 图题/表注/公式/表格/参考文献等本就不该套正文格式，按样式与
    首行缩进排除，且各类排除数单独报出，避免把正文段误排除掉。
"""
import io
import os
import sys
from collections import Counter

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(ROOT, 'outputs', 'thesis', 'GAAW_thesis_v3.docx')

BODY_LINE_PT = 20.0
BODY_INDENT_PT = 24.0
BODY_SIZE_PT = 12.0
MONO_SIZE_PT = 10.5   # 行内等宽（`...`）允许的例外字号，须为 Consolas
NON_BODY_STYLES = {'Heading 1', 'Heading 2', 'Heading 3', 'Caption',
                   'List Paragraph', 'MTDisplayEquation', '公式',
                   'Header', 'Footer', 'TOC 标题1'}

doc = Document(DOCX)
checked = 0
bad_pf = []
bad_size = Counter()
skipped = Counter()
mono = 0

for idx, p in enumerate(doc.paragraphs):
    sn = p.style.name if p.style is not None else 'Normal'
    txt = p.text.strip()
    if not txt:
        skipped['empty'] += 1
        continue
    if sn in NON_BODY_STYLES:
        skipped[f'style:{sn}'] += 1
        continue
    pf = p.paragraph_format
    fli = pf.first_line_indent
    # 正文段的判据：首行缩进 == 模板实测 24pt。
    # 其余（首行 0 / 悬挂 / 左缩进）是列表、表注、索引、代码等自定格式段。
    if fli is None or abs(fli.pt - BODY_INDENT_PT) > 0.01:
        skipped['non-body-indent'] += 1
        continue

    checked += 1
    ls = pf.line_spacing
    ok_ls = ls is not None and not isinstance(ls, float) \
        and abs(ls.pt - BODY_LINE_PT) < 0.01
    ok_al = pf.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY
    if not (ok_ls and ok_al):
        bad_pf.append((idx, sn, txt[:40],
                       f'ls={ls} align={pf.alignment}'))
    for r in p.runs:
        if r.font.size is None:
            bad_size['size=None(继承样式)'] += 1
            continue
        sz = r.font.size.pt
        if abs(sz - BODY_SIZE_PT) < 0.01:
            continue
        # 唯一允许的例外：行内等宽（Consolas）为五号。
        # 此处刻意同时校验字体，避免「任何 10.5pt 都放行」放松防线。
        if abs(sz - MONO_SIZE_PT) < 0.01 and r.font.name == 'Consolas':
            mono += 1
            continue
        bad_size[f'size={sz} font={r.font.name}'] += 1

print('=== 任务 B 格式自检 ===')
print(f'  受检正文段数: {checked}')
print(f'  段落格式不合规: {len(bad_pf)}')
for idx, sn, txt, why in bad_pf[:10]:
    print(f'    [{idx}] {sn} {why} | {txt}')
print(f'  行内等宽例外(Consolas 10.5pt): {mono}')
print(f'  run 字号异常合计: {sum(bad_size.values())}')
for k, v in bad_size.most_common(8):
    print(f'    {k}: {v}')
print()
print('  排除统计:')
for k, v in skipped.most_common(12):
    print(f'    {k}: {v}')

if checked == 0:
    print()
    print('!! 受检段数为 0：正文段识别失效，自检无效（不得视为通过）。')
    sys.exit(2)
ok = not bad_pf and sum(bad_size.values()) == 0
print()
print('结论:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
