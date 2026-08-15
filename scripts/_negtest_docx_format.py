# -*- coding: utf-8 -*-
"""audit_docx_format.py 的负例表：篡改成稿的段落格式/字号，
确认自检确实会报红。防「改了脚本但自检永远 PASS」的假绿。

用例：
  N1 把一个正文段行距改成 1.5 倍          -> 必须 FAIL
  N2 把一个正文段对齐改成左对齐            -> 必须 FAIL
  N3 把一个正文段的 run 字号改成 10.5pt(宋体) -> 必须 FAIL
  N4 把等宽 run 字体改成宋体但保留 10.5pt   -> 必须 FAIL（白名单须同时校验字体）
  N5 把全部正文段首行缩进清零（正文段识别失效）-> 必须 exit 2（受检数守卫）
"""
import io
import os
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'outputs', 'thesis', 'GAAW_thesis_v3.docx')
TMP = os.path.join(ROOT, 'outputs', 'thesis', '_negtest_fmt.docx')
AUDIT = os.path.join(ROOT, 'scripts', 'audit_docx_format.py')

NON_BODY = {'Heading 1', 'Heading 2', 'Heading 3', 'Caption',
            'List Paragraph', 'MTDisplayEquation', '公式',
            'Header', 'Footer', 'TOC 标题1'}


def body_paras(doc):
    out = []
    for p in doc.paragraphs:
        sn = p.style.name if p.style is not None else 'Normal'
        if sn in NON_BODY or not p.text.strip():
            continue
        fli = p.paragraph_format.first_line_indent
        if fli is None or abs(fli.pt - 24.0) > 0.01:
            continue
        out.append(p)
    return out


def run_audit(path):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, AUDIT], capture_output=True,
                       env=env, cwd=ROOT)
    return r.returncode


def n1(doc):
    body_paras(doc)[5].paragraph_format.line_spacing = 1.5


def n2(doc):
    body_paras(doc)[7].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT


def n3(doc):
    for r in body_paras(doc)[9].runs:
        if r.text.strip():
            r.font.size = Pt(10.5)
            r.font.name = '宋体'
            break


def n4(doc):
    for p in body_paras(doc):
        for r in p.runs:
            if r.font.size is not None and abs(r.font.size.pt - 10.5) < 0.01 \
                    and r.font.name == 'Consolas':
                r.font.name = '宋体'
                return
    raise SystemExit('N4 找不到等宽 run，用例无效')


def n5(doc):
    for p in body_paras(doc):
        p.paragraph_format.first_line_indent = Pt(0)


CASES = [('N1 行距改 1.5 倍', n1, 1),
         ('N2 对齐改左对齐', n2, 1),
         ('N3 正文 run 改 10.5pt 宋体', n3, 1),
         ('N4 等宽 run 字体改宋体（仍 10.5pt）', n4, 1),
         ('N5 全部正文段首行清零（识别失效）', n5, 2)]

# 基线：未篡改必须 PASS
shutil.copy2(SRC, TMP)
bak = SRC + '.negbak'
shutil.copy2(SRC, bak)
rc0 = run_audit(SRC)
print(f'  基线（未篡改）: exit={rc0} 期望 0  '
      f'{"PASS" if rc0 == 0 else "FAIL"}')

npass = 0
for name, fn, want in CASES:
    shutil.copy2(bak, SRC)
    doc = Document(SRC)
    fn(doc)
    doc.save(SRC)
    rc = run_audit(SRC)
    ok = (rc == want)
    npass += ok
    print(f'  {name}: exit={rc} 期望 {want}  {"PASS" if ok else "FAIL"}')

shutil.copy2(bak, SRC)
os.remove(bak)
os.remove(TMP)
print()
print(f'负例表: {npass}/{len(CASES)} PASS，基线 '
      f'{"PASS" if rc0 == 0 else "FAIL"}')
sys.exit(0 if (npass == len(CASES) and rc0 == 0) else 1)
