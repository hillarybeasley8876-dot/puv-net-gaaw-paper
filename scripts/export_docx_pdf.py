# -*- coding: utf-8 -*-
"""用 Word COM 打开 docx，更新全部域（目录/页码），另存为 PDF。

为什么必须走 COM：目录与页码是 Word 域（TOC / PAGE），
python-docx 只能写入域代码，不能计算结果；规范也明确页码不得手敲。
COM 打开后 Fields.Update + TablesOfContents.Update 才会生成真实页码。
"""
import os
import sys
import time

import win32com.client as win32

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 文件名可由命令行覆盖（不带扩展名）。默认 v3 = 当前成稿。
# 教训：曾写死 v2，实测把旧稿当本轮成果导出并据此做视觉验收，
# 差点得出错误结论。此处默认必须跟随当前成稿。
STEM = sys.argv[1] if len(sys.argv) > 1 else 'GAAW_thesis_v3'
DOCX = os.path.join(ROOT, 'outputs', 'thesis', STEM + '.docx')
PDF = os.path.join(ROOT, 'outputs', 'thesis', STEM + '.pdf')

WD_FORMAT_PDF = 17

if not os.path.exists(DOCX):
    print('[FATAL] docx 不存在:', DOCX)
    sys.exit(1)

word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = None
try:
    doc = word.Documents.Open(DOCX, ReadOnly=False)
    # 更新域两轮：第一轮生成目录条目，第二轮修正因目录占页导致的页码位移
    for rnd in (1, 2):
        doc.Fields.Update()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(i).Update()
        for i in range(1, doc.TablesOfFigures.Count + 1):
            doc.TablesOfFigures(i).Update()
        time.sleep(1)
        print(f'  域更新第 {rnd} 轮完成')

    doc.Repaginate()
    pages = doc.ComputeStatistics(2)      # wdStatisticPages
    words = doc.ComputeStatistics(0)      # wdStatisticWords
    chars = doc.ComputeStatistics(3)      # wdStatisticCharacters
    print(f'  页数 {pages}   字数 {words}   字符 {chars}')

    doc.Save()
    doc.SaveAs2(PDF, FileFormat=WD_FORMAT_PDF)
    print('已导出 PDF:', os.path.relpath(PDF, ROOT))
    print(f'  体积 {os.path.getsize(PDF) / 1024 / 1024:.2f} MB')
finally:
    if doc is not None:
        doc.Close(SaveChanges=0)
    word.Quit()
