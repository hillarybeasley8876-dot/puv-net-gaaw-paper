"""导出同济示例的实际正文（含每段的直接格式），用于人工提取排版规范。"""
import os

DOCX = r"E:\AE-CC托管\puv-net\refs\tongji_template\tongji_example.docx"
OUT = r"E:\AE-CC托管\puv-net\refs\tongji_template\tongji_example.dump.txt"

from docx import Document
from docx.oxml.ns import qn

doc = Document(DOCX)
lines = []
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if not t:
        continue
    runs = p.runs
    # 取第一个非空 run 的直接字体信息
    fname = fsize = fbold = feast = None
    for r in runs:
        if r.text.strip():
            fname = r.font.name
            fsize = round(r.font.size.pt, 1) if r.font.size else None
            fbold = r.font.bold
            rPr = r._element.rPr
            if rPr is not None and rPr.rFonts is not None:
                feast = rPr.rFonts.get(qn('w:eastAsia'))
            break
    pf = p.paragraph_format
    ls = pf.line_spacing
    ls = round(ls.pt, 1) if hasattr(ls, "pt") else ls
    fi = round(pf.first_line_indent.pt, 1) if pf.first_line_indent else None
    meta = "style={} | ea={} en={} sz={} b={} | align={} ls={} indent={}".format(
        p.style.name, feast, fname, fsize, fbold, pf.alignment, ls, fi)
    lines.append("[{:03d}] {}\n      {}".format(i, t[:150], meta))

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, len(lines), "paragraphs")
print()
for l in lines[:120]:
    print(l)
