# -*- coding: utf-8 -*-
"""
按 key 查 number, 输出 [key, number] 映射表, 写论文时按这个编号。
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
lines = ["按 key 查 number:"]
for r in sorted(d["references"], key=lambda x: x["number"]):
    lines.append("  [%d]  %-22s  %s" % (r["number"], r["key"], r["title"][:55]))
# 反向表: 按 key 字母序, 写作时更好查
lines.append("")
lines.append("按 key 字母序:")
for r in sorted(d["references"], key=lambda x: x["key"].lower()):
    lines.append("  %-24s -> [%d]" % (r["key"], r["number"]))
txt = "\n".join(lines) + "\n"
# 必须由脚本自己写 utf-8; PowerShell 重定向会写成 UTF-16/GBK
out = os.path.join(ROOT, "docs", "_key2num.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(txt)
print(txt)
print("[written] %s (%d refs)" % (out, len(d["references"])))
