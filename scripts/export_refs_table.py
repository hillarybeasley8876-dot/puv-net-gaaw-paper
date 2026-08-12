# -*- coding: utf-8 -*-
"""把最终引用库导出为 markdown 表, 方便人工分配章节。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "REFERENCES.json")
OUT = os.path.join(ROOT, "docs", "REFERENCES_TABLE.md")

d = json.load(open(SRC, encoding="utf-8"))
refs = d["references"]

lines = []
lines.append("# 论文引用分配表（按章节预分配）\n")
lines.append("> 本文 117 条引用按 `cite_in_chapter` 字段预分配到 7 章及子节。\n")
lines.append("> 写作时按分配表使用；如某节不足，可向相邻章节借调。\n")
lines.append("\n## 全部条目（按编号）\n")
lines.append("\n| # | key | topic | year | 建议引用章节 | 标题（短） |\n")
lines.append("|---|---|---|---|---|---|\n")
for r in refs:
    cite = ", ".join(r.get("cite_in_chapter", [])) or "—"
    title = r["title"].replace("|", "/").replace("\n", " ")[:58]
    lines.append("| %d | `%s` | %s | %s | %s | %s |\n" % (
        r["number"], r["key"], r.get("topic", "?"), r.get("year", "?"), cite, title))

# 按章节聚合
lines.append("\n## 按章节聚合（写入正文时的检索用）\n")
ch_map = {}
for r in refs:
    for c in r.get("cite_in_chapter", []):
        ch_map.setdefault(c, []).append(r)

for ch in sorted(ch_map):
    items = ch_map[ch]
    items.sort(key=lambda r: r["number"])
    lines.append("\n### §%s（%d 条）\n" % (ch, len(items)))
    lines.append("\n| # | 引用 | 主题 |\n|---|---|---|\n")
    for r in items:
        cite = "%s et al., %s, *%s*" % (
            (r.get("authors") or ["?"])[0].split()[-1] if r.get("authors") else "?",
            r.get("year", "?"),
            r["title"].replace("*", "").replace("|", "/")[:55])
        lines.append("| %d | %s | %s |\n" % (r["number"], cite, r.get("topic", "?")))

with open(OUT, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("[OK] %s (%d B, %d 条引用, %d 章节)" % (OUT, os.path.getsize(OUT), len(refs), len(ch_map)))
