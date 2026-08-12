# -*- coding: utf-8 -*-
"""审计最终引用库: 逐条可疑项检查, 输出需人工复核的清单。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "docs", "REFERENCES.json")

d = json.load(open(FINAL, encoding="utf-8"))
refs = d["references"]
print("总数: %d\n" % len(refs))

print("=" * 70)
print("[A] 早于 1995 年的条目 (需确认是否为经典文献而非误匹配)")
print("=" * 70)
for r in refs:
    y = r.get("year", "")
    if str(y).isdigit() and int(y) < 1995:
        print("  [%s] %s (%s)" % (y, r["title"][:62], r.get("key")))
        print("        channels=%s  url=%s" % (r["verified"]["channels"],
                                              r["verified"]["verify_urls"][0][:88]))

print("\n" + "=" * 70)
print("[B] 无作者信息的条目")
print("=" * 70)
n = 0
for r in refs:
    if not r.get("authors"):
        n += 1
        print("  %s | %s (%s)" % (r.get("key"), r["title"][:58], r.get("year")))
print("  共 %d 条" % n)

print("\n" + "=" * 70)
print("[C] 既无 DOI 又无 arXiv ID 的条目 (可查性最弱)")
print("=" * 70)
n = 0
for r in refs:
    if not r.get("doi") and not r.get("arxiv"):
        n += 1
        print("  %-22s %-52s %s" % (r.get("key"), r["title"][:50], r.get("year")))
print("  共 %d 条" % n)

print("\n" + "=" * 70)
print("[D] 标题疑似不完整 (< 20 字符)")
print("=" * 70)
for r in refs:
    if len(r["title"]) < 20:
        print("  %s | %r" % (r.get("key"), r["title"]))

print("\n" + "=" * 70)
print("[E] 核实通道只有 1 个且为 openalex 的条目 (证据最弱, 建议补 DOI)")
print("=" * 70)
n = 0
for r in refs:
    ch = r["verified"]["channels"]
    if ch == ["openalex"] and not r.get("doi"):
        n += 1
        print("  %-22s %-50s %s" % (r.get("key"), r["title"][:48], r.get("year")))
print("  共 %d 条" % n)
