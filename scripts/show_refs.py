# -*- coding: utf-8 -*-
"""查看引用核实结果汇总"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "REFERENCES.json")
REJ = os.path.join(ROOT, "docs", "REF_REJECTED.json")


def main():
    if not os.path.exists(OUT):
        print("[WAIT] REFERENCES.json 尚未生成，核实仍在进行")
        return 1
    d = json.load(open(OUT, encoding="utf-8"))
    refs = d["references"]
    print("=" * 62)
    print("ACCEPTED = %d   generated_at = %s" % (len(refs), d["generated_at"]))
    print("=" * 62)

    ch = Counter()
    for r in refs:
        for c in r["verified"]["channels"]:
            ch[c] += 1
    print("核实通道分布 :", dict(ch))

    n_doi = sum(1 for r in refs if r.get("doi"))
    n_arx = sum(1 for r in refs if r.get("arxiv"))
    print("有 DOI       : %d / %d" % (n_doi, len(refs)))
    print("有 arXiv ID  : %d / %d" % (n_arx, len(refs)))

    tp = Counter(r.get("topic", "?") for r in refs)
    print("\n主题分布:")
    for k, v in sorted(tp.items(), key=lambda x: -x[1]):
        print("  %-24s %d" % (k, v))

    yr = Counter(r.get("year", "?") for r in refs)
    print("\n年份分布:")
    for k in sorted(yr):
        print("  %s : %s" % (k, "#" * yr[k]))

    print("\n章节覆盖(cite_in_chapter):")
    cc = Counter()
    for r in refs:
        for s in r.get("cite_in_chapter", []):
            cc[s] += 1
    for k, v in sorted(cc.items()):
        print("  %-6s %d" % (k, v))

    if os.path.exists(REJ):
        rj = json.load(open(REJ, encoding="utf-8"))
        print("\n" + "=" * 62)
        print("REJECTED = %d" % rj["n_rejected"])
        print("=" * 62)
        for r in rj["rejected"]:
            detail = r.get("errors") or r.get("api_titles") or r.get("api_years") or ""
            print("- %-22s %-18s %s" % (r["key"], r["reason"], detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
