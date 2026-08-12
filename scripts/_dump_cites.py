# -*- coding: utf-8 -*-
"""人工辅助检查: 打印每条 [N] 所在的句子 + 该编号对应的论文 key, 便于人工核对语义"""
import json, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
mp = {r["number"]: (r["title"][:50], r["key"]) for r in d["references"]}

raw = open(os.path.join(ROOT, "docs/chapters/ch1_introduction.md"), encoding="utf-8").read()
# 按段落切, 每段中找 [N]
for para in raw.split("\n\n"):
    if "[" not in para:
        continue
    sents = re.split(r"(。|；|！|\?)", para)
    for i in range(0, len(sents) - 1, 2):
        if "[" in sents[i]:
            text = (sents[i] + sents[i + 1]).replace("\n", " ").strip()
            for m in re.finditer(r"\[(\d+)\]", sents[i]):
                n = int(m.group(1))
                if n in mp:
                    short = text[:90].replace("  ", " ")
                    print("[%2d](%-12s) | %s" % (n, mp[n][1], short))
            print("  ----")
