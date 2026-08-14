# -*- coding: utf-8 -*-
"""
引用语义人工核对表（七章，占位符阶段）。

为什么必须有这一步：自动校验只能验"key 在库中"与"上下文够字数"，
**抓不到"key 有效但语义错位"** —— 例如把 PU-GCN 的结论挂到 PU-Transformer 上。
这类错误只能靠人工逐条比对"所在句子"与"该文献实际讲什么"才能发现。

本脚本输出三列供人工核对：
  所在章节:行号 | 句子（截断） | key -> 库中标题
按 key 分组，使同一文献的全部引用位置相邻，便于一次性判断该 key 是否被一贯正确使用。

产物：docs/_cite_semantic_check.txt（脚本自写 utf-8，不用 shell 重定向）
"""
import json, os, io, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")
FILES = ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md",
         "ch4_design.md", "ch5_mechanism.md", "ch6_experiments.md", "ch7_conclusion.md"]

refs = json.load(io.open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
lib = {r["key"]: r for r in refs["references"]}

hits = defaultdict(list)   # key -> [(file, line, sentence)]
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    for li, line in enumerate(t.split("\n"), 1):
        if "{{cite:" not in line:
            continue
        # 按句切，定位每个 cite 所在的句子
        for seg in re.split(r"(?<=[。；！？])", line):
            for m in re.finditer(r"\{\{cite:([^}⟨]+)\}\}", seg):
                k = m.group(1).strip()
                # 清掉句中其他占位符便于阅读
                s = re.sub(r"\{\{cite:[^}]+\}\}", "", seg).strip()
                s = re.sub(r"\s+", " ", s)
                hits[k].append((f.replace("_introduction", "").replace("_related_work", "")
                                 .replace("_baseline", "").replace("_design", "")
                                 .replace("_mechanism", "").replace("_experiments", "")
                                 .replace("_conclusion", "").replace(".md", ""), li, s))

lines = []
lines.append("=" * 100)
lines.append("引用语义人工核对表（按 key 分组）")
lines.append("=" * 100)
lines.append("")
lines.append("核对方法：读每个 key 的库中标题，再逐条看它被用在什么句子里；")
lines.append("          若某条句子的论断不是该文献真正给出的，即为语义错位，须改稿或换 key。")
lines.append("")

for k in sorted(hits, key=lambda x: (-len(hits[x]), x.lower())):
    r = lib.get(k)
    n = r["number"] if r else "?"
    title = r["title"] if r else "!! 不在库中"
    year = r.get("year", "") if r else ""
    venue = (r.get("venue") or "")[:34] if r else ""
    lines.append("-" * 100)
    lines.append(f"[{n}] {k}   ({year}, {venue})")
    lines.append(f"     标题: {title}")
    note = (r or {}).get("note")
    if note:
        lines.append(f"     备注: {note[:150]}")
    lines.append(f"     被引 {len(hits[k])} 次:")
    for f, li, s in hits[k]:
        lines.append(f"       {f}:{li:<5} {s[:118]}")
    lines.append("")

out = os.path.join(ROOT, "docs", "_cite_semantic_check.txt")
io.open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")

print(f"[written] {out}")
print(f"  key 数 {len(hits)}，引用条目 {sum(len(v) for v in hits.values())}")
print()
# 控制台只打印引用次数最多的与最少的，全表看文件
print("引用次数分布：")
cnt = sorted(((len(v), k) for k, v in hits.items()), reverse=True)
print("  最多 8 个:", ", ".join(f"{k}x{n}" for n, k in cnt[:8]))
print("  仅 1 次的 key 数:", sum(1 for n, _ in cnt if n == 1))
missing = [k for k in hits if k not in lib]
print("  不在库中的 key:", missing if missing else "无")
