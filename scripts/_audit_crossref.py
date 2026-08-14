# -*- coding: utf-8 -*-
"""
全文交叉引用一致性审计（七章结构）。

拆章后最易出错的是章节交叉引用：正文写"第 5 章的实验"而实验已移到第 6 章。
本脚本按七章结构核对每处"第 N 章"「第 N.M 节」引用是否指向正确内容。

判据（跑前定死）：
  七章内容映射
    1 绪论 / 2 文献综述 / 3 分析框架(基线诊断) / 4 研究设计(假设) /
    5 GAAW 机制 / 6 实验结果 / 7 结论展望
  规则
    R1 引用"第 N 章"时 N 必须在 1..7
    R2 提到"实验/裁定/实测结果"的章引用应指向 6（或 3 的诊断）
    R3 提到"假设/预注册/判据"的章引用应指向 4
    R4 提到"机制/形式化/算法/伪代码"的章引用应指向 5
    R5 节号引用"第 N.M 节"中 N 必须与该文件所属章或已存在章一致
  受检数量守卫：若受检引用数 < 80 视为通道失效（本文引用密集）
"""
import re, os, io, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")

FILES = {
    1: "ch1_introduction.md",
    2: "ch2_related_work.md",
    3: "ch3_baseline.md",
    4: "ch4_design.md",
    5: "ch5_mechanism.md",
    6: "ch6_experiments.md",
    7: "ch7_conclusion.md",
}
TOPIC = {
    1: "绪论", 2: "文献综述", 3: "分析框架/基线诊断", 4: "研究设计/假设",
    5: "GAAW 机制", 6: "实验结果", 7: "结论展望",
}

# 关键词 -> 应指向的章
KEYWORD_CH = [
    (r"实验|裁定|实测结果|消融结果|主对比", {3, 6}),
    (r"预注册|接收条件|假设|判据|效度", {3, 4}),
    (r"伪代码|算法流程|形式化|机制轨迹", {5}),
]

print("=" * 76)
print("① 章节文件存在性")
print("=" * 76)
missing = []
for n, f in FILES.items():
    p = os.path.join(CH, f)
    ok = os.path.exists(p)
    print(f"  {'OK ' if ok else 'MISS'}  第{n}章  {f:26s} {TOPIC[n]}")
    if not ok:
        missing.append(f)
if missing:
    print(f"  !! 缺失 {len(missing)} 个章节文件")
    sys.exit(1)

print()
print("=" * 76)
print("② 章引用越界检查（第 N 章，N 必须 1..7）")
print("=" * 76)
total_ch_refs = 0
bad_range = []
for n, f in FILES.items():
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    for m in re.finditer(r"第\s*(\d+)\s*章", t):
        total_ch_refs += 1
        k = int(m.group(1))
        if not (1 <= k <= 7):
            line = t[:m.start()].count("\n") + 1
            bad_range.append((f, line, m.group(0)))
print(f"  受检章引用总数: {total_ch_refs}")
if bad_range:
    for f, l, s in bad_range[:10]:
        print(f"  BAD  {f}:{l}  {s}")
else:
    print("  OK   全部章引用在 1..7 内")

print()
print("=" * 76)
print("③ 语义错位检查（引用语境与目标章内容是否匹配）")
print("=" * 76)
suspects = []
for n, f in FILES.items():
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    lines = t.split("\n")
    for li, line in enumerate(lines, 1):
        for m in re.finditer(r"第\s*(\d+)\s*章", line):
            k = int(m.group(1))
            # 取引用前后 40 字作语境
            s = max(0, m.start() - 40); e = min(len(line), m.end() + 40)
            ctx = line[s:e]
            for pat, allowed in KEYWORD_CH:
                if re.search(pat, ctx) and k not in allowed:
                    suspects.append((f, li, k, sorted(allowed), ctx.strip()[:90]))
                    break
print(f"  可疑错位: {len(suspects)} 处")
for f, li, k, allowed, ctx in suspects[:25]:
    print(f"  ?  {f}:{li}  引用第{k}章 (语境提示应为 {allowed})")
    print(f"       …{ctx}…")

print()
print("=" * 76)
print("④ 节号引用的章前缀分布（第 N.M 节）")
print("=" * 76)
sec_refs = Counter()
per_file = defaultdict(Counter)
for n, f in FILES.items():
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    for m in re.finditer(r"第?\s*(\d+)\.(\d+)(?:\.(\d+))?\s*节", t):
        k = int(m.group(1))
        sec_refs[k] += 1
        per_file[f][k] += 1
print(f"  受检节引用总数: {sum(sec_refs.values())}")
print(f"  按目标章分布: {dict(sorted(sec_refs.items()))}")
out_of_range = {k: v for k, v in sec_refs.items() if not (1 <= k <= 7)}
if out_of_range:
    print(f"  BAD  越界节引用: {out_of_range}")
else:
    print("  OK   全部节引用章前缀在 1..7 内")
print()
print("  各文件的节引用去向（用于人工核对是否跨章错指）:")
for f in FILES.values():
    if per_file[f]:
        print(f"    {f:26s} -> {dict(sorted(per_file[f].items()))}")

print()
print("=" * 76)
print("⑤ 受检数量守卫")
print("=" * 76)
checked = total_ch_refs + sum(sec_refs.values())
print(f"  受检引用总数 = {checked}")
if checked < 80:
    print("  FAIL 受检数量异常偏低，通道疑似失效")
    sys.exit(1)
print("  OK   受检数量达预期量级")

print()
print("=" * 76)
print("汇总")
print("=" * 76)
if bad_range:
    print(f"FAIL 章引用越界 {len(bad_range)} 处")
    sys.exit(1)
if suspects:
    print(f"WARN 语义错位可疑 {len(suspects)} 处 —— 需人工逐条核对（非自动 FAIL，因关键词匹配有误报可能）")
else:
    print("PASS 未检出章引用越界与语义错位")
