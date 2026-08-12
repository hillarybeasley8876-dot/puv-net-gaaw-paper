# -*- coding: utf-8 -*-
"""
参考文献嵌入自检器:
  1. 扫描指定 .md, 提取所有 [N] 形式的引用标记
  2. 验证 N 是否在最终库中 (防止 hallucination)
  3. 验证每个引用标记前必须紧跟至少 1 句中文正文 (防止 [3] 这种孤立编号)
  4. 统计字数、引用密度、每章引用数, 输出报告
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
REF = os.path.join(DOCS, "REFERENCES.json")


def main():
    if len(sys.argv) < 2:
        print("用法: python selfcheck_inline_cites.py <chapter.md>")
        return 1
    src_path = sys.argv[1]
    if not os.path.isabs(src_path):
        src_path = os.path.join(ROOT, src_path)

    raw = open(src_path, encoding="utf-8").read()
    print("=" * 70)
    print("  内嵌引用自检: %s" % os.path.basename(src_path))
    print("  总字节: %d  估计字数: %d" % (len(raw), len(re.findall(r"[\u4e00-\u9fff]", raw))))
    print("=" * 70)

    # 合法编号集合
    d = json.load(open(REF, encoding="utf-8"))
    valid = {r["number"] for r in d["references"]}

    # 抽取所有 [N] 标记 (允许一个括号内多个: [12, 23, 45] 或 [1-3] 不支持)
    marks = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", raw)
    flat = []
    for m in marks:
        for x in re.split(r"\s*,\s*", m):
            flat.append(int(x))

    n_total = len(flat)
    n_unique = len(set(flat))
    uniq_nums = sorted(set(flat))

    print("\n[1] 引用标记统计")
    print("  出现标记总数 : %d" % n_total)
    print("  去重后编号数 : %d" % n_unique)
    print("  编号范围     : %d - %d" % (min(flat), max(flat)))

    # 有效性
    print("\n[2] 编号有效性 (与 REFERENCES.json 严格对齐)")
    invalid = [n for n in uniq_nums if n not in valid]
    if not invalid:
        print("  [PASS] 全部 %d 个编号在最终库中存在" % n_unique)
    else:
        print("  [FAIL] %d 个编号不在库中: %s" % (len(invalid), invalid))

    # 章节引用数
    print("\n[3] 各章引用数 (按一级标题分块)")
    blocks = re.split(r"(^# .+)$", raw, flags=re.M)
    chap_refs = []
    for i in range(1, len(blocks), 2):
        title = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        nums = re.findall(r"\[(\d+)\]", body)
        uniq = sorted(set(int(x) for x in nums))
        chap_refs.append((title, len(nums), len(uniq), uniq))
    for t, n, nu, arr in chap_refs:
        print("  %-50s  标记 %d  去重 %d" % (t[:48], n, nu))

    # 上下文检查: 每个 [N] 之前必须紧跟 1 句中文 (≥ 8 个汉字)
    print("\n[4] 引用上下文 (每个 [N] 前 80 字符内必须有 ≥ 8 个汉字)")
    bad = []
    for m in re.finditer(r"\[(\d+)\]", raw):
        ctx = raw[max(0, m.start() - 80): m.start()]
        if len(re.findall(r"[\u4e00-\u9fff]", ctx)) < 8:
            bad.append((m.group(0), ctx[-50:].replace("\n", " ")))
    if not bad:
        print("  [PASS] 所有 %d 个引用标记前均有充分中文上下文" % n_total)
    else:
        print("  [FAIL] %d 个引用标记上下文不足 (< 8 个汉字):" % len(bad))
        for tag, c in bad[:10]:
            print("    %s  ←  '%s...'" % (tag, c))

    # 悬空检查: 章节内编号被引但本节没明确语义指向 → 只提示, 不 FAIL
    print("\n[5] 编号使用频次 Top-10")
    cnt = defaultdict(int)
    for n in flat:
        cnt[n] += 1
    for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:10]:
        title = next((r["title"][:42] for r in d["references"] if r["number"] == n), "?")
        print("  [%-3d] x %2d  %s" % (n, c, title))

    # 字数
    print("\n[6] 估计字数 / 引用密度")
    chinese = len(re.findall(r"[\u4e00-\u9fff]", raw))
    print("  纯汉字数: %d" % chinese)
    print("  引用密度: %.1f 个 [N] / 千字" % (n_total * 1000 / max(1, chinese)))

    print("\n" + "=" * 70)
    print("  结论: %s" % ("通过" if not invalid and not bad else "需修正"))
    print("=" * 70)
    return 0 if (not invalid and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
