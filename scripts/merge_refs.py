# -*- coding: utf-8 -*-
"""
合并多批已核实引用 -> docs/REFERENCES.json (最终库), 并做完整性自检。

规则:
  - 只接受 verified.channels 非空的条目 (即真过了 API 核实)
  - 按 (归一化标题) 去重, 保留信息更全的一条
  - 输出统一编号 [1]..[N], 按 (年份, 首作者) 排序, 便于成稿引用
  - 打印 REJECTED 汇总, 缺口如实报告
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
BATCHES = [
    ("Batch1", "REFERENCES_B1.json", "REF_REJECTED_B1.json"),
    ("Batch2", "REFERENCES_B2.json", "REF_REJECTED_B2.json"),
]
FINAL = os.path.join(DOCS, "REFERENCES.json")

# ---------------------------------------------------------------
# 相关性排除清单 (2026-08-11)
# 这些条目「真实存在且已通过 API 核实」, 但与本文主题(点云上采样)
# 无实质关联, 是早期为对位「管理学七章壳子」而引入的。
# 论文骨架改回标准 CV 学位论文结构后, 这些栏目不再存在, 故一并移除。
# 移除属于相关性判断, 不是核实失败 —— 因此单独留痕, 不混入 REF_REJECTED_*。
# ---------------------------------------------------------------
EXCLUDE_IRRELEVANT = {
    "TRL": "技术成熟度评估(TRL)属航天/工程管理评估框架，与点云上采样无方法学关联",
    "CMMI": "能力成熟度模型集成属软件过程管理，与本文无关（该条本已核实未通过）",
    "ShowYourWork": "NLP 实验报告规范，本文实验协议已由 PU-GCN 官方口径与自建自检器界定，无需外部报告规范背书",
    "ReproducibilityML": "ML 可重复性倡议报告，同上，属泛领域方法论倡议而非本文技术依据",
}



def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def richness(r):
    """信息完整度打分, 用于去重时保留更好的一条"""
    return (bool(r.get("doi")) * 2 + bool(r.get("arxiv")) * 2
            + len(r.get("authors") or []) * 0.1 + bool(r.get("venue")) * 1
            + len(r.get("verified", {}).get("channels") or []))


def main():
    all_refs, seen, dups = [], {}, []
    rej_all = []
    excluded_hits = []

    for tag, af, rf in BATCHES:
        ap = os.path.join(DOCS, af)
        rp = os.path.join(DOCS, rf)
        if not os.path.exists(ap):
            print("[SKIP] %s 不存在: %s" % (tag, af))
            continue
        d = json.load(open(ap, encoding="utf-8"))
        refs = d.get("references", [])
        print("[%s] 载入 %d 条 (来源候选: %s)" % (tag, len(refs), d.get("source_candidates", "?")))
        for r in refs:
            if not (r.get("verified", {}).get("channels")):
                print("   [DROP] %s 无核实通道" % r.get("key"))
                continue
            if r.get("key") in EXCLUDE_IRRELEVANT:
                excluded_hits.append((r.get("key"), EXCLUDE_IRRELEVANT[r["key"]]))
                print("   [EXCLUDE-IRRELEVANT] %s" % r.get("key"))
                continue
            k = norm(r.get("title"))
            if k in seen:
                old = seen[k]
                dups.append((r.get("key"), old.get("key")))
                if richness(r) > richness(old):
                    all_refs[all_refs.index(old)] = r
                    seen[k] = r
                continue
            seen[k] = r
            all_refs.append(r)
        if os.path.exists(rp):
            rj = json.load(open(rp, encoding="utf-8"))
            for x in rj.get("rejected", []):
                x["batch"] = tag
                rej_all.append(x)

    # 排序 + 编号
    def sk(r):
        y = r.get("year") or "9999"
        a = (r.get("authors") or [""])[0]
        return (int(y) if str(y).isdigit() else 9999, a.split()[-1] if a else "")

    all_refs.sort(key=sk)
    for i, r in enumerate(all_refs, 1):
        r["number"] = i

    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out = {
        "note": ("论文参考文献库。每条均经 arXiv / Crossref / OpenAlex 官方 API 逐条核实，"
                 "verify_urls 可复查。核实未通过的条目一律不收录，见 REF_REJECTED_*.json。"),
        "generated_at": ts,
        "n_references": len(all_refs),
        "verification_policy": {
            "channels": ["arxiv", "crossref", "dblp", "openalex"],
            "title_match": "归一化后相等/包含/token-Jaccard>=0.8",
            "year_tolerance": "±1 年（容忍 preprint 与会议年差异）",
            "rejected_are_excluded": True,
        },
        "excluded_for_irrelevance": {
            "note": ("以下条目真实存在且已通过 API 核实，但与本文主题无实质关联，"
                     "按相关性判断移除；与「核实未通过」是两类不同问题。"),
            "n": len(excluded_hits),
            "items": [{"key": k, "reason": v} for k, v in sorted(set(excluded_hits))],
        },
        "references": all_refs,
    }
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print("最终引用库: %d 条 -> %s" % (len(all_refs), FINAL))
    print("=" * 66)

    ch = Counter()
    for r in all_refs:
        for c in r["verified"]["channels"]:
            ch[c] += 1
    print("核实通道 :", dict(ch))
    print("有 DOI   : %d" % sum(1 for r in all_refs if r.get("doi")))
    print("有 arXiv : %d" % sum(1 for r in all_refs if r.get("arxiv")))
    print("去重合并 : %d 组" % len(dups))
    for a, b in dups:
        print("   %s <-> %s" % (a, b))

    print("\n相关性排除 : %d 条" % len(set(excluded_hits)))
    for k, v in sorted(set(excluded_hits)):
        print("   %-18s %s" % (k, v))
    missing_ex = set(EXCLUDE_IRRELEVANT) - {k for k, _ in excluded_hits}
    if missing_ex:
        print("   (清单中未在任何批次出现: %s)" % sorted(missing_ex))

    tp = Counter(r.get("topic", "?") for r in all_refs)
    print("\n主题分布 (%d 类):" % len(tp))
    for k, v in sorted(tp.items(), key=lambda x: -x[1]):
        print("  %-24s %d" % (k, v))

    yrs = [int(r["year"]) for r in all_refs if str(r.get("year", "")).isdigit()]
    if yrs:
        print("\n年份跨度 : %d – %d" % (min(yrs), max(yrs)))
        recent = sum(1 for y in yrs if y >= 2020)
        print("2020 年后 : %d 条 (%.0f%%)" % (recent, 100.0 * recent / len(yrs)))

    print("\n" + "-" * 66)
    print("核实未通过 (不收录): %d 条" % len(rej_all))
    print("-" * 66)
    for x in rej_all:
        det = x.get("errors") or x.get("api_titles") or x.get("api_years") or ""
        print("  [%s] %-20s %-16s %s" % (x["batch"], x["key"], x["reason"], det))

    print("\n目标 70+ 篇: %s (当前 %d 条)" %
          ("达标" if len(all_refs) >= 70 else "未达标", len(all_refs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
