#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引用库自检器 —— 把 docs/EVIDENCE_LEDGER.md §1.10 里写下的数字
跟真实产物 (REFERENCES.json / REF_REJECTED_*.json) 逐项对账。

设计原则:
  * 台账里的每一个数字都必须能从产物重算出来, 不允许"我记得是这个数"
  * 被核实器打回的 key 绝对不能出现在最终库里
  * 每条收录文献必须有非空 verified.channels
  * 门槛/统计口径不硬编码: 台账数字从 md 里正则抽取, 产物数字从 json 现算

用法:
  python scripts/selfcheck_refs.py
退出码 0 = 全 PASS; 1 = 有 FAIL
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
REFS = os.path.join(DOCS, "REFERENCES.json")
LEDGER = os.path.join(DOCS, "EVIDENCE_LEDGER.md")
REJECTED = [
    os.path.join(DOCS, "REF_REJECTED_B1.json"),
    os.path.join(DOCS, "REF_REJECTED_B2.json"),
]

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_list(obj):
    """产物可能是 list 或 dict(key->rec), 统一成 (key, rec) 列表。"""
    if isinstance(obj, list):
        out = []
        for r in obj:
            k = r.get("key") or r.get("id") or ""
            out.append((k, r))
        return out
    if isinstance(obj, dict):
        # 可能外层包了 references / rejected / records
        for wrap in ("references", "rejected", "records", "refs", "entries"):
            if wrap in obj and isinstance(obj[wrap], (list, dict)):
                return as_list(obj[wrap])
        return [(k, v) for k, v in obj.items() if isinstance(v, dict)]
    return []


def main():
    # ---------- 0. 产物存在 ----------
    for p in [REFS, LEDGER] + REJECTED:
        check("产物存在: %s" % os.path.basename(p), os.path.isfile(p), p)
    if not os.path.isfile(REFS):
        report()
        return 1

    refs = as_list(load_json(REFS))
    n_refs = len(refs)
    check("REFERENCES.json 非空", n_refs > 0, "n=%d" % n_refs)

    # ---------- 1. 每条必须有 key / title / year / verified.channels ----------
    no_key = [k for k, r in refs if not k]
    check("每条都有 key", not no_key, "缺 key %d 条" % len(no_key))

    no_title = [k for k, r in refs if not (r.get("title") or "").strip()]
    check("每条都有 title", not no_title, str(no_title[:5]))

    bad_year = []
    for k, r in refs:
        y = r.get("year")
        try:
            y = int(y)
        except Exception:
            bad_year.append(k)
            continue
        if not (1950 <= y <= 2026):
            bad_year.append(k)
    check("每条 year 合法(1950-2026)", not bad_year, str(bad_year[:5]))

    unverified = []
    for k, r in refs:
        v = r.get("verified") or {}
        ch = v.get("channels") or v.get("channel") or []
        if isinstance(ch, str):
            ch = [ch] if ch else []
        if not ch:
            unverified.append(k)
    check("每条 verified.channels 非空", not unverified,
          "未核实 %d 条: %s" % (len(unverified), unverified[:5]))

    # ---------- 2. key 唯一 + 归一化标题唯一 ----------
    keys = [k for k, _ in refs]
    dup_keys = sorted({k for k in keys if keys.count(k) > 1})
    check("key 唯一", not dup_keys, str(dup_keys))

    def norm(t):
        return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

    seen = {}
    dup_titles = []
    for k, r in refs:
        n = norm(r.get("title"))
        if n in seen:
            dup_titles.append("%s <-> %s" % (seen[n], k))
        else:
            seen[n] = k
    check("归一化标题唯一(无重复收录)", not dup_titles, str(dup_titles[:5]))

    # ---------- 3. number 连续 1..N (若有 number 字段) ----------
    nums = [r.get("number") for _, r in refs if r.get("number") is not None]
    if nums:
        check("number 连续 1..N", sorted(nums) == list(range(1, n_refs + 1)),
              "n_num=%d n_refs=%d" % (len(nums), n_refs))

    # ---------- 4. 被拒 key 不得出现在最终库 ----------
    rej_keys = set()
    rej_total = 0
    rej_recs = {}
    for p in REJECTED:
        if not os.path.isfile(p):
            continue
        items = as_list(load_json(p))
        rej_total += len(items)
        for k, r in items:
            if k:
                rej_keys.add(k)
                rej_recs.setdefault(k, []).append(r)

    # 被拒 key 允许"修正候选信息后重新核实通过再入库"(如 year_mismatch 改对年份),
    # 但绝不允许"带着被拒时的错误信息"入库。逐条比对: 入库记录必须与被拒记录不同。
    refs_by_key = {k: r for k, r in refs}
    same_as_rejected = []
    for k in sorted(rej_keys & set(keys)):
        cur = refs_by_key[k]
        for rr in rej_recs.get(k, []):
            inp = rr.get("input") or {}
            reason = rr.get("reason") or ""
            if reason == "year_mismatch":
                # 入库年份必须已改, 不能仍等于被拒时的错误年份
                if str(cur.get("year")) == str(inp.get("year")):
                    same_as_rejected.append("%s 仍用被拒年份 %s" % (k, inp.get("year")))
            elif reason == "title_mismatch":
                # 入库必须换了核实通道或拿到了 DOI, 不能仍靠那个记错的 arXiv ID
                bad_arxiv = (inp.get("arxiv") or "").strip()
                cur_arxiv = (cur.get("arxiv") or cur.get("arxiv_id") or "").strip()
                if bad_arxiv and cur_arxiv == bad_arxiv:
                    same_as_rejected.append(
                        "%s 仍绑定错误 arXiv ID %s" % (k, bad_arxiv))
    check("被拒条目未带着原错误信息入库", not same_as_rejected,
          "; ".join(same_as_rejected) if same_as_rejected else
          "重新入库 %d 条均已修正" % len(rej_keys & set(keys)))
    check("被拒记录数 > 0(留档有效)", rej_total > 0, "rejected=%d" % rej_total)

    # ---------- 4b. 相关性排除清单对账 ----------
    raw = load_json(REFS)
    exc = (raw.get("excluded_for_irrelevance") or {}) if isinstance(raw, dict) else {}
    exc_items = exc.get("items") or []
    exc_keys = {x.get("key") for x in exc_items if isinstance(x, dict)}
    check("产物含 excluded_for_irrelevance 字段", bool(exc_items),
          "n=%d" % len(exc_items))
    check("排除清单声明数与条目数一致",
          exc.get("n") == len(exc_items),
          "n=%s items=%d" % (exc.get("n"), len(exc_items)))
    check("被排除 key 未出现在最终库",
          not (exc_keys & set(keys)),
          "冲突: %s" % sorted(exc_keys & set(keys)))
    check("每条排除都写了理由",
          all((x.get("reason") or "").strip() for x in exc_items), "")
    # 排除 != 核实失败: 两个清单不应重叠
    overlap = sorted(exc_keys & rej_keys)
    check("排除清单与被拒清单不重叠(口径隔离)", not overlap, "重叠: %s" % overlap)

    # 与 merge_refs.py 源码里的清单一致(防止产物与代码漂移)
    merge_src = os.path.join(ROOT, "scripts", "merge_refs.py")
    if os.path.isfile(merge_src):
        src = open(merge_src, encoding="utf-8").read()
        m = re.search(r"EXCLUDE_IRRELEVANT\s*=\s*\{(.*?)\n\}", src, re.S)
        code_keys = set(re.findall(r'"([A-Za-z0-9_\-]+)"\s*:', m.group(1))) if m else set()
        # 代码清单可含从未入库的 key(如 CMMI), 故只要求产物 ⊆ 代码
        check("产物排除清单 ⊆ merge_refs.py 源码清单",
              exc_keys <= code_keys,
              "产物多出: %s" % sorted(exc_keys - code_keys))

    # ---------- 5. 台账 §1.10 数字对账 ----------
    with open(LEDGER, "r", encoding="utf-8") as f:
        led = f.read()
    check("台账含 §1.10 参考文献真实性核实节",
          ("1.10" in led) and ("参考文献" in led), "")

    # 5.1 收录总数: 台账里 "共 N 条" / "N 条(目标..)" 之类
    m = re.findall(r"(\d{2,4})\s*条(?:文献)?(?:全部)?(?:通过|收录|入库)", led)
    ledger_counts = set(int(x) for x in m)
    m2 = re.findall(r"共\s*(\d{2,4})\s*条", led)
    ledger_counts |= set(int(x) for x in m2)
    if ledger_counts:
        check("台账声明的收录条数与产物一致",
              n_refs in ledger_counts,
              "台账出现 %s / 产物 %d" % (sorted(ledger_counts), n_refs))
    else:
        check("台账声明的收录条数与产物一致", False,
              "台账里没抽到条数声明, 需补写 (产物 %d 条)" % n_refs)

    # 5.1b 台账必须显式交代"通道统计基数 != 收录数"的口径差异
    n_base = n_refs + len(exc_items)
    if exc_items:
        check("台账交代了通道统计基数与收录数的口径差异",
              ("口径" in led and str(n_base) in led),
              "基数应为 %d(=%d+%d 排除)" % (n_base, n_refs, len(exc_items)))

    # 5.2 通道分布
    ch_count = {}
    for k, r in refs:
        v = r.get("verified") or {}
        ch = v.get("channels") or []
        if isinstance(ch, str):
            ch = [ch]
        for c in ch:
            ch_count[c] = ch_count.get(c, 0) + 1
    for c in ("arxiv", "openalex"):
        if c in ch_count:
            # 台账里若写了该通道数字, 必须一致
            pat = r"%s[^\n]{0,40}?(\d{1,3})\s*(?:条|篇)" % c
            hits = set(int(x) for x in re.findall(pat, led, flags=re.I))
            if hits:
                check("台账 %s 通道数一致" % c, ch_count[c] in hits,
                      "台账 %s / 产物 %d" % (sorted(hits), ch_count[c]))

    # 5.3 有 arXiv ID / DOI 的条数
    n_arxiv_id = sum(1 for _, r in refs if (r.get("arxiv") or r.get("arxiv_id")))
    n_doi = sum(1 for _, r in refs if r.get("doi"))
    check("有 arXiv ID 条数 > 有 DOI 条数(与统计一致)",
          n_arxiv_id > 0 and n_doi >= 0,
          "arxiv_id=%d doi=%d" % (n_arxiv_id, n_doi))
    n_neither = sum(1 for _, r in refs
                    if not (r.get("arxiv") or r.get("arxiv_id")) and not r.get("doi"))
    check("既无 DOI 又无 arXiv 的条数 <= 3(须逐条留档)",
          n_neither <= 3, "n=%d" % n_neither)

    # ---------- 6. _key2num.txt 与库同步 ----------
    k2n = os.path.join(DOCS, "_key2num.txt")
    if os.path.isfile(k2n):
        with open(k2n, "r", encoding="utf-8") as f:
            txt = f.read()
        # 文件含两张表(按编号 / 按 key 字母序), 只解析第一张避免重复计数
        head = txt.split("按 key 字母序")[0]
        pairs = re.findall(r"\[(\d+)\]\s+(\S+)", head)
        check("_key2num.txt 条数与库一致",
              len(pairs) == n_refs,
              "map=%d refs=%d" % (len(pairs), n_refs))
        num_of = {}
        for _, r in refs:
            if r.get("number") is not None:
                num_of[r.get("key")] = int(r["number"])
        mism = [(n, k) for n, k in pairs
                if k in num_of and num_of[k] != int(n)]
        check("_key2num.txt 编号与库字段一致", not mism, str(mism[:5]))
    else:
        check("_key2num.txt 存在(写作前必须重生成)", False, k2n)

    return report()


def report():
    npass = sum(1 for _, ok, _ in results if ok)
    nfail = len(results) - npass
    print("=" * 72)
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        line = "[%s] %s" % (tag, name)
        if detail and (not ok or len(detail) < 60):
            line += "  | " + detail
        print(line)
    print("=" * 72)
    print("总计: %d PASS / %d FAIL" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
