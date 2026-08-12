# -*- coding: utf-8 -*-
"""
逐条核实参考文献真实性。
输入: docs/REF_CANDIDATES.json  (候选清单, 人工/模型给出)
输出: docs/REFERENCES.json      (只保留核实通过的条目, 带核实来源 URL + 时间戳)
      docs/REF_REJECTED.json    (核不到 / 字段不符的条目, 如实留档)

核实通道 (只用官方 API, 不用搜索引擎摘要):
  - arXiv:   http://export.arxiv.org/api/query?id_list=XXXX.XXXXX
  - Crossref: https://api.crossref.org/works/<DOI>
  - DBLP:    https://dblp.org/search/publ/api?q=<title>&format=json   (会议论文兜底)

红线: 任何一条核不到, 直接进 REJECTED, 绝不留在 REFERENCES.json 里。
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
CAND = os.path.join(DOCS, "REF_CANDIDATES.json")
OUT = os.path.join(DOCS, "REFERENCES.json")
REJ = os.path.join(DOCS, "REF_REJECTED.json")

UA = "puv-net-thesis-ref-verifier/1.0 (academic reference verification)"
TIMEOUT = 30


def http_get(url, accept=None, retries=3):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", UA)
            if accept:
                req.add_header("Accept", accept)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            code = getattr(e, "code", None)
            # 404 = 确实不存在, 不重试; 5xx/超时 = 服务端问题, 退避重试
            if code == 404:
                raise
            time.sleep(1.5 * (k + 1))
    raise last


def norm_title(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def title_match(a, b):
    """标题匹配: 归一化后完全相等, 或一方包含另一方 (处理副标题差异)"""
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False, 0.0
    if na == nb:
        return True, 1.0
    if na in nb or nb in na:
        return True, 0.9
    # token Jaccard 兜底
    sa, sb = set(na.split()), set(nb.split())
    j = len(sa & sb) / max(1, len(sa | sb))
    return (j >= 0.8), j


# ---------------- arXiv ----------------
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def verify_arxiv(arxiv_id):
    url = "http://export.arxiv.org/api/query?id_list=%s&max_results=1" % urllib.parse.quote(arxiv_id)
    try:
        raw = http_get(url)
    except Exception as e:
        return None, "arxiv_http_error: %s" % e
    try:
        root = ET.fromstring(raw)
    except Exception as e:
        return None, "arxiv_xml_error: %s" % e
    entries = root.findall("a:entry", ARXIV_NS)
    if not entries:
        return None, "arxiv_no_entry"
    e = entries[0]
    idtxt = (e.findtext("a:id", "", ARXIV_NS) or "").strip()
    if "arxiv.org/abs" not in idtxt:
        return None, "arxiv_entry_not_paper"
    title = " ".join((e.findtext("a:title", "", ARXIV_NS) or "").split())
    if not title:
        return None, "arxiv_no_title"
    authors = [" ".join((a.findtext("a:name", "", ARXIV_NS) or "").split())
               for a in e.findall("a:author", ARXIV_NS)]
    published = (e.findtext("a:published", "", ARXIV_NS) or "").strip()
    year = published[:4] if published else ""
    doi = (e.findtext("a:doi", "", ARXIV_NS) or "").strip() or None
    return {
        "channel": "arxiv",
        "verify_url": url,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "arxiv_abs": idtxt,
    }, None


# ---------------- Crossref ----------------
def verify_crossref(doi):
    url = "https://api.crossref.org/works/%s" % urllib.parse.quote(doi, safe="/")
    try:
        raw = http_get(url, accept="application/json")
    except Exception as e:
        return None, "crossref_http_error: %s" % e
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, "crossref_json_error: %s" % e
    if d.get("status") != "ok":
        return None, "crossref_status_%s" % d.get("status")
    m = d.get("message") or {}
    titles = m.get("title") or []
    if not titles:
        return None, "crossref_no_title"
    title = " ".join(titles[0].split())
    authors = []
    for a in (m.get("author") or []):
        nm = ((a.get("given") or "") + " " + (a.get("family") or "")).strip()
        if nm:
            authors.append(nm)
    year = ""
    for k in ("published-print", "published-online", "published", "issued", "created"):
        p = m.get(k) or {}
        dp = p.get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            year = str(dp[0][0])
            break
    venue = ""
    ct = m.get("container-title") or []
    if ct:
        venue = " ".join(ct[0].split())
    elif m.get("event", {}).get("name"):
        venue = m["event"]["name"]
    return {
        "channel": "crossref",
        "verify_url": url,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": (m.get("DOI") or doi),
        "venue_api": venue,
        "type": m.get("type", ""),
    }, None


# ---------------- DBLP ----------------
def verify_dblp(title):
    url = "https://dblp.org/search/publ/api?q=%s&format=json&h=5" % urllib.parse.quote(title)
    try:
        raw = http_get(url, accept="application/json")
    except Exception as e:
        return None, "dblp_http_error: %s" % e
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, "dblp_json_error: %s" % e
    hits = (((d.get("result") or {}).get("hits") or {}).get("hit") or [])
    if not hits:
        return None, "dblp_no_hit"
    for h in hits:
        info = h.get("info") or {}
        t = " ".join((info.get("title") or "").rstrip(".").split())
        ok, score = title_match(title, t)
        if not ok:
            continue
        au = info.get("authors", {}).get("author", [])
        if isinstance(au, dict):
            au = [au]
        authors = [(x.get("text") if isinstance(x, dict) else str(x)) for x in au]
        return {
            "channel": "dblp",
            "verify_url": url,
            "title": t,
            "authors": authors,
            "year": str(info.get("year", "")),
            "doi": info.get("doi"),
            "venue_api": info.get("venue", ""),
            "dblp_url": info.get("url", ""),
            "match_score": round(score, 3),
        }, None
    return None, "dblp_title_mismatch"


# ---------------- OpenAlex (DBLP 兜底) ----------------
def verify_openalex(title):
    url = ("https://api.openalex.org/works?filter=title.search:%s&per-page=5"
           % urllib.parse.quote(title))
    try:
        raw = http_get(url, accept="application/json")
    except Exception as e:
        return None, "openalex_http_error: %s" % e
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, "openalex_json_error: %s" % e
    results = d.get("results") or []
    if not results:
        return None, "openalex_no_hit"
    for w in results:
        t = " ".join((w.get("title") or w.get("display_name") or "").split())
        ok, score = title_match(title, t)
        if not ok:
            continue
        authors = []
        for a in (w.get("authorships") or []):
            nm = ((a.get("author") or {}).get("display_name") or "").strip()
            if nm:
                authors.append(nm)
        doi = w.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        loc = (w.get("primary_location") or {}).get("source") or {}
        return {
            "channel": "openalex",
            "verify_url": url,
            "title": t,
            "authors": authors,
            "year": str(w.get("publication_year") or ""),
            "doi": doi or None,
            "venue_api": loc.get("display_name", "") or "",
            "openalex_id": w.get("id", ""),
            "match_score": round(score, 3),
        }, None
    return None, "openalex_title_mismatch"


def main():
    # CLI: verify_refs.py [候选文件] [接受输出] [拒绝输出]
    cand_path = sys.argv[1] if len(sys.argv) > 1 else CAND
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT
    rej_path = sys.argv[3] if len(sys.argv) > 3 else REJ
    if not os.path.isabs(cand_path):
        cand_path = os.path.join(ROOT, cand_path)
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    if not os.path.isabs(rej_path):
        rej_path = os.path.join(ROOT, rej_path)

    if not os.path.exists(cand_path):
        print("[FATAL] 候选清单不存在: %s" % cand_path)
        return 2
    with open(cand_path, "r", encoding="utf-8") as f:
        cands = json.load(f)
    if isinstance(cands, dict):
        cands = cands.get("candidates", [])

    accepted, rejected = [], []
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    for i, c in enumerate(cands, 1):
        key = c.get("key") or ("REF%03d" % i)
        title = (c.get("title") or "").strip()
        arxiv_id = (c.get("arxiv") or "").strip()
        doi = (c.get("doi") or "").strip()
        print("[%3d/%3d] %s :: %s" % (i, len(cands), key, title[:60]))

        checks = []
        errs = []

        if arxiv_id:
            r, e = verify_arxiv(arxiv_id)
            if r:
                checks.append(r)
            else:
                errs.append(e)
            time.sleep(0.6)
        if doi:
            r, e = verify_crossref(doi)
            if r:
                checks.append(r)
            else:
                errs.append(e)
            time.sleep(0.4)
        if not checks and title:
            r, e = verify_dblp(title)
            if r:
                checks.append(r)
            else:
                errs.append(e)
            time.sleep(0.6)
        if not checks and title:
            r, e = verify_openalex(title)
            if r:
                checks.append(r)
            else:
                errs.append(e)
            time.sleep(0.4)

        if not checks:
            rejected.append({"key": key, "input": c, "errors": errs,
                             "reason": "no_channel_verified", "checked_at": ts})
            print("      -> REJECT (%s)" % "; ".join(errs))
            continue

        # 标题一致性: 至少一个通道的标题与候选标题匹配
        tm = [title_match(title, ch["title"]) for ch in checks]
        if not any(ok for ok, _ in tm):
            rejected.append({"key": key, "input": c,
                             "api_titles": [ch["title"] for ch in checks],
                             "reason": "title_mismatch",
                             "scores": [round(s, 3) for _, s in tm],
                             "checked_at": ts})
            print("      -> REJECT title_mismatch: api=%r" % checks[0]["title"][:70])
            continue

        # 年份一致性: 候选给了年份就必须与任一通道一致 (容忍 arXiv preprint 早一年)
        cy = str(c.get("year") or "").strip()
        api_years = [ch.get("year", "") for ch in checks if ch.get("year")]
        year_ok = True
        if cy and api_years:
            year_ok = any(abs(int(y) - int(cy)) <= 1 for y in api_years if y.isdigit())
        if not year_ok:
            rejected.append({"key": key, "input": c, "api_years": api_years,
                             "reason": "year_mismatch", "checked_at": ts})
            print("      -> REJECT year_mismatch: cand=%s api=%s" % (cy, api_years))
            continue

        prim = checks[0]
        accepted.append({
            "key": key,
            "title": prim["title"],
            "authors": prim.get("authors", []),
            "year": prim.get("year") or cy,
            "venue": c.get("venue") or prim.get("venue_api", ""),
            "doi": next((ch.get("doi") for ch in checks if ch.get("doi")), None),
            "arxiv": arxiv_id or None,
            "arxiv_abs": next((ch.get("arxiv_abs") for ch in checks if ch.get("arxiv_abs")), None),
            "topic": c.get("topic", ""),
            "cite_in_chapter": c.get("cite_in_chapter", []),
            "verified": {
                "channels": [ch["channel"] for ch in checks],
                "verify_urls": [ch["verify_url"] for ch in checks],
                "checked_at": ts,
            },
        })
        print("      -> OK via %s" % ",".join(ch["channel"] for ch in checks))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "每条均经 arXiv/Crossref/DBLP/OpenAlex 官方 API 核实; 核不到的条目在对应 REJECTED 文件",
            "generated_at": ts,
            "source_candidates": os.path.basename(cand_path),
            "n_accepted": len(accepted),
            "references": accepted,
        }, f, ensure_ascii=False, indent=2)
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": ts, "n_rejected": len(rejected),
                   "rejected": rejected}, f, ensure_ascii=False, indent=2)

    print("\n=== 核实汇总 ===")
    print("ACCEPTED : %d  -> %s" % (len(accepted), out_path))
    print("REJECTED : %d  -> %s" % (len(rejected), rej_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
