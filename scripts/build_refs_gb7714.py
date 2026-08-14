# -*- coding: utf-8 -*-
"""
按 GB/T 7714 生成参考文献表（同济学位论文格式）。

格式依据 docs/FORMAT_TONGJI.md §2.3（从同济写作示例实测抽取）：
  中文期刊: 作者. 题名[J]. 刊名，年，卷（期）：起页-止页.
  英文期刊: Authors. Title[J]. Journal，Year，Vol（Iss）：Pages.
  专著:     作者. 书名[M]. 地点：出版社，年.
  会议:     作者. 题名[C]//会议名. 地点：出版者，年：页码.
  预印本:   Authors. Title[EB/OL]. arXiv:XXXX.XXXXX，Year.
  注：示例中英文条目亦使用全角逗号与全角括号，本脚本从模板取值。

作者规则（GB/T 7714）：
  3 名以内全列；超过 3 名列前 3 名后加"等"（英文加 "et al."）。
  英文作者按"姓, 名首字母."格式。本库的 authors 字段为"名 姓"顺序，需转换。

数据质量标记：
  本脚本对可疑条目输出 WARN，不自动修改库。已知问题在输出末尾汇总。

产物：docs/REFERENCES_GB7714.md
"""
import json, os, io, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "REFERENCES.json")
OUT = os.path.join(ROOT, "docs", "REFERENCES_GB7714.md")

d = json.load(io.open(SRC, encoding="utf-8"))
refs = sorted(d["references"], key=lambda x: int(x["number"]))

CN = re.compile(r"[\u4e00-\u9fff]")


def fmt_author_en(name):
    """'Zhao Chen' -> 'Chen Z.'；已是 'Chen, Z.' 形式则原样。"""
    name = name.strip()
    if "," in name:
        return name
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    surname = parts[-1]
    initials = " ".join(p[0].upper() + "." for p in parts[:-1] if p)
    return f"{surname} {initials}"


def fmt_authors(authors):
    if not authors:
        return "佚名"
    is_cn = any(CN.search(a) for a in authors)
    if is_cn:
        names = [a.strip() for a in authors]
        tail = "等" if len(names) > 3 else ""
    else:
        names = [fmt_author_en(a) for a in authors]
        tail = "et al." if len(names) > 3 else ""
    keep = names[:3]
    s = "，".join(keep) if is_cn else ", ".join(keep)
    if tail:
        s += ("，" + tail) if is_cn else ", " + tail
    return s


def classify(r):
    """判断文献类型标识。"""
    venue = (r.get("venue") or "").lower()
    if r.get("arxiv") and r.get("arxiv") != "None" and not venue:
        return "EB/OL"
    if any(k in venue for k in ("conference", "cvpr", "iccv", "eccv", "neurips",
                                "icml", "iclr", "siggraph", "accv", "aaai",
                                "proceedings", "workshop", "symposium")):
        return "C"
    if any(k in venue for k in ("transactions", "journal", "letters", "tog",
                                "tpami", "tvcg", "tip", "pattern recognition")):
        return "J"
    if any(k in venue for k in ("press", "routledge", "springer", "wiley",
                                "mit ", "cambridge", "oxford", "出版社")):
        return "M"
    if venue.startswith("arxiv") or (r.get("arxiv") and r.get("arxiv") != "None"):
        return "EB/OL"
    if venue:
        return "J"
    return "EB/OL"


def build(r):
    a = fmt_authors(r.get("authors") or [])
    # 作者串若以缩写点结尾，后面不再补句点（避免 'Cohen J..'）
    a_sep = "" if a.endswith(".") else "."
    title = (r.get("title") or "").rstrip(".")
    year = r.get("year") or "出版年不详"
    venue = r.get("venue") or ""
    kind = classify(r)
    doi = r.get("doi")
    arx = r.get("arxiv")

    if kind == "EB/OL":
        tail = f"arXiv:{arx}" if arx and arx != "None" else (venue or "")
        s = f"{a}{a_sep} {title}[EB/OL]. {tail}，{year}."
    elif kind == "M":
        s = f"{a}{a_sep} {title}[M]. {venue}，{year}."
    elif kind == "C":
        s = f"{a}{a_sep} {title}[C]//{venue}，{year}."
    else:
        s = f"{a}{a_sep} {title}[J]. {venue}，{year}."
    if doi and doi != "None":
        s += f" DOI：{doi}."
    return s, kind


lines = []
lines.append("# 参考文献（GB/T 7714 格式）\n")
lines.append("")
lines.append("> 本表由 `scripts/build_refs_gb7714.py` 从 `docs/REFERENCES.json` 生成，"
             "格式依据 `docs/FORMAT_TONGJI.md §2.3`（从同济写作示例实测抽取）。")
lines.append("> 排版要求：宋体 10.5pt，行距固定值 16pt，悬挂缩进 21pt。")
lines.append("> **本文件为生成产物，不要手工编辑**；如需修正条目请改 `REFERENCES.json` 后重新生成。")
lines.append("")
# 编号空档说明：正文 [N] 已定稿，剔除未引用条目后不重编号，否则引用错位。
_nums = sorted(int(r["number"]) for r in refs)
_gaps = [n for n in range(_nums[0], _nums[-1] + 1) if n not in set(_nums)]
if _gaps:
    lines.append(f"> **关于编号不连续**：本表共 {len(refs)} 条，"
                 f"编号范围 {_nums[0]}–{_nums[-1]}，其中 {len(_gaps)} 个编号空缺。"
                 "原因是文献库初稿收录了部分正文最终未引用的条目，"
                 "按「参考文献须与正文引用一一对应」的规范予以剔除；"
                 "而正文中的引用编号已定稿，重新编号会造成引用错位，"
                 "故保留原编号并容许空档。"
                 "被剔除条目及其理由完整记录于 `docs/REFERENCES.json` 的 "
                 "`excluded_unused` 字段，可供核查。")
    lines.append(f"> 空缺编号：{', '.join(str(n) for n in _gaps)}。")
    lines.append("")
lines.append("---")
lines.append("")

warns = []
kinds = {}
for r in refs:
    s, kind = build(r)
    kinds[kind] = kinds.get(kind, 0) + 1
    lines.append(f"[{r['number']}] {s}")
    lines.append("")
    # 数据质量检查
    au = r.get("authors") or []
    if not au:
        warns.append((r["number"], r["key"], "无作者字段"))
    if not r.get("year"):
        warns.append((r["number"], r["key"], "无年份"))
    if not r.get("venue") and not r.get("arxiv"):
        warns.append((r["number"], r["key"], "既无 venue 也无 arxiv"))
    if r.get("venue") and len(str(r.get("venue"))) > 90:
        warns.append((r["number"], r["key"], "venue 过长，可能含冗余信息"))

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")

print("=" * 74)
print("GB/T 7714 参考文献表生成")
print("=" * 74)
print(f"  [written] {OUT}")
print(f"  条目数 {len(refs)}")
print(f"  类型分布: {kinds}")
print()
print("=" * 74)
print("数据质量告警（不自动修改库）")
print("=" * 74)
if warns:
    for n, k, w in warns:
        print(f"  [{n:>3}] {k:24s} {w}")
    print(f"  共 {len(warns)} 条")
else:
    print("  无")
print()
print("  首 4 条示例：")
for r in refs[:4]:
    s, _ = build(r)
    print(f"    [{r['number']}] {s[:118]}")
