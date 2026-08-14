# -*- coding: utf-8 -*-
"""
修正 REFERENCES.json 中 [1] StatPower 的作者顺序。

问题：authors 字段为 ['Keith E. Muller', 'Jacob Cohen']，
      而《Statistical Power Analysis for the Behavioral Sciences》(Routledge)
      的原著者是 Jacob Cohen 独著；Keith E. Muller 是该书在 Technometrics 上的
      书评作者。OpenAlex 的记录把书评与原著合并，导致作者顺序颠倒。

依据：该条目 verified.channels = ['openalex']，仅单通道核验；
      DOI 10.2307/1270020 指向 Technometrics 的书评记录（JSTOR），
      而非 Routledge 原书。

处理：把 authors 改为 ['Jacob Cohen']，并在 note 字段记录本次修正的理由与证据，
      使改动可追溯。不改 number（避免全文编号平移）。
      DOI 保留但在 note 中标注其指向书评而非原书。

安全设计：改前备份，改后回读校验字段值。
"""
import json, io, os, shutil, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "REFERENCES.json")

d = json.load(io.open(SRC, encoding="utf-8"))
target = None
for r in d["references"]:
    if r["key"] == "StatPower":
        target = r
        break

if target is None:
    print("[FATAL] 未找到 key=StatPower")
    sys.exit(1)

print("=" * 74)
print("修正前")
print("=" * 74)
print(f"  authors: {target.get('authors')}")
print(f"  venue  : {target.get('venue')}")
print(f"  doi    : {target.get('doi')}")
print(f"  note   : {target.get('note', '(无)')}")

OLD_AUTHORS = ["Keith E. Muller", "Jacob Cohen"]
NEW_AUTHORS = ["Jacob Cohen"]

if target.get("authors") != OLD_AUTHORS:
    print()
    print(f"  [SKIP] authors 当前值不等于预期的待修值，不做改动。")
    print(f"         预期 {OLD_AUTHORS}")
    print(f"         实际 {target.get('authors')}")
    sys.exit(0)

stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z") or datetime.now().isoformat()
note_add = (
    f"[{datetime.now().strftime('%Y-%m-%d')} 作者字段修正] "
    f"原 authors 为 {OLD_AUTHORS}，改为 {NEW_AUTHORS}。"
    "理由：本书原著者为 Jacob Cohen 独著（Routledge）；Keith E. Muller 是该书在 "
    "Technometrics 上的书评作者。本条目 verified 仅经 openalex 单通道核验，"
    "该库把书评记录与原著合并，导致作者顺序颠倒。"
    "另注：doi 10.2307/1270020 指向 JSTOR 上的 Technometrics 书评记录而非 Routledge 原书，"
    "正文引用该条目时其语义为「统计功效分析的经典论述」，与书评无关。"
)

shutil.copyfile(SRC, SRC + ".bak_statpower")
target["authors"] = NEW_AUTHORS
prev = target.get("note")
target["note"] = (prev + " | " + note_add) if prev else note_add

io.open(SRC, "w", encoding="utf-8").write(
    json.dumps(d, ensure_ascii=False, indent=2) + "\n")

# 回读校验
d2 = json.load(io.open(SRC, encoding="utf-8"))
chk = next(r for r in d2["references"] if r["key"] == "StatPower")
print()
print("=" * 74)
print("修正后（回读校验）")
print("=" * 74)
print(f"  authors: {chk.get('authors')}")
print(f"  note   : {chk.get('note')[:150]}…")
print()
ok = chk.get("authors") == NEW_AUTHORS and "作者字段修正" in (chk.get("note") or "")
# 顺带确认条目总数与编号未变
same_n = len(d2["references"]) == len(d["references"])
same_num = chk["number"] == target["number"]
print(f"  条目总数不变: {same_n}   编号不变: {same_num}（number={chk['number']}）")
print()
print("  PASS  已修正并留痕" if (ok and same_n and same_num) else "  FAIL  校验不通过")
if not (ok and same_n and same_num):
    sys.exit(1)
print(f"  备份: {SRC}.bak_statpower")
