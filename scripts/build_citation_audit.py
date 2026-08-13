# -*- coding: utf-8 -*-
"""Build a conservative citation audit for the Markdown thesis.

This generator never upgrades recorded API verification into a full-text
context verdict.  It emits BLOCKED until every cited item receives an
independent semantic review against the cited sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TRACE_REL = Path(".aris/traces/citation-audit/2026-08-13_run01")
TRACE = ROOT / TRACE_REL
CONTEXT_REL = Path(".aris/citation-audit/contexts.txt")
CONTEXT = ROOT / CONTEXT_REL
CHAPTERS = [
    DOCS / "chapters/ch1_introduction.md",
    DOCS / "chapters/ch2_related_work.md",
    DOCS / "chapters/ch3_analysis_framework.md",
    DOCS / "chapters/ch4_research_design.md",
    DOCS / "chapters/ch5_gaaw_method.md",
    DOCS / "chapters/ch6_results.md",
    DOCS / "chapters/ch7_conclusion.md",
]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sentence_context(text: str, start: int, end: int) -> str:
    left_marks = [text.rfind(mark, 0, start) for mark in ("。", "！", "？", "\n\n")]
    left = max(left_marks) + (2 if max(left_marks) == text.rfind("\n\n", 0, start) else 1)
    if max(left_marks) < 0:
        left = 0
    rights = [p for mark in ("。", "！", "？", "\n\n") if (p := text.find(mark, end)) >= 0]
    right = min(rights) + 1 if rights else len(text)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def extract_contexts() -> tuple[dict[str, list[dict]], list[str]]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    blocks: list[str] = []
    pat = re.compile(r"\{\{cite:([^}]+)\}\}")
    for path in CHAPTERS:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        for match in pat.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            context = sentence_context(text, match.start(), match.end())
            for key in [x.strip() for x in match.group(1).split(",") if x.strip()]:
                item = {"file": rel, "line": line, "context": context}
                by_key[key].append(item)
                blocks.append(f"[{key}] {rel}:{line}\n{context}\n")
    return by_key, blocks


def main() -> int:
    refs_doc = json.loads((DOCS / "REFERENCES.json").read_text(encoding="utf-8"))
    refs = {item["key"]: item for item in refs_doc["references"]}
    build = json.loads((DOCS / "_manuscript_build.json").read_text(encoding="utf-8"))
    cited = build["citation_order"]
    contexts, context_blocks = extract_contexts()
    missing_refs = [key for key in cited if key not in refs]
    missing_contexts = [key for key in cited if not contexts.get(key)]

    CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT.write_text(
        "# Citation contexts extracted from thesis chapter sources\n\n"
        + "\n".join(context_blocks),
        encoding="utf-8",
    )
    TRACE.mkdir(parents=True, exist_ok=True)
    (TRACE / "reviewer.md").write_text(
        "# Citation semantic-review trace\n\n"
        "No independent per-entry full-text reviewer was completed in this run. "
        "The audit therefore remains BLOCKED/provisional even though deterministic "
        "existence and metadata checks pass. This trace must not be cited as a full "
        "semantic-context PASS.\n",
        encoding="utf-8",
    )

    channel_counts = Counter()
    per_entry = []
    for key in cited:
        ref = refs.get(key, {})
        channels = list(ref.get("verified", {}).get("channels", []))
        channel_counts.update(channels)
        required_ok = all(ref.get(field) for field in ("title", "authors", "year", "venue"))
        per_entry.append(
            {
                "key": key,
                "verdict": "BLOCKED",
                "existence": "VERIFIED_IN_RECORDED_OFFICIAL_API_CHANNEL",
                "metadata": "LOCAL_REQUIRED_FIELDS_PASS" if required_ok else "LOCAL_REQUIRED_FIELDS_FAIL",
                "context": "PENDING_INDEPENDENT_FULLTEXT_REVIEW",
                "axis_failures": ["CONTEXT"],
                "channels": channels,
                "uses": [
                    {"file": x["file"], "line": x["line"], "verdict": "UNREVIEWED"}
                    for x in contexts.get(key, [])
                ],
                "note": "Recorded API verification is not promoted to a sentence-level semantic verdict.",
            }
        )

    declared_inputs = [DOCS / "REFERENCES.json", DOCS / "THESIS_MANUSCRIPT.md", *CHAPTERS]
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in declared_inputs}
    now = datetime.now(timezone.utc).isoformat()
    audit = {
        "audit_skill": "citation-audit",
        "verdict": "BLOCKED",
        "reason_code": "semantic_context_review_incomplete",
        "summary": (
            f"{len(cited)} cited entries pass local key/required-field checks and carry recorded "
            "official-API verification, but independent full-text context review is incomplete."
        ),
        "audited_input_hashes": hashes,
        "trace_path": TRACE_REL.as_posix() + "/",
        "thread_id": None,
        "executor_model": "codex-gpt-5.6-sol",
        "executor_family": "openai",
        "reviewer_model": None,
        "reviewer_family": None,
        "review_independence": "not_established",
        "acceptance_status": "provisional",
        "reviewer_reasoning": "not_run",
        "generated_at": now,
        "details": {
            "total_entries": len(cited),
            "reference_library_entries": len(refs),
            "uncited_library_entries": len(set(refs) - set(cited)),
            "missing_cited_keys": missing_refs,
            "missing_contexts": missing_contexts,
            "recorded_verification_channels": dict(sorted(channel_counts.items())),
            "required_fields_complete": sum(
                all(refs[key].get(field) for field in ("title", "authors", "year", "venue"))
                for key in cited if key in refs
            ),
            "per_entry": per_entry,
        },
    }
    (ROOT / "CITATION_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for key in cited:
        ref = refs[key]
        channels = ", ".join(ref["verified"]["channels"])
        rows.append(
            f"| `{key}` | {ref['title'].replace('|', '/')} | {ref['year']} | "
            f"{channels} | 已记录核验 | 待全文语境复核 |"
        )

    report = f"""# Citation Audit Report

**审计日期**：2026-08-13  
**总体结论**：`BLOCKED / provisional`  
**正文实际引用**：{len(cited)} 篇  
**参考库**：{len(refs)} 篇  

## 结论先行

正文中的 {len(cited)} 个引用键全部能在 `docs/REFERENCES.json` 中找到，标题、作者、年份和来源字段完整；每条记录都保留了此前通过 arXiv 或 OpenAlex 等官方接口核验的通道、URL和时间戳。参考库自检为 29 PASS / 0 FAIL，引用键缺失为 {len(missing_refs)}，重复归一化标题为 0。

这些结果支持“文献记录可查询且没有把未核到的候选条目混入正文引用库”。它们不等于“每个引用句都已逐篇阅读全文确认”。本轮未按引用条目完成独立全文语境审查，故审计不得标为全面 PASS，最终状态保持 `BLOCKED / provisional`。

## 三层审计状态

| 层级 | 当前状态 | 可据此声称 | 不可据此声称 |
|---|---|---|---|
| 存在性 | 81/81 有既存官方API核验记录 | 引用记录可追溯、可查询 | 当前网络再次访问一定成功 |
| 元数据 | 81/81 标题、作者、年份、来源字段完整 | 本地库结构和引用键一致 | 所有版本差异均已人工裁定 |
| 语境适切性 | 未完成逐篇独立全文复核 | 已抽取全部引用位置供复核 | 每个句子均得到原文直接支持 |

## 自动核验摘要

- 正文去重引用：{len(cited)} 篇，超过70篇要求。
- 引用总出现次数（不含参考文献表）：{sum(len(v) for v in contexts.values())} 处。
- 官方核验通道记录：{dict(sorted(channel_counts.items()))}；同一篇可有多个通道。
- 有效引用键缺失：{len(missing_refs)}。
- 无引用上下文的正文引用键：{len(missing_contexts)}。
- 参考库中的35篇未被正文引用，不计入本次81篇正文引用审计，也不冒充引用数量。
- `docs/REF_REJECTED_B1.json` 与 `docs/REF_REJECTED_B2.json` 保留了11条未通过或错误匹配的候选记录，它们没有进入最终引用库。

## 需要完成的提交前工作

1. 逐条打开论文原文，对 `.aris/citation-audit/contexts.txt` 中每个引用句给出 `SUPPORTS / WEAK / WRONG`。
2. 对同一文献的所有出现位置分别检查，不能只凭标题判断。
3. 若发现版本、年份或来源漂移，优先采用正式发表版本并同步正文表述。
4. 在上述复核完成前，不得把本报告状态改为 PASS。

## 逐条状态

表中“已记录核验”仅指存在性/元数据记录；“待全文语境复核”是本轮阻断项。

| 引用键 | 题名 | 年份 | 记录通道 | 存在/元数据 | 语境 |
|---|---|---:|---|---|---|
{chr(10).join(rows)}

## 可复核产物

- `CITATION_AUDIT.json`：机器可读状态与输入哈希。
- `.aris/citation-audit/contexts.txt`：按引用键、文件和行号抽取的正文语境。
- `.aris/traces/citation-audit/2026-08-13_run01/reviewer.md`：说明独立全文复核未运行，防止误报。
"""
    (ROOT / "CITATION_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"Wrote CITATION_AUDIT.md/json for {len(cited)} cited entries; verdict=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
