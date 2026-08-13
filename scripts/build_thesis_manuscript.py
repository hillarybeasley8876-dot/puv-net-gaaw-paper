#!/usr/bin/env python3
"""Build the auditable Markdown thesis and its figure/table/formula indices.

The script intentionally keeps a fixed chapter order and resolves {{cite:KEY}}
markers by first appearance.  It does not invent missing bibliographic fields.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHAPTERS = [
    DOCS / "chapters" / "ch1_introduction.md",
    DOCS / "chapters" / "ch2_related_work.md",
    DOCS / "chapters" / "ch3_analysis_framework.md",
    DOCS / "chapters" / "ch4_research_design.md",
    DOCS / "chapters" / "ch5_gaaw_method.md",
    DOCS / "chapters" / "ch6_results.md",
    DOCS / "chapters" / "ch7_conclusion.md",
]

FORMULA_TITLES = {
    "1.1": "固定权重生成器目标",
    "1.2": "重建梯度与对抗梯度",
    "1.3": "梯度自适应对抗权重",
    "1.4": "点云上采样映射",
    "2.1": "双向Chamfer距离",
    "2.2": "统一多损失目标",
    "2.3": "损失梯度合成",
    "3.1": "4倍点云上采样任务",
    "3.2": "点云质量向量",
    "3.3": "双向Chamfer距离",
    "3.4": "生成器总损失",
    "3.5": "Hausdorff距离",
    "3.6": "最近邻间距变异系数",
    "3.7": "重建—对抗梯度比例",
    "3.8": "重建—对抗梯度余弦相似度",
    "4.1": "相对变化率",
    "4.2": "固定样本配对差值",
    "4.3": "seed层均值与标准误",
    "4.4": "seed内方法效应",
    "5.1": "判别器hinge损失",
    "5.2": "生成器对抗损失",
    "5.3": "固定对抗权重目标",
    "5.4": "VQGAN自适应判别权重",
    "5.5": "共同参数集合上的两条梯度",
    "5.6": "全参数梯度范数",
    "5.7": "目标梯度比例约束",
    "5.8": "GAAW动态权重",
    "5.9": "对抗权重线性引入因子",
    "5.10": "GAAW生成器目标",
    "5.11": "合成梯度范数与夹角",
    "5.12": "对抗损失常数缩放抵消",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_assets(text: str):
    lines = text.splitlines()
    figures: list[tuple[str, str, str]] = []
    tables: list[tuple[str, str]] = []
    formulas: list[tuple[str, str]] = []

    for i, line in enumerate(lines):
        fm = re.match(r"^!\[[^]]*\]\(([^)]+)\)\s*$", line.strip())
        if fm:
            for j in range(i + 1, min(i + 4, len(lines))):
                if not lines[j].strip():
                    continue
                cm = re.match(r"^图\s+(\d+\.\d+)\s+(.+)$", lines[j].strip())
                if cm:
                    figures.append((cm.group(1), cm.group(2).strip(), fm.group(1)))
                break

        tm = re.match(r"^表\s+(\d+\.\d+)\s+(.+)$", line.strip())
        if tm:
            nxt = ""
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip():
                    nxt = lines[j].strip()
                    break
            if nxt.startswith("|"):
                tables.append((tm.group(1), tm.group(2).strip()))

    for m in re.finditer(r"\\tag\{(\d+\.\d+)\}", text):
        no = m.group(1)
        formulas.append((no, FORMULA_TITLES.get(no, "数学模型")))

    return figures, tables, formulas


def unique_and_contiguous(items: list[tuple], label: str) -> None:
    numbers = [x[0] for x in items]
    if len(numbers) != len(set(numbers)):
        dup = sorted({x for x in numbers if numbers.count(x) > 1})
        raise RuntimeError(f"{label}编号重复: {dup}")
    by_chapter: dict[int, list[int]] = {}
    for no in numbers:
        ch, seq = map(int, no.split("."))
        by_chapter.setdefault(ch, []).append(seq)
    for ch, seqs in by_chapter.items():
        expected = list(range(1, len(seqs) + 1))
        if seqs != expected:
            raise RuntimeError(f"第{ch}章{label}编号不连续或未按出现顺序: {seqs}")


def reference_type(ref: dict) -> str:
    venue = str(ref.get("venue") or "").lower()
    if "routledge" in venue or "press" in venue or "book" in venue:
        return "M"
    if "arxiv" in venue or (not venue and ref.get("arxiv")):
        return "EB/OL"
    conferences = (
        "cvpr", "iccv", "eccv", "neurips", "nips", "iclr", "aaai", "ijcai",
        "acm mm", "siggraph", "3dv", "icml", "wacv", "iros", "icra",
    )
    if any(x in venue for x in conferences):
        return "C"
    return "J"


def format_reference(n: int, ref: dict) -> str:
    authors = ref.get("authors") or []
    author_text = ", ".join(str(x) for x in authors) if authors else "作者信息见原始记录"
    title = str(ref.get("title") or "").strip()
    venue = str(ref.get("venue") or "").strip()
    year = str(ref.get("year") or "").strip()
    typ = reference_type(ref)
    base = f"[{n}] {author_text}. {title}[{typ}]."
    if venue:
        base += f" {venue}，{year}."
    elif year:
        base += f" {year}."
    doi = ref.get("doi")
    arxiv = ref.get("arxiv")
    if doi:
        base += f" DOI: {doi}."
    elif arxiv:
        base += f" arXiv: {arxiv}."
    return base


def main() -> None:
    for path in CHAPTERS:
        if not path.exists():
            raise FileNotFoundError(path)

    chapter_texts = [read(p) for p in CHAPTERS]
    all_text = "\n\n".join(chapter_texts)

    ref_data = json.loads(read(DOCS / "REFERENCES.json"))
    refs = {r["key"]: r for r in ref_data["references"]}
    citation_order: list[str] = []
    for match in re.finditer(r"\{\{cite:([^}]+)\}\}", all_text):
        key = match.group(1)
        if key not in refs:
            raise RuntimeError(f"未解析引用键: {key}")
        if key not in citation_order:
            citation_order.append(key)
    if len(citation_order) < 70:
        raise RuntimeError(f"正文仅使用{len(citation_order)}篇文献，未达到70篇")
    cite_num = {key: i + 1 for i, key in enumerate(citation_order)}

    def resolve_citations(text: str) -> str:
        return re.sub(
            r"\{\{cite:([^}]+)\}\}",
            lambda m: f"[{cite_num[m.group(1)]}]",
            text,
        )

    figures: list[tuple[str, str, str]] = []
    tables: list[tuple[str, str]] = []
    formulas: list[tuple[str, str]] = []
    for text in chapter_texts:
        f, t, e = extract_assets(text)
        figures.extend(f)
        tables.extend(t)
        formulas.extend(e)
    unique_and_contiguous(figures, "图")
    unique_and_contiguous(tables, "表")
    unique_and_contiguous(formulas, "公式")

    idx_dir = DOCS / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    index_text = lambda title: title.replace("$", "")
    figure_md = "# 插图索引\n\n" + "\n".join(
        f"- 图 {no} {index_text(title)}" for no, title, _ in figures
    ) + "\n"
    table_md = "# 插表索引\n\n" + "\n".join(
        f"- 表 {no} {index_text(title)}" for no, title in tables
    ) + "\n"
    formula_md = "# 公式索引\n\n" + "\n".join(
        f"- 公式（{no}） {index_text(title)}" for no, title in formulas
    ) + "\n"
    citation_md = "# 引用编号映射\n\n" + "\n".join(
        f"- [{i}] `{key}` — {refs[key]['title']}" for i, key in enumerate(citation_order, 1)
    ) + "\n"
    (idx_dir / "FIGURE_INDEX.md").write_text(figure_md, encoding="utf-8")
    (idx_dir / "TABLE_INDEX.md").write_text(table_md, encoding="utf-8")
    (idx_dir / "FORMULA_INDEX.md").write_text(formula_md, encoding="utf-8")
    (idx_dir / "CITATION_MAP.md").write_text(citation_md, encoding="utf-8")

    bibliography = "# 参考文献\n\n" + "\n\n".join(
        format_reference(i, refs[key]) for i, key in enumerate(citation_order, 1)
    )
    parts = [
        "# 点云上采样中重建—对抗梯度失衡的测量与自适应调节研究",
        read(DOCS / "frontmatter" / "abstract_zh.md"),
        read(DOCS / "frontmatter" / "abstract_en.md"),
        "# 目录\n\n（Word版本使用自动目录字段生成。）",
        figure_md.strip(),
        table_md.strip(),
        formula_md.strip(),
        *(resolve_citations(x) for x in chapter_texts),
        bibliography,
        read(DOCS / "backmatter" / "acknowledgements.md"),
    ]
    manuscript = "\n\n".join(parts).strip() + "\n"
    out = DOCS / "THESIS_MANUSCRIPT.md"
    out.write_text(manuscript, encoding="utf-8")

    summary = {
        "title": "点云上采样中重建—对抗梯度失衡的测量与自适应调节研究",
        "chapters": [str(p.relative_to(ROOT)).replace("\\", "/") for p in CHAPTERS],
        "figures": len(figures),
        "tables": len(tables),
        "formulas": len(formulas),
        "citations": len(citation_order),
        "citation_order": citation_order,
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
    }
    (DOCS / "_manuscript_build.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {out}\nfigures={len(figures)} tables={len(tables)} "
        f"formulas={len(formulas)} cited_references={len(citation_order)}"
    )


if __name__ == "__main__":
    main()
