#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size").imageSize;
const {
  AlignmentType,
  Bookmark,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  LineRuleType,
  NumberFormat,
  PageBreak,
  PageNumber,
  PageReference,
  Packer,
  Paragraph,
  PositionalTab,
  PositionalTabAlignment,
  PositionalTabLeader,
  PositionalTabRelativeTo,
  ShadingType,
  Table,
  TableCell,
  TableOfContents,
  TableLayoutType,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const DOCS = path.join(ROOT, "docs");
const OUT_DIR = path.join(ROOT, "outputs", "thesis");
const OUT_DOCX = path.join(OUT_DIR, "GAAW_thesis_tongji.docx");
const TITLE = "点云上采样中重建—对抗梯度失衡的测量与自适应调节研究";
const BODY_WIDTH = 8500;

const CHAPTERS = [
  "ch1_introduction.md",
  "ch2_related_work.md",
  "ch3_analysis_framework.md",
  "ch4_research_design.md",
  "ch5_gaaw_method.md",
  "ch6_results.md",
  "ch7_conclusion.md",
].map((x) => path.join(DOCS, "chapters", x));

const thinBorder = { style: BorderStyle.SINGLE, size: 4, color: "666666" };
const tableBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder, insideHorizontal: thinBorder, insideVertical: thinBorder };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder };

function read(file) {
  return fs.readFileSync(file, "utf8").replace(/\r\n/g, "\n");
}

function cleanMarkdown(text) {
  return text.replace(/\\_/g, "_");
}

function latexToLinear(latex) {
  let s = latex.trim();
  const greek = {
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", Delta: "Δ", epsilon: "ε", varepsilon: "ε",
    theta: "θ", lambda: "λ", mu: "μ", rho: "ρ", sigma: "σ", tau: "τ",
    phi: "φ", psi: "ψ", omega: "ω", Theta: "Θ", mathcal: "", mathbf: "",
    mathrm: "", mathsf: "", operatorname: "", text: "",
  };
  s = s.replace(/\\(alpha|beta|gamma|delta|Delta|epsilon|varepsilon|theta|lambda|mu|rho|sigma|tau|phi|psi|omega|Theta)\b/g, (_, x) => greek[x]);
  s = s.replace(/\\operatorname\{([^{}]+)\}/g, "$1");
  s = s.replace(/\\(?:mathcal|mathbf|mathrm|mathsf|text)\{([^{}]+)\}/g, "$1");
  s = s.replace(/\\(?:mathcal|mathbf|mathrm|mathsf)\s*([A-Za-z])/g, "$1");
  const ops = [
    [/\\subseteq/g, "⊆"], [/\\approx/g, "≈"], [/\\times/g, "×"], [/\\infty/g, "∞"],
    [/\\nabla/g, "∇"], [/\\ldots/g, "…"], [/\\cdot/g, "·"], [/\\leq?|\\le/g, "≤"],
    [/\\geq?|\\ge/g, "≥"], [/\\ne/g, "≠"], [/\\in\b/g, "∈"], [/\\cup/g, "∪"],
    [/\\to/g, "→"], [/\\left|\\right/g, ""], [/\\langle/g, "⟨"], [/\\rangle/g, "⟩"],
    [/\\Vert|\\\|/g, "‖"], [/\\,/g, " "], [/\\!/g, ""], [/\\;/g, " "], [/\\quad/g, " "],
  ];
  s = s.replace(/\\sqrt\s*/g, "√");
  for (const [re, v] of ops) s = s.replace(re, v);
  for (let i = 0; i < 4; i++) {
    s = s.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1)/($2)");
    s = s.replace(/_\{([^{}]+)\}/g, "_$1");
    s = s.replace(/\^\{([^{}]+)\}/g, "^$1");
  }
  s = s.replace(/[{}]/g, "");
  s = s.replace(/\\([A-Za-z]+)/g, "$1");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function inlineRuns(text, base = {}) {
  const runs = [];
  const src = cleanMarkdown(text);
  const token = /(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$|\[(?:\d+)(?:[-,]\d+)*\])/g;
  let at = 0;
  let m;
  while ((m = token.exec(src)) !== null) {
    if (m.index > at) runs.push(new TextRun({ text: src.slice(at, m.index), ...base }));
    const value = m[0];
    if (value.startsWith("**")) {
      runs.push(new TextRun({ text: value.slice(2, -2), bold: true, ...base }));
    } else if (value.startsWith("`")) {
      runs.push(new TextRun({ text: value.slice(1, -1), font: { name: "Consolas", eastAsia: "宋体" }, size: 20 }));
    } else if (value.startsWith("$")) {
      runs.push(new TextRun({ text: latexToLinear(value.slice(1, -1)), font: "Cambria Math", size: base.size || 24 }));
    } else {
      runs.push(new TextRun({ text: value, superScript: true, size: 18, font: "宋体" }));
    }
    at = token.lastIndex;
  }
  if (at < src.length) runs.push(new TextRun({ text: src.slice(at), ...base }));
  return runs.length ? runs : [new TextRun({ text: "", ...base })];
}

function bodyParagraph(text, options = {}) {
  return new Paragraph({
    style: options.style || "BodyText",
    alignment: options.alignment,
    indent: options.noIndent ? { firstLine: 0 } : undefined,
    numbering: options.numbering,
    keepNext: options.keepNext,
    children: inlineRuns(text),
  });
}

function heading(level, text, pageBreakBefore = false) {
  const full = level === 1 && /^第\d+章\s+/.test(text)
    ? text.replace(/^(第\d+章)\s+/, "$1　")
    : text;
  return new Paragraph({
    text: full,
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    pageBreakBefore,
    keepNext: true,
  });
}

function bookmarkId(prefix, no) {
  return `${prefix}_${no.replace(".", "_")}`;
}

function captionParagraph(text, kind, no) {
  const prefix = kind === "figure" ? "fig" : "tab";
  return new Paragraph({
    style: "CaptionText",
    keepNext: kind === "table",
    children: [new Bookmark({ id: bookmarkId(prefix, no), children: inlineRuns(text, { size: 21 }) })],
  });
}

function pngForFigure(markdownPath, sourceFile) {
  const abs = path.resolve(path.dirname(sourceFile), markdownPath);
  const png = abs.replace(/\.svg$/i, ".png");
  const prepared = png.includes(`${path.sep}figures_schematic${path.sep}`)
    ? path.join(DOCS, "figures_word", path.basename(png))
    : png;
  if (!fs.existsSync(prepared)) throw new Error(`Figure PNG missing: ${prepared}`);
  return prepared;
}

function imageParagraph(file) {
  const data = fs.readFileSync(file);
  const dim = sizeOf(data);
  const maxW = 520;
  const maxH = 410;
  const scale = Math.min(maxW / dim.width, maxH / dim.height, 1);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 100, after: 80, line: 240, lineRule: LineRuleType.AUTO },
    children: [new ImageRun({ data, type: "png", transformation: { width: Math.round(dim.width * scale), height: Math.round(dim.height * scale) } })],
  });
}

function equationTable(no) {
  const manifest = JSON.parse(read(path.join(DOCS, "equations_thesis", "manifest.json")));
  const rec = manifest.find((x) => x.number === no);
  if (!rec) throw new Error(`Equation render missing: ${no}`);
  const file = path.join(ROOT, rec.png);
  const data = fs.readFileSync(file);
  const dim = sizeOf(data);
  // The middle cell is 6500 DXA (~4.51 in).  Keep the raster below that
  // physical width; 530 px overflowed the cell and clipped long equations.
  const scale = Math.min(425 / dim.width, 125 / dim.height, 1);
  const run = new ImageRun({ data, type: "png", transformation: { width: Math.round(dim.width * scale), height: Math.round(dim.height * scale) } });
  return new Table({
    width: { size: BODY_WIDTH, type: WidthType.DXA },
    columnWidths: [1000, 6500, 1000],
    borders: noBorders,
    rows: [new TableRow({
      cantSplit: true,
      children: [
        new TableCell({ width: { size: 1000, type: WidthType.DXA }, borders: noBorders, children: [new Paragraph("")] }),
        new TableCell({ width: { size: 6500, type: WidthType.DXA }, borders: noBorders, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80, line: 240, lineRule: LineRuleType.AUTO }, children: [run] })] }),
        new TableCell({ width: { size: 1000, type: WidthType.DXA }, borders: noBorders, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new Bookmark({ id: bookmarkId("eq", no), children: [new TextRun({ text: `（${no}）`, size: 24, font: "宋体" })] })] })] }),
      ],
    })],
  });
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((x) => x.trim());
}

function markdownTable(rows) {
  const clean = rows.filter((_, i) => i !== 1);
  const n = clean[0].length;
  const base = Math.floor(BODY_WIDTH / n);
  const widths = Array(n).fill(base);
  widths[n - 1] += BODY_WIDTH - base * n;
  const fontSize = n >= 8 ? 15 : n >= 6 ? 17 : 20;
  return new Table({
    width: { size: BODY_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    borders: tableBorders,
    rows: clean.map((cells, r) => new TableRow({
      cantSplit: true,
      tableHeader: r === 0,
      children: cells.map((cell, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
        margins: { top: 70, bottom: 70, left: 70, right: 70 },
        children: [new Paragraph({
          alignment: /^[-+—\d.±$]/.test(cell) ? AlignmentType.CENTER : AlignmentType.LEFT,
          spacing: { line: 260, lineRule: LineRuleType.EXACT, before: 0, after: 0 },
          children: inlineRuns(cell.replace(/^\*\*(.*)\*\*$/, "$1"), { bold: r === 0, size: fontSize, font: "宋体" }),
        })],
      })),
    })),
  });
}

function resolveCitations(text, citationOrder) {
  const map = new Map(citationOrder.map((k, i) => [k, i + 1]));
  return text.replace(/\{\{cite:([^}]+)\}\}/g, (_, key) => {
    if (!map.has(key)) throw new Error(`Unknown citation key: ${key}`);
    return `[${map.get(key)}]`;
  });
}

function parseMarkdown(text, sourceFile, state = {}) {
  const lines = text.split("\n");
  const children = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) { i++; continue; }

    if (line === "$$") {
      const block = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") block.push(lines[i++]);
      i++;
      const tag = block.join("\n").match(/\\tag\{(\d+\.\d+)\}/);
      if (!tag) throw new Error(`Equation without tag in ${sourceFile}`);
      children.push(equationTable(tag[1]));
      continue;
    }

    const hm = line.match(/^(#{1,3})\s+(.+)$/);
    if (hm) {
      const level = hm[1].length;
      const pageBreak = level === 1 && (state.hasHeading1 || false);
      children.push(heading(level, hm[2], pageBreak));
      if (level === 1) state.hasHeading1 = true;
      i++;
      continue;
    }

    const im = line.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
    if (im) {
      children.push(imageParagraph(pngForFigure(im[1], sourceFile)));
      i++;
      continue;
    }

    const figCap = line.match(/^图\s+(\d+\.\d+)\s+(.+)$/);
    if (figCap) {
      children.push(captionParagraph(`图 ${figCap[1]} ${figCap[2]}`, "figure", figCap[1]));
      i++;
      continue;
    }

    const tabCap = line.match(/^表\s+(\d+\.\d+)\s+(.+)$/);
    if (tabCap) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      if (j < lines.length && lines[j].trim().startsWith("|")) {
        children.push(captionParagraph(`表 ${tabCap[1]} ${tabCap[2]}`, "table", tabCap[1]));
        const rows = [];
        while (j < lines.length && lines[j].trim().startsWith("|")) rows.push(splitTableRow(lines[j++]));
        if (rows.length >= 2) children.push(markdownTable(rows));
        i = j;
        continue;
      }
    }

    if (line.startsWith("- ")) {
      children.push(bodyParagraph(line.slice(2), { noIndent: true, numbering: { reference: "bullet-list", level: 0 } }));
      i++;
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      children.push(bodyParagraph(line, { noIndent: true }));
      i++;
      continue;
    }

    if (line.startsWith(">")) {
      i++;
      continue;
    }

    children.push(bodyParagraph(line));
    i++;
  }
  return children;
}

function indexEntries(file, prefix) {
  const lines = read(file).split("\n").filter((x) => x.startsWith("- "));
  return lines.map((line) => {
    const text = line.slice(2);
    const m = text.match(/^(?:图\s+|表\s+|公式（)(\d+\.\d+)(?:）)?\s+(.+)$/);
    if (!m) throw new Error(`Bad index entry: ${line}`);
    const id = bookmarkId(prefix, m[1]);
    return new Paragraph({
      style: "IndexEntry",
      children: [
        new TextRun({ text, size: 24, font: "宋体" }),
        new TextRun({ children: [new PositionalTab({ alignment: PositionalTabAlignment.RIGHT, relativeTo: PositionalTabRelativeTo.MARGIN, leader: PositionalTabLeader.DOT })] }),
        new PageReference(id, { hyperlink: true }),
      ],
    });
  });
}

function footer() {
  return new Footer({
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "宋体" })] })],
  });
}

function coverChildren() {
  const spacer = (n) => Array.from({ length: n }, () => new Paragraph(""));
  return [
    ...spacer(2),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "硕士学位论文", bold: true, size: 44, font: "隶书" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "（学术学位）", bold: true, size: 32, font: "隶书" })] }),
    ...spacer(3),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 640 }, children: [new TextRun({ text: TITLE, bold: true, size: 36, font: "黑体" })] }),
    ...spacer(5),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "作者姓名：__________", size: 28, font: "宋体" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "学科专业：__________", size: 28, font: "宋体" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "指导教师：__________", size: 28, font: "宋体" })] }),
    ...spacer(3),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "同济大学", bold: true, size: 28, font: "黑体" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "二〇二六年八月", size: 24, font: "宋体" })] }),
  ];
}

function frontChildren(citationOrder) {
  const state = { hasHeading1: false };
  const zh = parseMarkdown(read(path.join(DOCS, "frontmatter", "abstract_zh.md")), path.join(DOCS, "frontmatter", "abstract_zh.md"), state);
  const en = parseMarkdown(read(path.join(DOCS, "frontmatter", "abstract_en.md")), path.join(DOCS, "frontmatter", "abstract_en.md"), state);
  const toc = [heading(1, "目录", true), new TableOfContents("", { hyperlink: true, headingStyleRange: "1-3" })];
  return [
    ...zh,
    ...en,
    ...toc,
    heading(1, "插图索引", true),
    ...indexEntries(path.join(DOCS, "indices", "FIGURE_INDEX.md"), "fig"),
    heading(1, "插表索引", true),
    ...indexEntries(path.join(DOCS, "indices", "TABLE_INDEX.md"), "tab"),
    heading(1, "公式索引", true),
    ...indexEntries(path.join(DOCS, "indices", "FORMULA_INDEX.md"), "eq"),
  ];
}

function mainChildren(citationOrder) {
  const state = { hasHeading1: false };
  const out = [];
  for (const chapter of CHAPTERS) {
    const text = resolveCitations(read(chapter), citationOrder);
    out.push(...parseMarkdown(text, chapter, state));
  }
  const manuscript = read(path.join(DOCS, "THESIS_MANUSCRIPT.md"));
  const refsPart = manuscript.split("# 参考文献\n")[1].split("\n# 致谢")[0].trim();
  out.push(heading(1, "参考文献", true));
  for (const para of refsPart.split(/\n\s*\n/)) {
    if (para.trim()) out.push(new Paragraph({
      style: "ReferenceText",
      children: [new TextRun({ text: para.trim(), size: 21, font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" } })],
    }));
  }
  const ack = read(path.join(DOCS, "backmatter", "acknowledgements.md"));
  out.push(...parseMarkdown(ack, path.join(DOCS, "backmatter", "acknowledgements.md"), state));
  return out;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const build = JSON.parse(read(path.join(DOCS, "_manuscript_build.json")));
  const citationOrder = build.citation_order;

  const doc = new Document({
    creator: "Codex",
    title: TITLE,
    description: "按同济大学学位论文模板规则生成的七章论文稿",
    numbering: {
      config: [{
        reference: "bullet-list",
        levels: [{ level: 0, format: "bullet", text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 240 } } } }],
      }],
    },
    styles: {
      default: {
        document: {
          run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" }, size: 24, color: "000000" },
          paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { line: 400, lineRule: LineRuleType.EXACT, before: 0, after: 0 } },
        },
      },
      paragraphStyles: [
        { id: "BodyText", name: "正文", basedOn: "Normal", next: "BodyText", quickFormat: true, run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" }, size: 24 }, paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 }, spacing: { line: 400, lineRule: LineRuleType.EXACT, before: 0, after: 0 }, widowControl: true } },
        { id: "Heading1", name: "章标题", basedOn: "Normal", next: "BodyText", quickFormat: true, run: { font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "黑体" }, size: 32, bold: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 300 }, outlineLevel: 0, keepNext: true } },
        { id: "Heading2", name: "一级节标题", basedOn: "Normal", next: "BodyText", quickFormat: true, run: { font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "黑体" }, size: 30, bold: false }, paragraph: { alignment: AlignmentType.LEFT, spacing: { before: 260, after: 140 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "二级节标题", basedOn: "Normal", next: "BodyText", quickFormat: true, run: { font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "黑体" }, size: 28, bold: false }, paragraph: { alignment: AlignmentType.LEFT, spacing: { before: 220, after: 120 }, outlineLevel: 2, keepNext: true } },
        { id: "CaptionText", name: "图表题", basedOn: "Normal", next: "BodyText", quickFormat: true, run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" }, size: 21 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { line: 240, lineRule: LineRuleType.AUTO, before: 60, after: 140 }, keepNext: true } },
        { id: "ReferenceText", name: "参考文献条目", basedOn: "Normal", next: "ReferenceText", quickFormat: true, run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" }, size: 21 }, paragraph: { alignment: AlignmentType.LEFT, indent: { left: 420, hanging: 420 }, spacing: { line: 320, lineRule: LineRuleType.EXACT, before: 0, after: 0 } } },
        { id: "IndexEntry", name: "索引条目", basedOn: "Normal", next: "IndexEntry", quickFormat: true, run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "宋体" }, size: 24 }, paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { line: 400, lineRule: LineRuleType.EXACT, before: 0, after: 0 } } },
      ],
    },
    sections: [
      {
        properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1700, right: 1700 } }, verticalAlign: VerticalAlign.CENTER },
        children: coverChildren(),
      },
      {
        properties: { type: "nextPage", page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1700, right: 1700 }, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
        footers: { default: footer() },
        children: frontChildren(citationOrder),
      },
      {
        properties: { type: "nextPage", page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1700, right: 1700 }, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
        footers: { default: footer() },
        children: mainChildren(citationOrder),
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_DOCX, buffer);
  process.stdout.write(`Wrote ${OUT_DOCX} (${buffer.length} bytes)\n`);
}

main().catch((err) => {
  process.stderr.write(`${err.stack || err}\n`);
  process.exit(1);
});
