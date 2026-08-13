#!/usr/bin/env node
"use strict";

// Render the thesis LaTeX blocks through MathJax and convert the resulting
// self-contained SVG to high-resolution PNG for Word insertion.

const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..");
const CHAPTERS = [
  "ch1_introduction.md",
  "ch2_related_work.md",
  "ch3_analysis_framework.md",
  "ch4_research_design.md",
  "ch5_gaaw_method.md",
  "ch6_results.md",
  "ch7_conclusion.md",
].map((x) => path.join(ROOT, "docs", "chapters", x));
const OUT = path.join(ROOT, "docs", "equations_thesis");

function htmlEscape(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function extractEquations() {
  const out = [];
  for (const file of CHAPTERS) {
    const text = fs.readFileSync(file, "utf8");
    const re = /\$\$([\s\S]*?)\$\$/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const tag = m[1].match(/\\tag\{(\d+\.\d+)\}/);
      if (!tag) continue;
      const latex = m[1].replace(/\\tag\{\d+\.\d+\}/g, "").trim();
      out.push({ number: tag[1], latex, source: path.relative(ROOT, file).replaceAll("\\", "/") });
    }
  }
  return out;
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const equations = extractEquations();
  const html = `<!doctype html><html><head><meta charset="utf-8">
  <script>window.MathJax={tex:{packages:{'[+]':['ams']}},svg:{fontCache:'local'},startup:{typeset:true}};</script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>body{margin:0;background:white;color:black}.eq{display:inline-block;padding:8px 14px;font-size:24px;white-space:nowrap}</style>
  </head><body>${equations.map((e, i) => `<div class="eq" id="eq-${i}">\\[${htmlEscape(e.latex)}\\]</div><br>`).join("")}</body></html>`;

  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const page = await browser.newPage({ viewport: { width: 2400, height: 1600 }, deviceScaleFactor: 1 });
  await page.setContent(html, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise, null, { timeout: 120000 });
  await page.evaluate(() => MathJax.startup.promise);

  const manifest = [];
  for (let i = 0; i < equations.length; i++) {
    const eq = equations[i];
    const svg = await page.locator(`#eq-${i} mjx-container > svg`).first().evaluate((node) => node.outerHTML);
    const stem = `E${eq.number.replace(".", "_")}`;
    const svgPath = path.join(OUT, `${stem}.svg`);
    const pngPath = path.join(OUT, `${stem}.png`);
    fs.writeFileSync(svgPath, svg, "utf8");
    const meta = await sharp(Buffer.from(svg), { density: 300 })
      .flatten({ background: "#ffffff" })
      .png()
      .toFile(pngPath);
    manifest.push({
      ...eq,
      svg: path.relative(ROOT, svgPath).replaceAll("\\", "/"),
      png: path.relative(ROOT, pngPath).replaceAll("\\", "/"),
      width: meta.width,
      height: meta.height,
    });
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  process.stdout.write(`Rendered ${manifest.length} equations to ${OUT}\n`);
}

main().catch((err) => {
  process.stderr.write(`${err.stack || err}\n`);
  process.exit(1);
});
