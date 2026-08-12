# -*- coding: utf-8 -*-
"""
第 1、2 章成稿自检。红线:
  1. 全部 {{cite:KEY}} 的 KEY 必须存在于 REFERENCES.json
  2. 不得出现硬编号 [N]
  3. 不得出现无数字支撑的结果性表述（排除"禁令声明行"本身：以 '>' 开头或含"不使用"的行）
  4. 图中出现的方法名必须都能在引用库内找到对应条目
  5. 图表清单声明的图片文件必须真实存在
不硬编码任何"通过阈值"; 判定标准即"零违规"。
"""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAPTERS = ["docs/chapters/ch1_introduction.md", "docs/chapters/ch2_related_work.md"]
FIGDIR = "docs/figures_schematic"
BANNED = ["显著提升", "显著改善", "有效解决", "首次提出", "大幅", "明显优于", "远优于", "完美"]

# 图中可能出现的方法名 -> REFERENCES.json 中的 key
NAME2KEY = {
    "PointNet": "PointNet", "PointNet++": "PointNet2", "PointCNN": "PointCNN",
    "KPConv": "KPConv", "DGCNN": "DGCNN", "PCT": "PCT",
    "Point Transformer": "PointTransformer",
    "MLS": "MLS", "LOP": "LOP", "WLOP": "WLOP", "EAR": "EAR",
    "Deep Points\nConsolidation": "DeepPointsConsolidation",
    "PU-Net": "PUNet", "EC-Net": "EC-Net", "MPU": "MPU", "PU-GAN": "PUGAN",
    "PU-GCN": "PUGCN", "Dis-PU": "Dis-PU", "PU-Transformer": "PUTransformer",
    "Grad-PU": "Grad-PU", "PUDM": "PUDM", "RepKPU": "RepKPU",
}


def rd(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def main():
    fails = []
    refs = json.loads(rd("docs/REFERENCES.json"))["references"]
    keys = {x["key"] for x in refs}
    print("引用库条目数: %d" % len(refs))

    print("\n[1/5] cite 有效性 + 硬编号")
    for c in CHAPTERS:
        s = rd(c)
        cites = re.findall(r"\{\{cite:([^}]+)\}\}", s)
        bad = sorted({x for x in cites if x not in keys})
        hard = re.findall(r"(?<!\w)\[\d{1,3}\]", s)
        zh = len(re.findall(r"[\u4e00-\u9fff]", s))
        print("  %-24s cites=%3d uniq=%2d 中文=%5d invalid=%s hardnum=%d"
              % (os.path.basename(c), len(cites), len(set(cites)), zh, bad or "none", len(hard)))
        if bad:
            fails.append("%s: 无效 cite key %s" % (c, bad))
        if hard:
            fails.append("%s: 出现硬编号 %s" % (c, hard[:5]))

    print("\n[2/5] 违规结果性表述（跳过禁令声明行）")
    for c in CHAPTERS:
        hits = []
        for i, line in enumerate(rd(c).split("\n"), 1):
            if line.lstrip().startswith(">") or "不使用" in line:
                continue  # 禁令声明行本身允许引用这些词
            for b in BANNED:
                if b in line:
                    hits.append((i, b))
        print("  %-24s %s" % (os.path.basename(c), hits or "clean"))
        if hits:
            fails.append("%s: 违规表述 %s" % (c, hits))

    print("\n[3/5] 图中方法名 vs 引用库")
    src = rd("scripts/make_schematic_figures.py")
    # 按函数名精确切片: 函数在文件中的物理顺序不固定, 不能用 index 区间硬切。
    bodies = dict(re.findall(r"^def (fig_[0-9_a-z]+)\(\):\n(.*?)(?=^def |\Z)",
                             src, re.S | re.M))
    want = ["fig_1_1_task_illustration", "fig_1_2_roadmap", "fig_1_3_chapter_map",
            "fig_2_1_timeline", "fig_2_2_taxonomy", "fig_2_3_backbone_evolution",
            "fig_2_4_loss_quality_map"]
    absent = [w for w in want if w not in bodies]
    if absent:
        fails.append("出图脚本缺少函数: %s" % absent)
    seg = "\n".join(bodies.get(w, "") for w in want)
    found = sorted(n for n in NAME2KEY if n in seg)
    missing = [(n, NAME2KEY[n]) for n in found if NAME2KEY[n] not in keys]
    print("  扫描函数 %d 个, 抽出方法名 %d 个:" % (len(want) - len(absent), len(found)))
    print("    %s" % ", ".join(n.replace("\n", " ") for n in found))
    print("  引用库缺失: %s" % (missing or "none"))
    if not found:
        fails.append("方法名抽取为空，自检脚本本身失效")
    if missing:
        fails.append("图中方法名不在引用库: %s" % missing)

    print("\n[4/5] 表格结构")
    for c in CHAPTERS:
        s = rd(c)
        for m in re.finditer(r"\*\*(表 \d-\d)\s+([^*]+)\*\*", s):
            tag = m.group(1)
            rows = 0
            for line in s[m.end():].split("\n"):
                st = line.strip()
                if not st.startswith("|"):
                    if rows:
                        break
                    continue
                if set(st) <= set("|-: "):
                    continue
                rows += 1
            print("  %-8s %-28s 表头+数据行=%d -> 数据行=%d"
                  % (tag, m.group(2).strip()[:26], rows, rows - 1))
            if rows - 1 < 3:
                fails.append("%s 数据行不足 3 行" % tag)

    print("\n[5/5] 图表清单声明的文件是否存在")
    for c in CHAPTERS:
        for m in re.finditer(r"`(figures_schematic/[A-Za-z0-9_]+\.png)`", rd(c)):
            rel = "docs/" + m.group(1)
            ok = os.path.isfile(os.path.join(ROOT, rel))
            print("  %-52s %s" % (m.group(1), "OK" if ok else "MISSING"))
            if not ok:
                fails.append("清单声明的图片不存在: %s" % rel)

    print("\n" + "=" * 62)
    if fails:
        print("FAIL  共 %d 项:" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("PASS  全部红线通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
