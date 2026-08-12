# -*- coding: utf-8 -*-
"""
按 topic 分组导出第 2 章可用文献 (含 number/year/title), 供写作时逐条查表。
避免凭印象写 [N]。
"""
import json, os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
refs = d["references"]

# 第 2 章各节关心的 topic
SECTION = {
    "2.1 上采样方法": ["upsampling-core", "upsampling-gan", "upsampling-transformer",
                   "upsampling-arbitrary", "upsampling-diffusion", "upsampling-lidar",
                   "geometry-optim", "geometry-recon"],
    "2.2 骨干网络": ["point-backbone", "point-transformer", "transformer-core"],
    "2.3 GAN与生成": ["gan-core", "gan-sr", "gan-stability", "point-gen"],
    "2.4 多目标损失": ["loss-balance", "optimization", "metric-cd"],
    "其他(按需)": ["survey", "implicit", "denoise", "sampling", "dataset",
                "downstream", "completion", "toolchain", "statistics"],
}

lines = []
for sec, tops in SECTION.items():
    lines.append("=" * 78)
    lines.append("### " + sec)
    lines.append("=" * 78)
    for t in tops:
        got = sorted([r for r in refs if r.get("topic") == t],
                     key=lambda x: (int(x.get("year") or 9999), x["number"]))
        if not got:
            continue
        lines.append("")
        lines.append("-- topic: %s (%d 条) --" % (t, len(got)))
        for r in got:
            lines.append("   [%3d] %-4s %-22s %s"
                         % (r["number"], r.get("year"), r["key"],
                            (r.get("title") or "")[:62]))
    lines.append("")

txt = "\n".join(lines) + "\n"
out = os.path.join(ROOT, "docs", "_ch2_refs_by_topic.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(txt)
print(txt)
print("[written] %s" % out)
