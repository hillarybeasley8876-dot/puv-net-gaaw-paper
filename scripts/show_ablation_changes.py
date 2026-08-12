# -*- coding: utf-8 -*-
"""列出当前 8 组消融的真实改动 —— 用于核实"创新点在代码里到底存在什么"。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "runs" / "ablation_design" / "ablation_matrix.json")
               .read_text(encoding="utf-8"))
for name, g in d["groups"].items():
    print("%-18s %s" % (name, g["desc"]))
    for k, v in g["changes"].items():
        print("      %s: %s -> %s" % (k, v["from"], v["to"]))
    for k, v in (g.get("explicit_but_unchanged") or {}).items():
        print("      (同值) %s = %s" % (k, v))
