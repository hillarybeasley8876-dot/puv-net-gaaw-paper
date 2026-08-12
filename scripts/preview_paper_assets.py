# -*- coding: utf-8 -*-
"""生成一套"合成消融组"的示意图，供人眼检查图表排版质量。

产物落 paper_assets_TRIAL/，仅用于版式审阅，不是论文数字。
每张图的 .data.json 里会写明 synthetic=true。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "bpa", ROOT / "scripts" / "build_paper_assets.py")
bpa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpa)

METRICS = ("cd", "hd", "nuc")


def synth(base, name, imp, desc):
    g = json.loads(json.dumps({k: v for k, v in base.items()
                               if k != "selection"}))
    g["dir"], g["desc"] = name, desc
    for m in METRICS:
        f = 1 - imp[m] / 100.0
        g["plateau"][m]["mean"] = base["plateau"][m]["mean"] * f
        g["curves"][m] = [v * f for v in base["curves"][m]]
    return g


def main() -> int:
    base = bpa.load_run(ROOT / "runs" / "B001_reproduce")
    design = json.loads(bpa.DESIGN.read_text(encoding="utf-8"))
    thr = design["significance_thresholds_pct"]
    # 用真实 8 组的名字，好看清标签挤不挤
    imps = {
        "A1_cd_balance": {"cd": 1.8, "hd": 3.1, "nuc": 1.4},
        "A2_cd_boost_bwd": {"cd": -1.2, "hd": -0.9, "nuc": 0.3},
        "B1_adv_fixed": {"cd": 0.9, "hd": 4.2, "nuc": 2.8},
        "B2_adv_adaptive": {"cd": 2.4, "hd": 6.8, "nuc": 5.1},
        "C1_uniform": {"cd": 0.2, "hd": 1.1, "nuc": 6.3},
        "D1_scale_qk": {"cd": 0.4, "hd": 0.7, "nuc": 0.1},
        "AC_combo": {"cd": 2.1, "hd": 3.6, "nuc": 6.9},
        "BD_combo": {"cd": 2.6, "hd": 7.1, "nuc": 5.4},
    }
    groups = {k: synth(base, k, v, design["groups"][k]["desc"])
              for k, v in imps.items()}

    out = ROOT / "paper_assets_TRIAL"
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = list(bpa.fig_group_bars(base, groups, figdir))
    figs += list(bpa.fig_group_curves(base, groups, figdir))
    f3 = bpa.fig_improvement(base, groups, thr, figdir)
    if f3:
        figs.append(f3)
    bpa.build_tables(design, base, groups, out)

    # 标记合成，防止日后误当真实结果
    for p in figs:
        dj = p.with_suffix(".data.json")
        d = json.loads(dj.read_text(encoding="utf-8"))
        d["synthetic"] = True
        d["warning"] = "版式审阅用合成数据，不是实验结果"
        dj.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    for p in figs:
        print("%-26s %8d B" % (p.name, p.stat().st_size))
    print("\n[注意] paper_assets_TRIAL 全部为合成版式样例，跑完真实组后删除。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
