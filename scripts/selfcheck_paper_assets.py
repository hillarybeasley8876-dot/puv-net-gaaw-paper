# -*- coding: utf-8 -*-
"""build_paper_assets.py 的自检 —— 用真实 B-001 曲线 + 合成消融组，
把 T1(柱状) / T2(曲线) / T3(改善率) 三类图和主表全部走一遍。

合成组只写进系统临时目录，绝不落进 runs/ ，不会污染真实产物。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "bpa", ROOT / "scripts" / "build_paper_assets.py")
bpa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpa)

METRICS = ("cd", "hd", "nuc")
MNAME = {"cd": "CD", "hd": "HD", "nuc": "NUC"}
THR = {"cd": 0.66, "hd": 2.18, "nuc": 0.93}


def synth(base: dict, name: str, imp: dict, desc: str) -> dict:
    """按给定改善率(%)缩放 baseline，造一个"实验组"。"""
    g = json.loads(json.dumps({k: v for k, v in base.items()
                               if k != "selection"}))
    g["dir"] = name
    g["desc"] = desc
    for m in METRICS:
        f = 1 - imp[m] / 100.0
        g["plateau"][m]["mean"] = base["plateau"][m]["mean"] * f
        g["curves"][m] = [v * f for v in base["curves"][m]]
    return g


def main() -> int:
    ok = fail = 0
    base = bpa.load_run(ROOT / "runs" / "B001_reproduce")
    if base is None:
        print("[FAIL] 读不到 B-001，无法自检")
        return 1
    print(f"[OK ] load_run 回退补算 plateau 成功 "
          f"derived={base['plateau_derived']} range={base['epoch_range']}")
    ok += 1

    design = json.loads(bpa.DESIGN.read_text(encoding="utf-8"))
    groups = {
        "SYNTH_full": synth(base, "SYNTH_full",
                            {"cd": 4.0, "hd": 6.0, "nuc": 5.0}, "合成-三项全绿"),
        "SYNTH_part": synth(base, "SYNTH_part",
                            {"cd": 3.0, "hd": 0.4, "nuc": 0.2}, "合成-部分绿"),
        "SYNTH_trade": synth(base, "SYNTH_trade",
                             {"cd": 5.0, "hd": 7.0, "nuc": -4.0}, "合成-有劣化"),
        "SYNTH_null": synth(base, "SYNTH_null",
                            {"cd": 0.1, "hd": 0.5, "nuc": 0.2}, "合成-无效"),
    }

    tmp = Path(tempfile.mkdtemp(prefix="bpa_selfcheck_"))
    try:
        figdir = tmp / "figures"
        figdir.mkdir(parents=True, exist_ok=True)

        f1 = bpa.fig_group_bars(base, groups, figdir)
        f2 = bpa.fig_group_curves(base, groups, figdir)
        f3 = bpa.fig_improvement(base, groups, THR, figdir)
        allf = list(f1) + list(f2) + ([f3] if f3 else [])

        want = 3 + 3 + 1
        if len(allf) == want:
            print(f"[OK ] 图数量 {len(allf)} == 期望 {want} (T1x3 + T2x3 + T3)")
            ok += 1
        else:
            print(f"[FAIL] 图数量 {len(allf)} != 期望 {want}")
            fail += 1

        for p in allf:
            dj = p.with_suffix(".data.json")
            if p.exists() and p.stat().st_size > 5000 and dj.exists():
                ok += 1
                print(f"[OK ] {p.name:<24} {p.stat().st_size:>7} B  data.json OK")
            else:
                fail += 1
                print(f"[FAIL] {p.name} 缺失/过小/无 data.json")

        # 压平缺陷回归（本项目已复发 3 次）：柱状图必须让组间差异可见。
        # 判据：数据跨度须占纵轴跨度的一定比例，否则柱高被压成齐平。
        for m in METRICS:
            meta = json.loads((figdir / f"T1_plateau_{m}.data.json")
                              .read_text(encoding="utf-8"))
            sc = meta["scale"]
            vals = [v * sc for v in meta["mean"]]
            errs = [v * sc for v in meta["std"]]
            lo_d = min(v - e for v, e in zip(vals, errs))
            hi_d = max(v + e for v, e in zip(vals, errs))
            y0, y1 = meta["ylim"]
            span_axis = y1 - y0
            span_data = hi_d - lo_d
            ratio = span_data / span_axis if span_axis > 0 else 0.0
            if ratio >= 0.35 and meta.get("y_axis_starts_at_zero") is False:
                print(f"[OK ] T1 {MNAME[m]:<3} 数据跨度占轴 {ratio:.0%}"
                      f"（>=35%，差异可见）")
                ok += 1
            else:
                print(f"[FAIL] T1 {MNAME[m]} 数据跨度仅占轴 {ratio:.0%}"
                      f"，柱高被压平；zero_start={meta.get('y_axis_starts_at_zero')}")
                fail += 1
            # 非零起点必须在图里明确声明，避免误导读者
            if "非零起点" in meta.get("note", ""):
                ok += 1
            else:
                print(f"[FAIL] T1 {MNAME[m]} 未声明非零起点")
                fail += 1

        # NUC 不得被 1e3 缩放（比例量混轴是本项目已复发两次的缺陷）
        nuc_meta = json.loads(
            (figdir / "T1_plateau_nuc.data.json").read_text(encoding="utf-8"))
        if abs(nuc_meta["scale"] - 1.0) < 1e-12:
            print("[OK ] NUC 未被缩放（比例量不混轴）")
            ok += 1
        else:
            print(f"[FAIL] NUC scale={nuc_meta['scale']}，应为 1.0")
            fail += 1
        cd_meta = json.loads(
            (figdir / "T1_plateau_cd.data.json").read_text(encoding="utf-8"))
        if abs(cd_meta["scale"] - 1e3) < 1e-9:
            print("[OK ] CD 按 1e3 缩放")
            ok += 1
        else:
            print(f"[FAIL] CD scale={cd_meta['scale']}")
            fail += 1

        # 主表裁定必须与注入的改善率一致
        verdicts = bpa.build_tables(design, base, groups, tmp)
        expect = {"SYNTH_full": "ACCEPT_FULL", "SYNTH_part": "ACCEPT_PART",
                  "SYNTH_trade": "REJECT_TRADE", "SYNTH_null": "REJECT_NULL"}
        for k, want_v in expect.items():
            got = verdicts[k]["verdict"]
            if got == want_v:
                print(f"[OK ] 表裁定 {k:<13} {got}")
                ok += 1
            else:
                print(f"[FAIL] 表裁定 {k} = {got}，期望 {want_v}")
                fail += 1

        md = (tmp / "TABLE_ablation.md").read_text(encoding="utf-8")
        tex = (tmp / "TABLE_ablation.tex").read_text(encoding="utf-8")
        checks = [
            ("md 含 baseline 行", "**baseline**" in md),
            ("md 含全部合成组", all(k in md for k in expect)),
            ("md 标注门槛", "0.66" in md and "2.18" in md),
            ("tex 三线表", r"\toprule" in tex and r"\bottomrule" in tex),
            ("tex 下划线已转义", r"SYNTH\_full" in tex),
            ("tex 含表注门槛", "tablenotes" in tex),
        ]
        for label, cond in checks:
            if cond:
                print(f"[OK ] {label}")
                ok += 1
            else:
                print(f"[FAIL] {label}")
                fail += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 62}\n自检: {ok} PASS / {fail} FAIL")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
