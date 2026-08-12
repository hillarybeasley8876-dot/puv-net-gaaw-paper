# -*- coding: utf-8 -*-
"""verdict_ablation.py 的自检 —— 用 B-001 真实平台区做底，注入已知改善率，
验证四档裁定与边缘标记都正确。

只在临时目录里造 summary_stats.json，不触碰任何真实 run 目录。
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
    "vab", ROOT / "scripts" / "verdict_ablation.py")
vab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vab)

from puvnet.metrics.selection import plateau_stats  # noqa: E402

THR = {"cd": 0.66, "hd": 2.18, "nuc": 0.93}
METRICS = ("cd", "hd", "nuc")


def real_base() -> dict:
    m = json.loads((ROOT / "runs" / "B001_reproduce" / "metrics.json")
                   .read_text(encoding="utf-8"))["records"]
    return plateau_stats(m, frac=0.5)


def write_summary(d: Path, plateau: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary_stats.json").write_text(
        json.dumps({"plateau": plateau}, ensure_ascii=False), encoding="utf-8")


def scaled(plateau: dict, imp: dict) -> dict:
    """把平台区均值按给定改善率(%)缩放；正=变好(值变小)。"""
    out = json.loads(json.dumps(plateau))
    for m in METRICS:
        out[m]["plateau_mean"] = plateau[m]["plateau_mean"] * (1 - imp[m] / 100)
    return out


def main() -> int:
    base_pl = real_base()
    tmp = Path(tempfile.mkdtemp(prefix="vab_selfcheck_"))
    ok = fail = 0
    try:
        write_summary(tmp / "base", base_pl)
        base = vab.load_plateau(tmp / "base")
        assert base is not None, "load_plateau 读不出真实结构"
        print(f"[OK ] load_plateau 键名匹配真实 plateau_stats 输出 "
              f"epochs={base['_epochs']}")
        ok += 1

        cases = [
            # (名称, 改善率%, 期望裁定, 期望需补种子)
            ("三项全绿-充裕", {"cd": 5.0, "hd": 8.0, "nuc": 6.0},
             "ACCEPT_FULL", False),
            ("三项全绿-边缘", {"cd": 0.8, "hd": 2.5, "nuc": 1.1},
             "ACCEPT_FULL", True),
            ("部分绿其余持平", {"cd": 3.0, "hd": 0.5, "nuc": 0.1},
             "ACCEPT_PART", True),   # hd/nuc 落在门槛 1.5x 内 -> 应标记补种子
            ("一项劣化", {"cd": 5.0, "hd": 8.0, "nuc": -3.0},
             "REJECT_TRADE", False),
            ("全在门槛内", {"cd": 0.2, "hd": 1.0, "nuc": 0.3},
             "REJECT_NULL", True),
            ("完全无变化", {"cd": 0.0, "hd": 0.0, "nuc": 0.0},
             "REJECT_NULL", False),
        ]
        for name, imp, want_v, want_seed in cases:
            exp_dir = tmp / f"exp_{name}"
            write_summary(exp_dir, scaled(base_pl, imp))
            exp = vab.load_plateau(exp_dir)
            r = vab.judge(base, exp, THR)
            got_v, got_seed = r["verdict"], r["needs_extra_seed"]
            # 复核改善率算得准（容差 1e-6 个百分点）
            drift = max(abs(row["improve_pct"] - imp[row["metric"]])
                        for row in r["rows"])
            good = (got_v == want_v and got_seed == want_seed
                    and drift < 1e-6)
            if good:
                print(f"[OK ] {name:<14} -> {got_v:<13} "
                      f"补种子={got_seed} 改善率误差={drift:.2e}")
                ok += 1
            else:
                print(f"[FAIL] {name:<14} -> {got_v} (期望 {want_v}) "
                      f"补种子={got_seed} (期望 {want_seed}) drift={drift:.2e}")
                fail += 1

        # 门槛来源必须是设计存档，脚本内不得有魔数
        src = (ROOT / "scripts" / "verdict_ablation.py").read_text(
            encoding="utf-8")
        if "0.66" in src or "2.18" in src or "0.93" in src:
            print("[FAIL] verdict_ablation.py 内出现门槛魔数，应只从设计存档读")
            fail += 1
        else:
            print("[OK ] 门槛无魔数，只从 ablation_matrix.json 读")
            ok += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 60}\n自检: {ok} PASS / {fail} FAIL")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
