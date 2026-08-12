# -*- coding: utf-8 -*-
"""承诺兑现审计器 —— 一次性扫出全文所有「已声明但未兑现」的实验债务。

动机
----
第 3 章 3.5.5 预注册了主指标 cv_nn 与 2SE 门槛，但 8 组消融只落了 CD/HD/NUC，
属于「考题与答卷不是一套」。这类债务若一次只发现一个，会让用户反复被追加要求。
本脚本把所有已写入文档的承诺集中抽取，与 runs/ 下实际存档逐条比对，
一次性给出完整债务清单。

判据来源
--------
只从文档与存档读取，脚本内不硬编码任何门槛数值（守 EXPERIMENT_LOG 的
「不得事后改判据」纪律）。所有正则命中的原文一并打印，供人工核对语义。

用法
    python scripts/audit_promises.py
    python scripts/audit_promises.py --json docs/_promise_audit.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RUNS = ROOT / "runs"
CH = DOCS / "chapters"

# 论文级 run（消融 + 基线）。跨机分组必须保留，不得并表。
RUNS_3090 = ["B002_baseline150", "ABL_A1_cd_balance", "ABL_A2_cd_boost_bwd",
             "ABL_D1_scale_qk", "ABL_C1_uniform", "ABL_AC_combo"]
RUNS_5090 = ["B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"]

# 承诺关键词 → 该承诺要求的存档证据字段/文件
# value: (正则, 说明, 检查函数名)
PROMISE_PATTERNS = [
    (r"主指标[为是]\s*\$?\\?mathrm\{?cv\}?_?\{?\\?text\{?nn|主指标为\s*\$\\mathrm\{cv\}_\{\\text\{nn\}\}\$|主指标.{0,12}cv",
     "预注册主指标 cv_nn", "cv_nn"),
    (r"2\\?mathrm\{?SE\}?|2SE", "2SE 显著性门槛（需多 seed）", "seed"),
    (r"同等参数量的?朴素扩容|朴素扩容", "同等参数量朴素扩容对照组", "capacity_ctrl"),
    (r"P2F", "P2F 指标（点到面距离）", "p2f"),
    (r"四分位后向分量|Q4/Q1", "分层指标 Q4/Q1 后向分量", "strata"),
    (r"REJECT_NULL|ACCEPT_FULL|ACCEPT_PARTIAL", "预注册裁定标签", "verdict"),
]


def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _hist_fields(run: str) -> set[str]:
    p = RUNS / run / "history.json"
    if not p.exists():
        return set()
    h = json.loads(p.read_text(encoding="utf-8"))
    return set(h[0].keys()) if h else set()


def _summary(run: str) -> dict:
    p = RUNS / run / "summary_stats.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _seed_of(run: str) -> str | None:
    p = RUNS / run / "config.yaml"
    for line in _read(p).splitlines():
        ls = line.strip()
        if ls.startswith("seed:"):
            return ls.split(":", 1)[1].strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    debts: list[dict] = []
    print("=" * 78)
    print("承诺兑现审计 —— 已声明 vs 已落盘")
    print("=" * 78)

    # ---------- 1. 抽取文档中的承诺原文 ----------
    print("\n[1] 文档承诺抽取")
    src_files = sorted(CH.glob("*.md")) + [
        DOCS / "THESIS_OUTLINE.md", DOCS / "EXPERIMENT_LOG.md",
        DOCS / "STYLE_GUIDE.md",
    ]
    found: dict[str, list[tuple[str, int, str]]] = {}
    for pat, desc, tag in PROMISE_PATTERNS:
        rx = re.compile(pat)
        hits: list[tuple[str, int, str]] = []
        for f in src_files:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if rx.search(line):
                    hits.append((f.name, i, line.strip()[:160]))
        found[tag] = hits
        print(f"  {desc:32s} 命中 {len(hits):3d} 处"
              + ("" if hits else "   <- 文档中未声明"))

    # ---------- 2. 逐条核对存档 ----------
    print("\n[2] 存档核对")

    all_runs = RUNS_3090 + RUNS_5090
    done_runs = [r for r in all_runs if (RUNS / r / "history.json").exists()]
    print(f"  存在 history.json 的 run: {len(done_runs)}/{len(all_runs)}")
    for r in all_runs:
        n = len(json.loads((RUNS / r / "history.json").read_text(encoding="utf-8"))) \
            if (RUNS / r / "history.json").exists() else 0
        seed = _seed_of(r)
        gpu = "5090" if r in RUNS_5090 else "3090"
        print(f"    {r:26s} {n:3d} ep  seed={seed}  [{gpu}]")

    # --- cv_nn ---
    print("\n  -- cv_nn（预注册主指标）--")
    cv_missing = []
    for r in done_runs:
        f = _hist_fields(r)
        s = _summary(r)
        has = any("cv" in k and "nn" in k for k in f) or \
              any("cv" in str(k) and "nn" in str(k) for k in s.keys())
        if not has:
            cv_missing.append(r)
    print(f"    缺 cv_nn 的 run: {len(cv_missing)}/{len(done_runs)}")
    if cv_missing:
        print(f"      {', '.join(cv_missing)}")
        debts.append({
            "debt": "cv_nn 未测",
            "promised_in": [h[:2] for h in found.get("cv_nn", [])[:3]],
            "runs_missing": cv_missing,
            "fix": "复用 ch3_diagnose.py 推理口径，对各 run best.pt 补测 cv_nn",
            "needs_retrain": False,
        })

    # --- seed ---
    print("\n  -- 多 seed（2SE 门槛前提）--")
    seeds: dict[str, list[str]] = {}
    for r in done_runs:
        seeds.setdefault(str(_seed_of(r)), []).append(r)
    print(f"    不同 seed 数量 = {len(seeds)}: {list(seeds.keys())}")
    if len(seeds) < 2:
        print("    -> 单 seed，无法计算跨 seed SE；当前 σ 均为跨 epoch 口径")
        debts.append({
            "debt": "单 seed，2SE 无法兑现",
            "promised_in": [h[:2] for h in found.get("seed", [])[:3]],
            "fix": "补 seed 重跑（用户已批准 baseline+B2 两组 ×2 seed）",
            "needs_retrain": True,
        })

    # --- P2F ---
    print("\n  -- P2F --")
    p2f_runs = [r for r in done_runs
                if any("p2f" in k.lower() for k in _hist_fields(r))
                or any("p2f" in str(k).lower() for k in _summary(r))]
    print(f"    含 P2F 的 run: {len(p2f_runs)}/{len(done_runs)}")
    if len(p2f_runs) < len(done_runs):
        miss = [r for r in done_runs if r not in p2f_runs]
        print(f"      缺: {', '.join(miss)}")
        debts.append({
            "debt": "P2F 未测（3.5.5 列为同报指标）",
            "promised_in": [h[:2] for h in found.get("p2f", [])[:3]],
            "runs_missing": miss,
            "fix": "需 mesh/法向量；若无 GT mesh 则须在文中声明不可测并说明理由",
            "needs_retrain": False,
        })

    # --- 朴素扩容对照 ---
    print("\n  -- 同等参数量朴素扩容对照 --")
    cap = [d.name for d in RUNS.iterdir()
           if d.is_dir() and re.search(r"capacity|wider|scale_up|naive", d.name, re.I)]
    print(f"    命中目录: {cap if cap else '无'}")
    if found.get("capacity_ctrl") and not cap:
        debts.append({
            "debt": "朴素扩容对照组未跑",
            "promised_in": [h[:2] for h in found.get("capacity_ctrl", [])[:3]],
            "fix": ("若第 4 章主线改为训练机制（无参数量增加），"
                    "该对照前提消失，须在 3.5.5 回改中显式注销并留痕"),
            "needs_retrain": None,
        })

    # --- Q4/Q1 分层 ---
    print("\n  -- Q4/Q1 分层后向分量 --")
    diag = json.loads(_read(DOCS / "_ch3_diag.json") or "{}")
    has_strata = "sparsity_strata" in diag
    print(f"    baseline 已有分层诊断: {has_strata}")
    strata_runs = [r for r in done_runs
                   if (DOCS / f"_diag_{r}.json").exists()]
    print(f"    各消融组分层诊断产物: {len(strata_runs)}/{len(done_runs)}")
    if found.get("strata") and len(strata_runs) < len(done_runs):
        debts.append({
            "debt": "改进组的 Q4/Q1 分层后向分量未测",
            "promised_in": [h[:2] for h in found.get("strata", [])[:3]],
            "fix": "与 cv_nn 同一次推理一并产出（同脚本，零额外成本）",
            "needs_retrain": False,
        })

    # ---------- 3. 汇总 ----------
    print("\n" + "=" * 78)
    print(f"债务清单：{len(debts)} 条")
    print("=" * 78)
    for i, d in enumerate(debts, 1):
        nr = d.get("needs_retrain")
        cost = ("需重训" if nr else "不需重训" if nr is False else "取决于定位改动")
        print(f"\n  [{i}] {d['debt']}   ({cost})")
        print(f"      修法: {d['fix']}")
        if d.get("promised_in"):
            print(f"      承诺出处: {d['promised_in']}")
        if d.get("runs_missing"):
            print(f"      缺失 run: {len(d['runs_missing'])} 个")

    need_gpu = [d for d in debts if d.get("needs_retrain") is True]
    print(f"\n  需要 GPU 时间的债务：{len(need_gpu)} 条")
    for d in need_gpu:
        print(f"    - {d['debt']}")

    if args.json:
        p = pathlib.Path(args.json)
        p.write_text(json.dumps(
            {"debts": debts, "promise_hits": found,
             "runs": {r: {"epochs": len(json.loads(
                 (RUNS / r / "history.json").read_text(encoding="utf-8")))
                 if (RUNS / r / "history.json").exists() else 0,
                 "seed": _seed_of(r),
                 "gpu": "5090" if r in RUNS_5090 else "3090"}
                 for r in all_runs}},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] 落盘 {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
