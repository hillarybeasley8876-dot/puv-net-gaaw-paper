# -*- coding: utf-8 -*-
"""消融组裁定器 —— 按跑前定死的门槛与接受准则自动出结论。

用法:
    python scripts/verdict_ablation.py                    # 裁定全部已完成组
    python scripts/verdict_ablation.py --group A1_cd_balance

设计约束（写死，禁止调用方绕过）:
  * 门槛只从 runs/ablation_design/ablation_matrix.json 读，脚本内不出现魔数
  * 平台区定义只从各 run 的 summary_stats.json 读（训练时落盘，事后不可改）
  * 三指标一律「越小越好」；改善率 = (base - exp) / base
  * 裁定四档：ACCEPT_FULL / ACCEPT_PART / REJECT_TRADE / REJECT_NULL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "runs" / "ablation_design" / "ablation_matrix.json"
BASE_RUN = ROOT / "runs" / "B002_baseline150"
METRICS = ("cd", "hd", "nuc")


def load_plateau(run_dir: Path) -> dict | None:
    """从 summary_stats.json 取平台区均值/标准差。

    键名以 plateau_stats() 的真实返回为准（已用 B-001 数据核对）：
      plateau[<metric>] = {plateau_mean, plateau_std, best, best_epoch,
                           best_sigma_from_mean}
      plateau['epoch_range'] = [start, end]
    """
    f = run_dir / "summary_stats.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    pl = d.get("plateau")
    if not isinstance(pl, dict):
        return None
    out = {}
    for m in METRICS:
        sub = pl.get(m)
        if not isinstance(sub, dict) or "plateau_mean" not in sub:
            return None
        out[m] = {"mean": float(sub["plateau_mean"]),
                  "std": float(sub["plateau_std"]),
                  "best": sub.get("best"),
                  "best_epoch": sub.get("best_epoch")}
    out["_epochs"] = pl.get("epoch_range")
    out["_n"] = pl.get("plateau_n")
    return out


def judge(base: dict, exp: dict, thr: dict) -> dict:
    """逐指标算改善率并出总裁定。"""
    rows = []
    n_better = n_worse = n_flat = 0
    for m in METRICS:
        b = float(base[m]["mean"])
        e = float(exp[m]["mean"])
        imp = (b - e) / b * 100.0          # 正 = 变好（三项都越小越好）
        t = float(thr[m])
        if imp > t:
            tag, n_better = "BETTER", n_better + 1
        elif imp < -t:
            tag, n_worse = "WORSE", n_worse + 1
        else:
            tag, n_flat = "FLAT", n_flat + 1
        rows.append({"metric": m, "base_mean": b, "exp_mean": e,
                     "improve_pct": imp, "threshold_pct": t, "tag": tag,
                     "base_std": float(base[m]["std"]),
                     "exp_std": float(exp[m]["std"])})

    if n_worse:
        verdict = "REJECT_TRADE"
        reason = f"{n_worse} 项劣化超门槛 -> 判 trade-off，不得声称改进"
    elif n_better == len(METRICS):
        verdict = "ACCEPT_FULL"
        reason = "三项改善均超门槛 -> 可作为主表改进项"
    elif n_better:
        verdict = "ACCEPT_PART"
        reason = (f"{n_better} 项超门槛、{n_flat} 项持平 -> 附录报告，"
                  f"须写明持平项")
    else:
        verdict = "REJECT_NULL"
        reason = "三项均在门槛内 -> 判无效，如实报告"

    # 落在门槛 1.5 倍内的组证据偏弱，需补种子
    marginal = [r["metric"] for r in rows
                if 0 < abs(r["improve_pct"]) <= 1.5 * r["threshold_pct"]]
    return {"verdict": verdict, "reason": reason, "rows": rows,
            "marginal_metrics": marginal,
            "needs_extra_seed": bool(marginal)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    args = ap.parse_args()

    if not DESIGN.exists():
        print(f"[FAIL] 缺设计存档 {DESIGN}，先跑 make_ablation_configs.py")
        return 2
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    thr = design["significance_thresholds_pct"]
    rule = design.get("acceptance_rule", {})

    base = load_plateau(BASE_RUN)
    if base is None:
        print(f"[WAIT] baseline 平台区未就绪：{BASE_RUN/'summary_stats.json'}")
        print("       B-002 跑完才会落盘，现在无法裁定任何组。")
        return 1

    print("=" * 74)
    print("消融裁定（门槛跑前定死，来自 ablation_matrix.json）")
    print(f"  门槛: CD {thr['cd']}% / HD {thr['hd']}% / NUC {thr['nuc']}%")
    print(f"  baseline 平台区 epochs: {base.get('_epochs')}")
    print("=" * 74)

    groups = design["groups"]          # dict: name -> {config, out_dir, ...}
    names = [args.group] if args.group else list(groups.keys())
    results = {}
    for name in names:
        g = groups.get(name)
        if g is None:
            print(f"[SKIP] 设计里没有组 {name}")
            continue
        run_dir = ROOT / g["out_dir"]   # out_dir 形如 "runs/ABL_A1_cd_balance"
        exp = load_plateau(run_dir)
        if exp is None:
            print(f"\n[{name}] 未完成（无 summary_stats.json）  {run_dir.name}")
            continue
        r = judge(base, exp, thr)
        r["run_dir"] = g["out_dir"]
        r["plateau_epochs"] = exp.get("_epochs")
        r["changes"] = g.get("changes")
        results[name] = r
        print(f"\n[{name}]  ==> {r['verdict']}")
        print(f"  {r['reason']}")
        for row in r["rows"]:
            sign = "+" if row["improve_pct"] >= 0 else ""
            print(f"    {row['metric'].upper():<4} "
                  f"{row['base_mean']:.6f} -> {row['exp_mean']:.6f}  "
                  f"改善 {sign}{row['improve_pct']:.2f}% "
                  f"(门槛 {row['threshold_pct']}%)  [{row['tag']}]")
        if r["needs_extra_seed"]:
            print(f"  ⚠️ 边缘指标 {r['marginal_metrics']} 落在门槛 1.5 倍内，"
                  f"证据偏弱，建议补种子")

    if results:
        out = ROOT / "runs" / "ablation_design" / "verdicts.json"
        out.write_text(json.dumps(
            {"thresholds_pct": thr, "acceptance_rule": rule,
             "baseline_run": BASE_RUN.name,
             "baseline_plateau_epochs": base.get("_epochs"),
             "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n[存档] {out}")
    else:
        print("\n(暂无已完成的消融组)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
