"""从 B2 存档反解 ρ（对抗梯度/CD 梯度 范数比）逐 epoch 序列。

原理
----
`adaptive_adv_weight` (puvnet/models/pu_gan.py:294) 的定义：

    w_auto = target_ratio * ||g_cd|| / ||g_adv||

令 ρ := ||g_adv|| / ||g_cd||，则

    ρ = target_ratio / w_auto

`train_adv_w_adaptive` 已逐 epoch 落盘（ABL_B2_adv_adaptive），
因此 ρ 序列**无需重跑**即可从存档反解。

口径限制（必须随图/表一同披露，不得省略）
--------------------------------------
`train_adv_w_adaptive` 是 **epoch 内所有 batch 的算术平均**
(scripts/train_pu.py:339 `v / nb`)。w_auto 是比值，而
mean(a/b) != mean(a)/mean(b)，故反解得到的

    rho_hat(ep) = target_ratio / mean_batch(w_auto)

是「epoch 内 w_auto 均值的倒数」，属**调和口径**的 ρ 代表值，
不等于 mean_batch(rho)。它可用于展示**跨 epoch 的数量级演化趋势**，
不可当作某个 batch 的精确 ρ 值引用。

target_ratio 从该 run 的 config.yaml 读取，脚本内不出现字面量。

用法
----
    python scripts/analyze_rho_trace.py                 # 打印统计
    python scripts/analyze_rho_trace.py --json out.json # 兼落盘
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ADAPTIVE = "ABL_B2_adv_adaptive"
RUN_FIXED = "ABL_B1_adv_fixed"
KEY_AUTO = "train_adv_w_adaptive"
KEY_EFF = "train_adv_w_effective"
KEY_WARM = "train_adv_warmup_factor"
# 平台区口径沿用主表定案（2026-08-11）：ep75-149
PLATEAU_LO, PLATEAU_HI = 75, 149


def _load(run: str) -> tuple[list[dict], dict]:
    d = ROOT / "runs" / run
    hp, cp = d / "history.json", d / "config.yaml"
    if not hp.exists():
        sys.exit(f"[FAIL] 缺 {hp}")
    if not cp.exists():
        sys.exit(f"[FAIL] 缺 {cp}")
    hist = json.loads(hp.read_text(encoding="utf-8"))
    cfg: dict = {}
    for line in cp.read_text(encoding="utf-8").splitlines():
        ls = line.strip()
        if ls.startswith("#") or ":" not in ls:
            continue
        k, _, v = ls.partition(":")
        cfg.setdefault(k.strip(), v.strip())
    return hist, cfg


def _fnum(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "min": min(xs), "max": max(xs),
        "median": st.median(xs),
        "mean": st.fmean(xs),
        "p10": st.quantiles(xs, n=10)[0] if len(xs) >= 10 else None,
        "p90": st.quantiles(xs, n=10)[8] if len(xs) >= 10 else None,
        "decades": (max(xs) / min(xs)) if min(xs) > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="统计结果落盘路径")
    args = ap.parse_args()

    hist, cfg = _load(RUN_ADAPTIVE)
    if "adv_target_ratio" not in cfg:
        sys.exit("[FAIL] config.yaml 无 adv_target_ratio，拒绝硬编码回退")
    tr = float(cfg["adv_target_ratio"])
    if cfg.get("adaptive_adv", "").lower() != "true":
        sys.exit(f"[FAIL] {RUN_ADAPTIVE} 的 adaptive_adv != true，反解不适用")

    print("=" * 74)
    print(f"ρ 反解 — {RUN_ADAPTIVE}")
    print(f"  target_ratio (来自 config.yaml) = {tr}")
    print(f"  ρ = target_ratio / w_auto   [w_auto = {KEY_AUTO}]")
    print("=" * 74)

    rows = []
    for rec in hist:
        ep = int(rec["epoch"])
        w = _fnum(rec.get(KEY_AUTO))
        if w is None or w <= 0:
            print(f"  [WARN] ep{ep:03d} w_auto 缺失/非正 = {rec.get(KEY_AUTO)!r}")
            continue
        rows.append({
            "epoch": ep,
            "w_auto": w,
            "rho_hat": tr / w,
            "warmup": _fnum(rec.get(KEY_WARM)),
            "w_eff": _fnum(rec.get(KEY_EFF)),
        })
    if not rows:
        sys.exit("[FAIL] 无可用 w_auto 记录")

    rho = [r["rho_hat"] for r in rows]
    wau = [r["w_auto"] for r in rows]
    plat = [r for r in rows if PLATEAU_LO <= r["epoch"] <= PLATEAU_HI]

    print(f"\n全程 {len(rows)} epoch:")
    s_r, s_w = _stats(rho), _stats(wau)
    for name, s in (("w_auto", s_w), ("rho_hat", s_r)):
        print(f"  {name:8s} min={s['min']:.6g} median={s['median']:.6g} "
              f"max={s['max']:.6g} 跨度={s['decades']:.4g}×")

    print(f"\n采样点（epoch: w_auto -> rho_hat, warmup）:")
    for ep in (0, 1, 5, 10, 20, 40, 60, 75, 100, 125, 149):
        m = next((r for r in rows if r["epoch"] == ep), None)
        if m:
            print(f"  ep{ep:03d}: w_auto={m['w_auto']:.6g} "
                  f"-> rho={m['rho_hat']:.6g}  (warmup={m['warmup']:.3g}, "
                  f"w_eff={m['w_eff']:.6g})")

    if plat:
        s_p = _stats([r["rho_hat"] for r in plat])
        print(f"\n平台区 ep{PLATEAU_LO}-{PLATEAU_HI} ({s_p['n']} ep):")
        print(f"  rho_hat median={s_p['median']:.6g} "
              f"min={s_p['min']:.6g} max={s_p['max']:.6g}")

    # --- 与 B1 固定权重对照：这是 GAAW 机制的核心证据 ---
    h1, c1 = _load(RUN_FIXED)
    w_fixed = sorted({round(_fnum(r.get(KEY_EFF)), 6) for r in h1
                      if _fnum(r.get(KEY_EFF)) is not None})
    print(f"\n对照 {RUN_FIXED}（adaptive_adv={c1.get('adaptive_adv')}）:")
    print(f"  w_eff 取值集合（去重后 {len(w_fixed)} 个）= {w_fixed}")
    print(f"  cfg w_adv = {c1.get('w_adv')}  ← 全程固定，仅受 warmup 缩放")

    # --- 与文档中已引用的标定值对照 ---
    print("\n与既有文档引用值的关系（诚实性核对）:")
    cal_w = float(cfg.get("w_adv", "nan"))
    print(f"  文档引用 w_auto≈{cal_w}（B-001/E-000 单次标定）"
          f" -> 对应 rho={tr / cal_w:.6g} = 1/{cal_w / tr:.4g}")
    near = min(rows, key=lambda r: abs(r["w_auto"] - cal_w))
    print(f"  B2 全程离该标定值最近的 epoch = ep{near['epoch']:03d} "
          f"(w_auto={near['w_auto']:.6g})")
    frac_below = sum(1 for r in rows if r["w_auto"] < cal_w) / len(rows)
    print(f"  B2 中 w_auto < {cal_w} 的 epoch 占比 = {frac_below:.1%}")

    out = {
        "run": RUN_ADAPTIVE,
        "target_ratio": tr,
        "formula": "rho = target_ratio / w_auto",
        "caveat": ("w_auto 为 epoch 内 batch 算术平均；"
                   "rho_hat 为调和口径代表值，非 mean(rho)。"
                   "仅可用于数量级演化趋势，不可作单 batch 精确值引用。"),
        "source_field": KEY_AUTO,
        "plateau": [PLATEAU_LO, PLATEAU_HI],
        "stats_w_auto": s_w,
        "stats_rho_hat": s_r,
        "stats_rho_plateau": _stats([r["rho_hat"] for r in plat]) if plat else None,
        "fixed_run": {"run": RUN_FIXED, "w_eff_unique": w_fixed,
                      "w_adv_cfg": c1.get("w_adv")},
        "doc_calibration": {"w_auto": cal_w, "rho": tr / cal_w,
                            "b2_frac_below": frac_below},
        "rows": rows,
    }
    if args.json:
        p = pathlib.Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        print(f"\n[OK] 落盘 {p}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
