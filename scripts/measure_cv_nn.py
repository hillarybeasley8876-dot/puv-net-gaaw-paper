# -*- coding: utf-8 -*-
"""补测预注册主指标 cv_nn 与分层 Q4/Q1 —— 覆盖全部论文级 run。

背景
----
第 3 章 3.5.5 预注册主指标为 cv_nn（最近邻间距变异系数），门槛 2SE，
但 8 组消融的 history/summary 只落了 CD/HD/NUC（见 scripts/audit_promises.py）。
本脚本对各 run 的 best.pt 做**推理**补测，不重训、不改权重。

口径一致性（关键）
----------------
完全复用 scripts/ch3_diagnose.py 的实现与常量：
  * nn_dist / cross_nn 逐字复用（欧氏、排除自身）
  * N_SAMPLE=200、SEED=20260811、VAL_RATIO=0.05、augment=False
  * 稀疏度四分位分层逻辑相同
因此 baseline 的复算值必须与 docs/_ch3_diag.json 一致 —— 脚本内置该校验，
不一致即 FAIL 并中止（防止悄悄换口径产生不可比数字）。

跨机红线
--------
输出按 GPU 分组标注。3090 组与 5090 组**不得并入同一张表**。

用法
    python scripts/measure_cv_nn.py --smoke          # 8 样本快验通路
    python scripts/measure_cv_nn.py                  # 全量 200 样本
    python scripts/measure_cv_nn.py --runs A,B       # 指定 run
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from puvnet.data.pu_dataset import PUTrainDataset  # noqa: E402
from puvnet.models.pu_transformer import PUTransformer  # noqa: E402

# --- 与 ch3_diagnose.py 一致的跑前定死常量 ---
SEED = 20260811
VAL_RATIO = 0.05
N_FULL = 200
N_SMOKE = 8

# 论文级 run 与其 GPU 归属（跨机红线：两组不得并表）
RUN_GPU = {
    "B002_baseline150": "3090",
    "ABL_A1_cd_balance": "3090",
    "ABL_A2_cd_boost_bwd": "3090",
    "ABL_D1_scale_qk": "3090",
    "ABL_C1_uniform": "3090",
    "ABL_AC_combo": "3090",
    "B002_baseline150_5090": "5090",
    "ABL_B1_adv_fixed": "5090",
    "ABL_B2_adv_adaptive": "5090",
}
# baseline 复算校验：必须与 _ch3_diag.json 对齐（相对容差）
CALIB_RUN = "B002_baseline150"
CALIB_TOL = 1e-6


# ---------------------------------------------------------------- 工具
# 以下三个函数逐字复用 ch3_diagnose.py，不得改动实现
def nn_dist(p: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def cross_nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return d.min(axis=1)


def desc(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "cv": float(v.std() / v.mean()) if v.mean() > 0 else 0.0,
        "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)), "max": float(v.max()),
        # SE 用于 2SE 门槛（此处为跨样本 SE，非跨 seed SE —— 口径须显式区分）
        "se_sample": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
        "n": int(len(v)),
    }


def load_generator(run: str) -> tuple[PUTransformer, int | None]:
    """从 run 的 best.pt 载入生成器。兼容 PUTransGAN 包装前缀。"""
    p = ROOT / "runs" / run / "ckpt" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("gen_state", ck.get("model_state", ck)))
    if any(k.startswith("generator.") for k in sd):
        sd = {k[len("generator."):]: v for k, v in sd.items()
              if k.startswith("generator.")}
    # 判别器权重存在时须剔除，否则 strict=False 会静默漏载生成器
    sd = {k: v for k, v in sd.items() if not k.startswith("discriminator.")}
    net = PUTransformer(up_ratio=4)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[FAIL] {run} 权重不匹配 "
                         f"missing={missing[:5]} unexpected={unexpected[:5]}")
    net.eval()
    return net, ck.get("epoch")


def measure(run: str, ds, pick: np.ndarray) -> dict:
    net, best_ep = load_generator(run)
    per = []
    t0 = time.time()
    with torch.no_grad():
        for c, i in enumerate(pick):
            inp, gt = ds[int(i)]
            inp_t = inp.unsqueeze(0) if torch.is_tensor(inp) \
                else torch.tensor(inp)[None]
            gt_np = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
            pred = net(inp_t.float())[0].numpy()
            inp_np = inp_t[0].numpy()

            nn_p, nn_g = nn_dist(pred), nn_dist(gt_np)
            fwd = cross_nn(pred, gt_np)
            bwd = cross_nn(gt_np, pred)
            f2, b2 = float((fwd ** 2).mean()), float((bwd ** 2).mean())
            d_to_in = cross_nn(gt_np, inp_np)
            edges = np.percentile(d_to_in, [0, 25, 50, 75])
            hi_edges = list(np.percentile(d_to_in, [25, 50, 75])) + [np.inf]

            per.append({
                "idx": int(i),
                "nn_pred_cv": float(nn_p.std() / nn_p.mean()),
                "nn_gt_cv": float(nn_g.std() / nn_g.mean()),
                "nn_pred_mean": float(nn_p.mean()),
                "nn_gt_mean": float(nn_g.mean()),
                "cd_fwd": f2, "cd_bwd": b2, "cd": f2 + b2,
                "bwd_share": b2 / (f2 + b2),
                "hd": float(max(fwd.max(), bwd.max()) ** 2),
                "bwd_by_sparsity": [
                    float((bwd[(d_to_in >= lo) & (d_to_in < hi)] ** 2).mean())
                    for lo, hi in zip(edges, hi_edges)
                ],
            })
            if (c + 1) % 50 == 0:
                print(f"    {run}: {c + 1}/{len(pick)}  "
                      f"({time.time() - t0:.0f}s)")

    def col(k):
        return [r[k] for r in per]

    q = [float(mean([r["bwd_by_sparsity"][j] for r in per])) for j in range(4)]
    cv_pred = np.array(col("nn_pred_cv"))
    return {
        "run": run, "gpu": RUN_GPU.get(run), "best_epoch": best_ep,
        "n_sample": len(per),
        # --- 预注册主指标 ---
        "cv_nn": desc(cv_pred),
        "cv_nn_gt": desc(np.array(col("nn_gt_cv"))),
        "cv_ratio_mean": float(mean(col("nn_pred_cv")) / mean(col("nn_gt_cv"))),
        # --- 同报 ---
        "cd_infer": desc(np.array(col("cd"))),
        "hd_infer": desc(np.array(col("hd"))),
        "bwd_share": desc(np.array(col("bwd_share"))),
        # --- 分层 Q4/Q1 ---
        "strata": {"q_labels": ["Q1(最密)", "Q2", "Q3", "Q4(最疏)"],
                   "bwd_mean_by_quartile": q,
                   "q4_over_q1": (q[3] / q[0]) if q[0] > 0 else None},
        "per_sample": per,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", help="逗号分隔；默认全部存在 best.pt 的论文 run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-merge", action="store_true",
                    help="不合并已有结果，整表覆盖（默认合并，防单 run 补测冲掉全表）")
    args = ap.parse_args()

    n_sample = N_SMOKE if args.smoke else N_FULL
    if args.runs:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        runs = [r for r in RUN_GPU
                if (ROOT / "runs" / r / "ckpt" / "best.pt").exists()]

    print("=" * 78)
    print(f"cv_nn 补测（预注册主指标） n_sample={n_sample} seed={SEED}")
    print(f"待测 run: {len(runs)}")
    print("=" * 78)

    ds = PUTrainDataset(source="pu1k", up_ratio=4, noise_beta=0.0, augment=False)
    n_total = len(ds)
    n_val = int(n_total * VAL_RATIO)
    val_start = n_total - n_val
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(np.arange(val_start, n_total),
                              size=min(n_sample, n_val), replace=False))
    print(f"dataset={n_total} val=[{val_start},{n_total}) 抽样 {len(pick)}\n")

    results = {}
    for r in runs:
        print(f"  [{r}]  GPU={RUN_GPU.get(r)}")
        try:
            results[r] = measure(r, ds, pick)
            m = results[r]["cv_nn"]
            print(f"    cv_nn mean={m['mean']:.6f} se={m['se_sample']:.6f} "
                  f"| Q4/Q1={results[r]['strata']['q4_over_q1']:.4f} "
                  f"({results[r]['elapsed_s']}s)")
        except Exception as e:
            print(f"    [SKIP] {type(e).__name__}: {e}")

    # ---------- 口径校验：baseline 必须复算出 _ch3_diag.json 的值 ----------
    print("\n" + "-" * 78)
    print("口径一致性校验（防止悄悄换口径）")
    calib_ok = None
    dp = ROOT / "docs" / "_ch3_diag.json"
    if CALIB_RUN in results and dp.exists() and not args.smoke:
        ref = json.loads(dp.read_text(encoding="utf-8"))
        ref_cv = ref["spacing"]["nn_pred_cv"]["mean"]
        got_cv = results[CALIB_RUN]["cv_nn"]["mean"]
        rel = abs(got_cv - ref_cv) / ref_cv
        calib_ok = rel <= CALIB_TOL
        print(f"  _ch3_diag.json nn_pred_cv.mean = {ref_cv:.9f}")
        print(f"  本次复算        cv_nn.mean     = {got_cv:.9f}")
        print(f"  相对偏差 = {rel:.3e}  (容差 {CALIB_TOL:.0e}) -> "
              f"{'PASS' if calib_ok else 'FAIL'}")
        if not calib_ok:
            print("  [FAIL] 口径不一致，拒绝输出 —— 新旧数字不可比，须先查因")
            return 2
    else:
        print("  (smoke 或缺 baseline/_ch3_diag.json，跳过校验)")

    # ---------- 分组汇总（跨机红线）----------
    print("\n" + "=" * 78)
    print("cv_nn 汇总 —— 按 GPU 分组，两组不得并表")
    print("=" * 78)
    for gpu in ("3090", "5090"):
        grp = [r for r in results if results[r]["gpu"] == gpu]
        if not grp:
            continue
        base = f"B002_baseline150{'_5090' if gpu == '5090' else ''}"
        print(f"\n[{gpu}]  基准 = {base}")
        print(f"  {'run':26s} {'cv_nn':>10s} {'SE':>9s} "
              f"{'Δ%':>8s} {'Δ/2SE':>7s} {'Q4/Q1':>7s}")
        b = results.get(base)
        for r in grp:
            m = results[r]["cv_nn"]
            q = results[r]["strata"]["q4_over_q1"]
            if b and r != base:
                d = m["mean"] - b["cv_nn"]["mean"]
                pct = 100.0 * d / b["cv_nn"]["mean"]
                # 2SE 门槛：合并 SE（保守取两者平方和根）
                se2 = 2.0 * ((m["se_sample"] ** 2
                              + b["cv_nn"]["se_sample"] ** 2) ** 0.5)
                ratio = abs(d) / se2 if se2 else float("nan")
                print(f"  {r:26s} {m['mean']:10.6f} {m['se_sample']:9.6f} "
                      f"{pct:+8.2f} {ratio:7.2f} {q:7.4f}")
            else:
                print(f"  {r:26s} {m['mean']:10.6f} {m['se_sample']:9.6f} "
                      f"{'--':>8s} {'--':>7s} {q:7.4f}")
        print("  注：Δ/2SE 为跨样本 SE 口径（同一权重、200 样本），"
              "非跨 seed SE；两者不可混用。")

    out = args.out or ("docs/_cv_nn_SMOKE.json" if args.smoke
                       else "docs/_cv_nn_measure.json")
    p = ROOT / out

    # 合并写入：--runs 单跑时不得覆盖已有 run 的结果。
    # 2026-08-12 踩坑：补测 ABL_AC_combo 一个 run，把之前 9 run 全表冲掉了。
    # 口径守护：只有 seed / n_sample / val_ratio 三项完全一致才允许合并，
    # 否则不同口径的数字会被并进同一张表（这是比丢数据更严重的错）。
    merged = {}
    if p.exists() and not args.no_merge:
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            same_caliber = (old.get("seed") == SEED
                            and old.get("n_sample") == n_sample
                            and old.get("val_ratio") == VAL_RATIO)
            if same_caliber and isinstance(old.get("runs"), dict):
                merged.update(old["runs"])
                kept = [k for k in merged if k not in results]
                if kept:
                    print(f"\n[merge] 保留已有 {len(kept)} 个 run: {', '.join(sorted(kept))}")
            elif not same_caliber:
                bak = p.with_suffix(".oldcaliber.json")
                bak.write_text(json.dumps(old, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                print(f"\n[merge] 口径不一致（seed/n_sample/val_ratio），"
                      f"不合并；旧文件另存 {bak.name}")
        except Exception as e:  # 读不动就不合并，但要留痕，不静默
            print(f"\n[merge] 读取旧文件失败({e})，本次不合并")
    merged.update(results)

    p.write_text(json.dumps({
        "note": ("各 run best.pt 推理补测；口径逐字复用 ch3_diagnose.py。"
                 "se_sample 为跨样本 SE，非跨 seed SE。"
                 "写入为合并模式：同名 run 覆盖，其余保留。"),
        "seed": SEED, "n_sample": n_sample, "val_ratio": VAL_RATIO,
        "calib_vs_ch3_diag": calib_ok,
        "runs": merged,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OK] 落盘 {p}（本次 {len(results)} 个 / 表内共 {len(merged)} 个）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
