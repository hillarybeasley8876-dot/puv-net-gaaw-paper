"""E-000 指标校准与合理性基线（v2）。

目的
----
在任何真实模型跑出数字之前，先搞清楚四个指标的【行为特性】与【数值范围】。
否则拿到真实模型的 CD=0.5e-3 时，无法判断这是好、是坏、还是管线有 bug。

⚠️ v1 已作废，作废原因见 docs/EXPERIMENT_LOG.md「E-000-v1」条目：
  1. **指标协议错误**：CD/HD 用了 L2 距离，而 PU-GCN 官方协议是平方距离
     （一手代码核实见 refs/pu_gcn/CD_PROTOCOL_SOURCE.md），导致所有数值偏离文献 25~50 倍。
  2. **"作弊上界"构造错误**：v1 用 `gt_half_dup`（GT 抽一半后复制两份）当上界，
     但 `scripts/debug_cd_scale.py` 已实测证明
     `CD(gt_half_dup, gt) == CD(gt_half, gt) == 6.9460e-3` —— 复制点对 CD 零贡献，
     该构造实际是**残缺点云**（丢了一半 GT 覆盖），不是上界，命名与定位都错。

方法（v2 参考上采样器）
----------------------
  1. gt_exact      : 直接把 GT 当预测（CD 恒等于 0）—— 验证管线无系统性偏移
  2. gt_jitter_tiny: GT + sigma=0.001 抖动 —— **真正的性能上界参考**
                     （形状与密度分布都对，只带可控的微小扰动）
  3. gt_half_dup   : GT 抽一半再复制（v1 误称"上界"）—— 保留，但重新定位为
                     **"覆盖缺失"参考**，用于展示 CD 对丢点的惩罚
  4. copy4         : 每点复制 4 份（点数对，但零新信息）—— 性能下界参考
  5. jitter_small  : 输入复制 + sigma=0.005 抖动
  6. jitter_large  : 输入复制 + sigma=0.02  抖动

预注册判据（写死在此，跑出不合意只改结论，不改判据）
----------------------------------------------------
  E1. gt_exact 的 CD 必须 < 1e-12（管线无偏移的硬验证）
  E2. gt_jitter_tiny 的 CD 必须【优于文献 SOTA 0.451e-3】
      —— 这是协议是否对齐的验收线：一个几乎等于 GT 的预测若还打不过文献值，
         说明量纲/协议仍有问题
  E3. jitter_large 的 CD 必须差于 jitter_small（噪声越大越差，单调性）
  E4. copy4 的 NUC 必须显著差于 gt_jitter_tiny（NUC 应能识别重复点造成的极端聚集）
  E5. copy4 的 P2F 优于 jitter_small —— 【预期成立的负面发现】，
      证明 P2F 对点分布不敏感，不能单独证明方法优劣，须写入 Limitations
  E6. gt_half_dup 的 CD 必须差于 gt_jitter_tiny
      —— 验证 v1 的构造确实不是上界（复现 v1 的错误以留证）

E5 不是失败，而是必须写进 Limitations 的证据。

诚信约束
--------
本脚本的数字是【参考标尺】，不是任何方法的性能声明。
任何参考上采样器都不是本文方法，也不是 baseline。
gt_* 系列全部使用了真值信息，属"作弊"参照，绝不可出现在论文性能对比表中。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import PU1KTestSet, farthest_point_sample  # noqa: E402
from puvnet.metrics.pointcloud import normalize_point_cloud  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import evaluate_one, upsample_model  # noqa: E402

OUT_DIR = ROOT / "runs" / "E000_metric_calibration"
SEED = 20260811
N_MODELS = 20          # 全量 127 太慢（P2F 是瓶颈），取前 20 个建立标尺

# 预注册常量：文献值，来自 arXiv 2111.12242 Tab.1（非本项目实测）
LIT_CD_SOTA = 0.451e-3
LIT_HD_SOTA = 3.843e-3
LIT_P2F_SOTA = 1.277e-3


class Copy4(torch.nn.Module):
    def __init__(self, r=4):
        super().__init__()
        self.r = r

    def forward(self, x):
        return x.repeat_interleave(self.r, dim=1)


class Jitter(torch.nn.Module):
    def __init__(self, r=4, sigma=0.01):
        super().__init__()
        self.r, self.sigma = r, sigma

    def forward(self, x):
        out = x.repeat_interleave(self.r, dim=1)
        g = torch.Generator(device=out.device).manual_seed(SEED)
        return out + torch.randn(out.shape, generator=g,
                                 device=out.device) * self.sigma


def _agg(rows: list[dict]) -> dict:
    """把逐模型结果聚合成均值（None 项跳过，不填估算值）。"""
    cds = [r["cd"] for r in rows]
    hds = [r["hd"] for r in rows]
    p2fs = [r["p2f_mean"] for r in rows if r["p2f_mean"] is not None]
    nucs = [r["nuc"] for r in rows if r["nuc"] is not None]
    return {
        "cd": float(np.mean(cds)), "hd": float(np.mean(hds)),
        "p2f": float(np.mean(p2fs)) if p2fs else None,
        "nuc": float(np.mean(nucs)) if nucs else None,
        "n": len(cds), "n_p2f": len(p2fs), "n_nuc": len(nucs),
    }


def _show(tag: str, r: dict) -> None:
    p2f = "    None" if r["p2f"] is None else f"{r['p2f']*1e3:>8.4f}"
    nuc = "     None" if r["nuc"] is None else f"{r['nuc']:>9.6f}"
    print(f"  {tag:<16} CD={r['cd']*1e3:>10.6f}  HD={r['hd']*1e3:>10.4f}  "
          f"P2F={p2f}  NUC={nuc}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = PU1KTestSet(input_n=2048, up_ratio=4)
    names = ts.names[:N_MODELS]
    rng = np.random.default_rng(SEED)

    print("=" * 90)
    print("E-000 指标校准 v2：建立四指标的行为标尺（PU-GCN 官方 squared 协议）")
    print("=" * 90)
    print(f"device={device}  测试模型数={len(names)}  seed={SEED}")
    print(f"协议依据: refs/pu_gcn/CD_PROTOCOL_SOURCE.md")
    print("⚠️ 本脚本产出的是参考标尺，不是任何方法的性能声明。")
    print("⚠️ gt_* 系列使用了真值信息，属作弊参照，绝不可进论文对比表。")
    print()

    # --- 组 A：经模型的参考上采样器（只看输入） ---
    refs = {
        "copy4": Copy4(4).to(device),
        "jitter_small": Jitter(4, 0.005).to(device),
        "jitter_large": Jitter(4, 0.02).to(device),
    }
    results: dict[str, dict] = {}
    per_model: dict[str, list] = {}

    print("[组 A] 仅用输入的参考上采样器（无真值信息）")
    for tag, model in refs.items():
        rows = []
        for name in names:
            inp, gt = ts.load(name)
            pred, _ = upsample_model(model, inp, 4, device=device, seed=SEED)
            rows.append(evaluate_one(pred, gt, mesh_path=ts.mesh_path(name),
                                     seed=SEED))
        results[tag] = _agg(rows)
        per_model[tag] = [{"cd": r["cd"], "hd": r["hd"],
                           "p2f": r["p2f_mean"], "nuc": r["nuc"]} for r in rows]
        _show(tag, results[tag])

    # --- 组 B：使用真值的作弊参照（建立上界与失效模式） ---
    print()
    print("[组 B] 使用真值的作弊参照（用于定位上界与指标失效模式）")

    def _gt_exact(gt):
        return gt.copy()

    def _gt_jitter_tiny(gt, i):
        r = np.random.default_rng(SEED + 1000 + i)
        return gt + r.standard_normal(gt.shape).astype(gt.dtype) * 0.001

    def _gt_half_dup(gt):
        idx = farthest_point_sample(gt, len(gt) // 2, seed=SEED)
        half = gt[idx]
        return np.concatenate([half, half], axis=0)[:len(gt)]

    cheat_builders = {
        "gt_exact": lambda gt, i: _gt_exact(gt),
        "gt_jitter_tiny": _gt_jitter_tiny,
        "gt_half_dup": lambda gt, i: _gt_half_dup(gt),
    }
    for tag, build in cheat_builders.items():
        rows = []
        for i, name in enumerate(names):
            inp, gt = ts.load(name)
            pred = build(gt, i)
            rows.append(evaluate_one(pred, gt, mesh_path=ts.mesh_path(name),
                                     seed=SEED))
        results[tag] = _agg(rows)
        per_model[tag] = [{"cd": r["cd"], "hd": r["hd"],
                           "p2f": r["p2f_mean"], "nuc": r["nuc"]} for r in rows]
        _show(tag, results[tag])

    # --- 预注册判据检验 ---
    print()
    print("=" * 90)
    print("预注册判据检验（判据在跑之前已写死在本文件顶部）")
    print("=" * 90)
    checks = {}

    c = results["gt_exact"]["cd"] < 1e-12
    checks["E1_gt_exact_zero"] = bool(c)
    print(f"  E1 gt_exact CD < 1e-12（管线无偏移）  : {'PASS' if c else 'FAIL'}  "
          f"(CD={results['gt_exact']['cd']:.3e})")

    up = results["gt_jitter_tiny"]["cd"]
    c = up < LIT_CD_SOTA
    checks["E2_upper_bound_beats_sota"] = bool(c)
    print(f"  E2 gt_jitter_tiny 优于文献 SOTA       : {'PASS' if c else 'FAIL'}  "
          f"({up*1e3:.6f}e-3 vs {LIT_CD_SOTA*1e3:.3f}e-3)")
    print(f"     -> 这是【协议是否对齐】的验收线")

    c = results["jitter_large"]["cd"] > results["jitter_small"]["cd"]
    checks["E3_noise_monotonic"] = bool(c)
    print(f"  E3 噪声越大 CD 越差（单调性）         : {'PASS' if c else 'FAIL'}  "
          f"({results['jitter_large']['cd']*1e3:.4f} > "
          f"{results['jitter_small']['cd']*1e3:.4f})")

    c = results["copy4"]["nuc"] > results["gt_jitter_tiny"]["nuc"]
    checks["E4_nuc_detects_dup"] = bool(c)
    print(f"  E4 NUC 能识别重复点聚集               : {'PASS' if c else 'FAIL'}  "
          f"(copy4 {results['copy4']['nuc']:.6f} vs "
          f"gt_jitter_tiny {results['gt_jitter_tiny']['nuc']:.6f})")

    c = results["copy4"]["p2f"] < results["jitter_small"]["p2f"]
    checks["E5_p2f_blind_to_distribution"] = bool(c)
    print(f"  E5 P2F 对分布不敏感（负面发现）       : {'成立' if c else '不成立'}  "
          f"(copy4 {results['copy4']['p2f']*1e3:.4f} < "
          f"jitter_small {results['jitter_small']['p2f']*1e3:.4f})")
    print("     -> 若成立：P2F 不可单独用于证明方法优劣，须写入 Limitations")

    c = results["gt_half_dup"]["cd"] > results["gt_jitter_tiny"]["cd"]
    checks["E6_half_dup_is_not_upper_bound"] = bool(c)
    print(f"  E6 gt_half_dup 差于真上界（证 v1 有误）: {'PASS' if c else 'FAIL'}  "
          f"({results['gt_half_dup']['cd']*1e3:.4f} > "
          f"{results['gt_jitter_tiny']['cd']*1e3:.6f})")

    all_pass = all(checks[k] for k in
                   ["E1_gt_exact_zero", "E2_upper_bound_beats_sota",
                    "E3_noise_monotonic", "E4_nuc_detects_dup",
                    "E6_half_dup_is_not_upper_bound"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E-000_metric_calibration",
        "version": "v2",
        "supersedes": "v1（L2 协议 + gt_half_dup 误当上界，已作废）",
        "protocol_source": "refs/pu_gcn/CD_PROTOCOL_SOURCE.md",
        "cd_hd_convention": "squared distance, CD=mean(fwd)+mean(bwd), "
                            "HD=max(max(fwd),max(bwd)), pred/gt normalized independently",
        "p2f_convention": "L2 distance, original (un-normalized) coordinate scale",
        "seed": SEED, "n_models": len(names), "device": device,
        "results": results,
        "per_model": per_model,
        "checks": checks,
        "all_hard_checks_pass": bool(all_pass),
        "note": "参考标尺，非任何方法的性能声明；gt_* 为作弊参照，不得进论文对比表",
        "literature_reference": {
            "PU-Transformer_PU1K_4x": {"cd": LIT_CD_SOTA, "hd": LIT_HD_SOTA,
                                       "p2f": LIT_P2F_SOTA},
            "source": "arXiv 2111.12242 Tab.1（文献值，非本项目实测）",
        },
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    print()
    print("=" * 90)
    print("与文献值的量级对照（仅供判断管线是否可信，非性能比较）")
    print("=" * 90)
    print(f"  文献 PU-Transformer PU1K 4x   : CD={LIT_CD_SOTA*1e3:.3f}  "
          f"HD={LIT_HD_SOTA*1e3:.3f}  P2F={LIT_P2F_SOTA*1e3:.3f} (×1e-3)")
    gj = results["gt_jitter_tiny"]
    print(f"  本管线 gt_jitter_tiny（真上界）: CD={gj['cd']*1e3:.6f}  "
          f"HD={gj['hd']*1e3:.4f}  P2F={gj['p2f']*1e3:.4f}")
    print()
    print(f"  硬判据（E1/E2/E3/E4/E6）全部通过? {all_pass}")
    print(f"\n结果已落盘: {OUT_DIR / 'result.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
