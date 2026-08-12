"""点云上采样评测入口 —— 所有论文性能数字的唯一出口。

评测协议（依据 refs/pu_gcn/CD_PROTOCOL_SOURCE.md —— 官方一手代码逐行核实）
------------------------------------------------------------------------
1. 输入整模型 (2048 点) -> patch 化 -> 逐 patch 上采样 -> 合并 -> FPS 降到 r*N
2. **CD / HD**：平方距离；CD = mean(pred->gt) + mean(gt->pred)；
   HD = max(max(pred->gt), max(gt->pred))；
   且 pred 与 gt **各自独立**归一化（官方 evaluate.py 两次调用 normalize_point_cloud）。
   统一走 `puvnet.metrics.pointcloud.eval_cd_hd_official`，本模块不手写归一化。
3. **P2F**：L2 距离（开方），在**原始坐标尺度**上对 original_meshes/*.off 计算
   （官方 evaluation.cpp 直接读 .off 与 .xyz，全程无归一化）。trimesh 替代 CGAL。
4. **NUC**：本项目简化实现（欧氏邻域 vs 官方测地 disk），仅作方法间相对比较，
   不与文献绝对值对表。

⚠️ 三个曾经写错过的地方，改动前务必先读 CD_PROTOCOL_SOURCE.md：
  - CD/HD 曾误用 L2 距离（"与 PU-GCN evaluate.py 一致"的注释是错的），
    导致数值偏离文献 25~50 倍；官方是**平方距离**。
  - 评测阶段 pred/gt 各自独立归一化；**训练阶段**则必须共享 gt 的 center/scale。
    两者不可混淆。
  - P2F 不做归一化。

patch 化推理的关键设计
----------------------
单个 patch 覆盖不了整个模型，必须分块。做法（沿用 PU-GCN/Grad-PU 思路）：
  a. 用 FPS 在输入上选 M 个 seed 点
  b. 每个 seed 取其 knn 邻域 (patch_size 点) 作为一个 patch
  c. patch 独立归一化 -> 模型上采样 -> 还原到原尺度
  d. 所有 patch 输出拼接（会有重叠冗余）-> FPS 降到目标点数 r*N

⚠️ 覆盖率必须检查：若 seed 数不足，模型部分区域没有任何 patch 覆盖，
   输出会出现空洞，CD 会异常偏大且难以定位原因。本模块显式检查并报警。

⚠️ **patch 内加噪的 sigma 是 patch 归一化坐标系下的相对量**，
   与在整模型原始坐标系加噪的 sigma 不可直接比较
   （实测 patch 最大半径 ≈0.18，整模型 ≈0.49，差约 2.7 倍）。
   E-000 的 jitter_* 参照器属前者，`exp_e000_aux_jitter.py` 属后者。

诚信约束
--------
- 本模块只计算并输出数字，不做任何「挑选最好的一次」逻辑。
- 所有随机性由 --seed 控制，同 seed 必须复现同一结果。
- 未跑的项输出 None，绝不填充估算值。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"E:\AE-CC托管\puv-net")
sys.path.insert(0, str(ROOT))

from puvnet.data.pu_dataset import (  # noqa: E402
    PU1KTestSet, farthest_point_sample, normalize_patch, add_gaussian_noise,
)
from puvnet.metrics.pointcloud import (  # noqa: E402
    chamfer_distance, hausdorff_distance, point_to_surface, uniformity_nuc,
    normalize_point_cloud, eval_cd_hd_official,
)


# --------------------------------------------------------------------------
# 环境指纹（复现性）
# --------------------------------------------------------------------------
def env_fingerprint() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(0)))
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------------
# patch 化推理
# --------------------------------------------------------------------------
def extract_patches(pc: np.ndarray, patch_size: int, n_patches: int,
                    seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """用 FPS seed + knn 邻域切 patch。

    返回 (patch 列表, 覆盖计数)。覆盖计数用于检查是否有点未被任何 patch 覆盖。
    """
    from scipy.spatial import cKDTree

    n = len(pc)
    patch_size = min(patch_size, n)
    seed_idx = farthest_point_sample(pc, min(n_patches, n), seed=seed)
    tree = cKDTree(pc)
    patches, coverage = [], np.zeros(n, dtype=np.int64)
    for si in seed_idx:
        _, idx = tree.query(pc[si], k=patch_size)
        idx = np.atleast_1d(idx)
        patches.append(pc[idx])
        coverage[idx] += 1
    return patches, coverage


@torch.no_grad()
def upsample_model(model, pc: np.ndarray, up_ratio: int,
                   patch_size: int = 256, patch_mult: float = 3.0,
                   batch: int = 32, device: str = "cuda",
                   seed: int = 0) -> tuple[np.ndarray, dict]:
    """对整模型点云做 patch 化上采样。

    patch_mult : seed 数 = ceil(N / patch_size * patch_mult)。
        取 3.0 保证充分重叠覆盖；过小会留空洞。

    返回 (上采样点云 r*N 点, 诊断信息)
    """
    n = len(pc)
    target = n * up_ratio
    n_patches = int(np.ceil(n / patch_size * patch_mult))

    patches, coverage = extract_patches(pc, patch_size, n_patches, seed=seed)
    uncovered = int((coverage == 0).sum())

    # 逐 patch 归一化 -> 推理 -> 还原
    model.eval()
    outs = []
    norm_data = []
    for p in patches:
        q, c, s = normalize_patch(p)
        norm_data.append((c, s))
        outs.append(q)
    arr = np.stack(outs).astype(np.float32)            # (M, patch_size, 3)

    preds = []
    for i in range(0, len(arr), batch):
        chunk = torch.from_numpy(arr[i:i + batch]).to(device)
        out = model(chunk)                             # (B, patch_size*r, 3)
        preds.append(out.cpu().numpy())
    pred = np.concatenate(preds, axis=0)               # (M, patch_size*r, 3)

    # 还原到原尺度并拼接
    merged = []
    for k, (c, s) in enumerate(norm_data):
        merged.append(pred[k] * s + c)
    merged = np.concatenate(merged, axis=0)

    # FPS 降到目标点数（去除 patch 重叠冗余）
    if len(merged) > target:
        idx = farthest_point_sample(merged, target, seed=seed)
        final = merged[idx]
    else:
        final = merged

    diag = {
        "n_input": n,
        "n_patches": len(patches),
        "patch_size": patch_size,
        "n_merged_before_fps": int(len(merged)),
        "n_final": int(len(final)),
        "n_target": target,
        "uncovered_points": uncovered,
        "coverage_mean": float(coverage.mean()),
        "coverage_min": int(coverage.min()),
    }
    return final, diag


# --------------------------------------------------------------------------
# 单模型评测
# --------------------------------------------------------------------------
def evaluate_one(pred: np.ndarray, gt: np.ndarray,
                 mesh_path: Path | None = None,
                 compute_p2f: bool = True,
                 compute_nuc: bool = True,
                 seed: int = 0) -> dict:
    """对单个模型算四指标 —— 严格对齐 PU-GCN 官方评测协议。

    协议依据：`refs/pu_gcn/CD_PROTOCOL_SOURCE.md`（含官方源码 sha256 存证）。
    以下三条是曾经踩过坑的地方，改动前务必先读该文件：

    1. **CD / HD 用平方距离**，且 pred 与 gt **各自独立**归一化
       （官方 evaluate.py 两次调用 normalize_point_cloud，返回的 centroid/scale 未复用）。
       统一走 `eval_cd_hd_official`，不在本函数里手写归一化。
       ⚠️ 注意与训练阶段区分：训练时 input/gt 必须共享 gt 的 center/scale。
    2. **P2F 在原始坐标尺度上算**（官方 evaluation.cpp 直接读 .off 与 .xyz，无归一化），
       距离本体是 L2（开方），与 CD/HD 的平方量纲不同。
    3. NUC 是本项目简化实现，只可做方法间相对比较，不与文献绝对值对表。
    """
    res: dict = dict(eval_cd_hd_official(pred, gt))
    res.update({"p2f_mean": None, "p2f_std": None, "nuc": None})

    if compute_p2f and mesh_path is not None and Path(mesh_path).exists():
        try:
            import trimesh
            mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
            # 官方协议：原始尺度，mesh 与 pred 均不归一化
            m, s = point_to_surface(np.asarray(pred, dtype=np.float64), mesh)
            res["p2f_mean"], res["p2f_std"] = m, s
            # 供跨模型可比性核查：PU1K 的 mesh 已预归一化，此值应大致同量级
            res["p2f_gt_scale"] = res["gt_scale"]
        except Exception as e:               # 不让单个 mesh 失败拖垮整轮评测
            res["p2f_error"] = f"{type(e).__name__}: {e}"

    if compute_nuc:
        try:
            # NUC 内部会自行归一化，传原始 pred 即可
            res["nuc"] = uniformity_nuc(pred, seed=seed)["nuc_mean"]
        except Exception as e:
            res["nuc_error"] = f"{type(e).__name__}: {e}"

    return res


# --------------------------------------------------------------------------
# 全测试集评测
# --------------------------------------------------------------------------
def evaluate_dataset(model, up_ratio: int = 4, input_n: int = 2048,
                     device: str = "cuda", patch_size: int = 256,
                     patch_mult: float = 3.0, noise_beta: float = 0.0,
                     limit: int | None = None, seed: int = 0,
                     save_clouds: Path | None = None,
                     verbose: bool = True) -> dict:
    """在 PU1K 测试集上评测，返回逐模型与汇总结果。"""
    ts = PU1KTestSet(input_n=input_n, up_ratio=up_ratio)
    names = ts.names if limit is None else ts.names[:limit]

    per_model, diags = {}, {}
    t0 = time.time()
    for i, name in enumerate(names):
        inp, gt = ts.load(name)
        if noise_beta > 0:
            # 噪声按整模型半径比例注入，与训练时约定一致
            _, _, s = normalize_point_cloud(inp)
            inp = inp + np.random.default_rng(seed + i).standard_normal(
                inp.shape).astype(np.float32) * (noise_beta * s)

        pred, diag = upsample_model(
            model, inp, up_ratio, patch_size=patch_size,
            patch_mult=patch_mult, device=device, seed=seed)
        res = evaluate_one(pred, gt, mesh_path=ts.mesh_path(name), seed=seed)
        per_model[name] = res
        diags[name] = diag

        if save_clouds is not None:
            save_clouds = Path(save_clouds)
            save_clouds.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_clouds / f"{name}.npz",
                                input=inp, pred=pred, gt=gt)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(names)}] {name[:40]:<40} "
                  f"CD={res['cd']*1e3:.4f}")

    # 汇总（只对成功算出的项取均值）
    def agg(key: str):
        vals = [v[key] for v in per_model.values() if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    total_uncovered = sum(d["uncovered_points"] for d in diags.values())
    summary = {
        "n_models": len(names),
        "cd_mean": agg("cd"),
        "hd_mean": agg("hd"),
        "p2f_mean": agg("p2f_mean"),
        "nuc_mean": agg("nuc"),
        "elapsed_sec": time.time() - t0,
        "total_uncovered_points": total_uncovered,
        "p2f_failures": sum(1 for v in per_model.values() if "p2f_error" in v),
    }
    return {"summary": summary, "per_model": per_model, "diagnostics": diags,
            "config": {
                "up_ratio": up_ratio, "input_n": input_n,
                "patch_size": patch_size, "patch_mult": patch_mult,
                "noise_beta": noise_beta, "seed": seed,
                "cd_squared": False, "protocol_note":
                    "CD/HD 在 gt 归一化尺度下计算；NUC 为简化实现仅作相对比较",
            },
            "env": env_fingerprint()}


# --------------------------------------------------------------------------
# 自检：用「恒等上采样」验证评测管线本身是否正确
# --------------------------------------------------------------------------
class _RepeatUpsampler(torch.nn.Module):
    """把每个点复制 r 份的假模型。用于验证评测管线的合理性下界。

    预期：CD 不会是 0（点数对但分布是重复的），但也不会离谱地大。
    这是检查 patch 拼接/还原/FPS 是否搞错坐标系的有效手段 ——
    若坐标系错了，CD 会大到荒谬。
    """

    def __init__(self, r: int) -> None:
        super().__init__()
        self.r = r

    def forward(self, x):                      # (B, N, 3) -> (B, N*r, 3)
        return x.repeat_interleave(self.r, dim=1)


class _JitterUpsampler(torch.nn.Module):
    """复制 + 小抖动。比纯复制更接近真实上采样器的行为。"""

    def __init__(self, r: int, sigma: float = 0.02) -> None:
        super().__init__()
        self.r, self.sigma = r, sigma

    def forward(self, x):
        out = x.repeat_interleave(self.r, dim=1)
        g = torch.Generator(device=out.device).manual_seed(0)
        return out + torch.randn(out.shape, generator=g,
                                 device=out.device) * self.sigma


def self_check() -> bool:
    ok = True
    print("=" * 78)
    print("evaluate 自检（用假模型验证评测管线，不涉及任何真实性能声明）")
    print("=" * 78)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # 1. patch 覆盖率
    ts = PU1KTestSet()
    inp, gt = ts.load(ts.names[0])
    patches, coverage = extract_patches(inp, 256, 24, seed=0)
    print(f"[1] patch 数={len(patches)}  覆盖 min={coverage.min()} "
          f"mean={coverage.mean():.2f}  未覆盖点={int((coverage==0).sum())}")
    ok &= len(patches) == 24

    # 2. 覆盖率不足时应能检测出来（故意只用 2 个 patch）
    _, cov_bad = extract_patches(inp, 256, 2, seed=0)
    unc = int((cov_bad == 0).sum())
    print(f"[2] 仅 2 patch 时未覆盖点={unc}  (应远大于 0，证明检查有效)")
    ok &= unc > 0

    # 3. 恒等复制上采样 —— 验证坐标系没搞错
    m_rep = _RepeatUpsampler(4).to(device)
    pred, diag = upsample_model(m_rep, inp, 4, device=device, seed=0)
    res = evaluate_one(pred, gt, mesh_path=ts.mesh_path(ts.names[0]), seed=0)
    print(f"[3] 复制上采样: n_final={diag['n_final']} (目标 {diag['n_target']})")
    p2f_txt = "None" if res["p2f_mean"] is None else f"{res['p2f_mean']*1e3:.4f}"
    print(f"    CD={res['cd']*1e3:.4f}  HD={res['hd']*1e3:.4f}  P2F={p2f_txt}")
    ok &= diag["n_final"] == diag["n_target"]
    # 坐标系正确时，CD 应在合理范围（<50e-3）；坐标系错则会爆到几百
    ok &= res["cd"] * 1e3 < 50.0
    print(f"    坐标系合理性: CD*1e3={res['cd']*1e3:.4f} < 50 -> "
          f"{'OK' if res['cd']*1e3 < 50 else 'FAIL(坐标系可能错)'}")

    # 4. squared CD 口径下，抖动上采样应【显著优于】纯复制
    #    机理：复制上采样产生大量共位点，gt->pred 方向存在大量"无点可覆盖"的
    #    远距离最近邻，平方后被进一步放大。这是官方 squared 协议相对 L2 协议
    #    的一个实际差异，也是本项目从 L2 改回 squared 后的可观测后果。
    #    预注册判据：cd_jitter < cd_copy（不设倍数下限，只判方向）
    m_jit = _JitterUpsampler(4, sigma=0.01).to(device)
    pred2, _ = upsample_model(m_jit, inp, 4, device=device, seed=0)
    res2 = evaluate_one(pred2, gt, mesh_path=None, compute_p2f=False, seed=0)
    cd_jit, cd_cp = res2["cd"] * 1e3, res["cd"] * 1e3
    p4 = cd_jit < cd_cp
    print(f"[4] 抖动上采样: CD={cd_jit:.4f}  复制版 CD={cd_cp:.4f}  "
          f"抖动优于复制? {p4}  (squared 口径下应成立)")
    ok &= p4
    ok &= 0.01 < cd_jit < 200.0

    # 5. 同 seed 复现性
    p_a, _ = upsample_model(m_rep, inp, 4, device=device, seed=7)
    p_b, _ = upsample_model(m_rep, inp, 4, device=device, seed=7)
    same = np.allclose(p_a, p_b)
    print(f"[5] 同 seed 两次结果完全一致? {same}")
    ok &= same

    # 6. P2F 能算出来（trimesh 读 .off）
    has_p2f = res["p2f_mean"] is not None
    print(f"[6] P2F 计算成功? {has_p2f}  "
          f"{'' if has_p2f else res.get('p2f_error', '')}")
    ok &= has_p2f

    print()
    print(f"自检结果: {'ALL PASS' if ok else 'FAILED'}")
    print()
    print("⚠️ 本自检只验证评测【管线】正确性，假模型的 CD 值不构成任何性能声明。")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="点云上采样评测")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--up-ratio", type=int, default=4)
    ap.add_argument("--input-n", type=int, default=2048)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--patch-mult", type=float, default=3.0)
    ap.add_argument("--noise-beta", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save-clouds", type=str, default=None)
    args = ap.parse_args()

    if args.self_check:
        return 0 if self_check() else 1

    if args.ckpt is None:
        print("需要 --ckpt 或 --self-check", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from puvnet.models.pu_transformer import PUTransformer
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("model_config", {})
    model = PUTransformer(up_ratio=args.up_ratio, **cfg).to(device)
    model.load_state_dict(ck["model"])

    out = evaluate_dataset(
        model, up_ratio=args.up_ratio, input_n=args.input_n, device=device,
        patch_size=args.patch_size, patch_mult=args.patch_mult,
        noise_beta=args.noise_beta, limit=args.limit, seed=args.seed,
        save_clouds=Path(args.save_clouds) if args.save_clouds else None)

    s = out["summary"]
    print()
    print("=" * 60)
    print("评测汇总（单位 ×10⁻³）")
    print("=" * 60)
    for k in ("cd_mean", "hd_mean", "p2f_mean", "nuc_mean"):
        v = s[k]
        print(f"  {k:<12} = {'None' if v is None else f'{v*1e3:.4f}'}")
    print(f"  用时         = {s['elapsed_sec']:.1f}s")
    print(f"  未覆盖点总数 = {s['total_uncovered_points']}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                default=float), encoding="utf-8")
        print(f"\n结果已落盘: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
