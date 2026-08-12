"""点云上采样训练主脚本（新命题：PUTG-Net = PU-Transformer + PU-GAN）。

与旧 `scripts/train.py` 的关系
-----------------------------
旧 `train.py` 属旧命题（「二维表面推理三维内部结构」）遗留，依赖 `PUVNet` /
`losses.reconstruction.total_loss` / `surface|interior` 分阶段，**与新命题完全无关**，
不做修改也不复用。本文件是新命题的唯一训练入口。

设计要点
--------
1. **一切由 config 驱动**：消融只改 yaml 不改代码（项目铁律）。
   B-001 纯复现 = `loss.w_adv/w_uniform/w_repulsion` 全 0。
2. **产物严格照 `docs/ARTIFACT_POLICY.md`**：
   config.yaml / env.json / history.json / metrics.json / ckpt/{best,last}.pt /
   figures/*.png + 同名 .data.json / clouds/*.npz。
   过程图训练完就补不回来，故中间 epoch 的固定样本输出点云必须当场落盘。
3. **训练 loss 与评测指标严格分离**：
   - 训练：`puvnet.losses.upsampling`（torch，可微，patch 内 input/gt 共享归一化）
   - 论文数字：`scripts/evaluate.py`（numpy，pred/gt 各自独立归一化，官方协议）
   本脚本内的 val 指标属**训练期监控**，用于早停与曲线，
   **不得直接写进论文主表** —— 主表必须由 `evaluate.py` 在测试集上产出。
   history.json / metrics.json 中已用 `monitor_` 前缀显式标记这一点。
4. **M2（判别器过强）**：`d_steps` 控制 D:G 更新比。
5. **固定验证样本索引写死在 config**，所有 run 共用，跨 run 定性图才可比。

用法
----
    python scripts/train_pu.py --config configs/r001_local_smoke.yaml
    python scripts/train_pu.py --config configs/b001_reproduce.yaml --override epochs=5
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
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.data.pu_dataset import PUTrainDataset
from puvnet.losses.upsampling import UpsamplingLoss, chamfer_loss_split
from puvnet.metrics.pointcloud import eval_cd_hd_official, uniformity_nuc
from puvnet.metrics.selection import (CompositeSelector, convergence_check,
                                      plateau_stats)
from puvnet.models.pu_gan import PUGANDiscriminator, PUTransGAN
from puvnet.models.pu_transformer import PUTransformer
from puvnet.viz.visualize import (
    plot_metric_curves,
    plot_nn_histogram,
    plot_point_clouds,
    plot_training_curves,
)


# =============================================================================
# 环境指纹
# =============================================================================

def env_fingerprint() -> dict:
    """采集复现所需的环境指纹。跨机器对比（本机 3090 vs 云端）时靠它排查差异。"""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["gpu_total_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    # git commit（不在 git 仓库时留 None，不报错）
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


# =============================================================================
# 模型构建
# =============================================================================

def build_model(cfg: dict) -> PUTransGAN:
    """按 config 建模。判别器只在 w_adv>0 时建 —— 否则纯 Transformer 基线。"""
    mcfg = dict(cfg["model"])
    gen = PUTransformer(**mcfg)

    need_d = float(cfg["loss"].get("w_adv", 0.0)) > 0
    dis = PUGANDiscriminator(**cfg.get("discriminator", {})) if need_d else None
    return PUTransGAN(gen, discriminator=dis,
                      gan_mode=cfg.get("gan_mode", "hinge"))


def build_loss(cfg: dict) -> UpsamplingLoss:
    return UpsamplingLoss(**cfg["loss"])


# =============================================================================
# 训练期监控指标（注意：不是论文数字）
# =============================================================================

@torch.no_grad()
def monitor_eval(model, loader, loss_fn, device, epoch: int) -> dict:
    """验证集上的训练期监控。

    ⚠️ 这里算的是 **patch 级** 指标，且走 `eval_cd_hd_official`（pred/gt 各自独立
    归一化）。它与论文主表的差别在于：主表是**整模型**评测（2048→8192，含 patch
    合并与 FPS 重采样），且带 P2F。故两者数值不可直接互比，只能各自看趋势。
    """
    model.eval()
    agg_loss: dict[str, float] = {}
    cds, hds, nucs = [], [], []
    nb = 0
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        pred = model(inp)
        _, logs = loss_fn(pred, gt, model=None if not loss_fn.needs_gan else model,
                          epoch=epoch)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                agg_loss[k] = agg_loss.get(k, 0.0) + float(v)
        nb += 1
        # 指标只在第一个 batch 的前若干样本上算（numpy KD-tree 较慢，避免拖慢训练）
        if nb == 1:
            p_np = pred.detach().cpu().numpy()
            g_np = gt.detach().cpu().numpy()
            for j in range(min(4, len(p_np))):
                r = eval_cd_hd_official(p_np[j], g_np[j])
                cds.append(r["cd"])
                hds.append(r["hd"])
                # uniformity_nuc 返回按半径分档的 dict，取 nuc_mean 作监控标量
                nucs.append(uniformity_nuc(p_np[j])["nuc_mean"])
    model.train()

    out = {f"monitor_loss_{k}": v / max(nb, 1) for k, v in agg_loss.items()}
    out["monitor_cd"] = float(np.mean(cds)) if cds else None
    out["monitor_hd"] = float(np.mean(hds)) if hds else None
    out["monitor_nuc"] = float(np.mean(nucs)) if nucs else None
    return out


@torch.no_grad()
def dump_fixed_samples(model, ds, indices: list[int], device,
                       clouds_dir: Path, tag: str) -> list[dict]:
    """把固定验证样本的输出点云落盘。

    **这一步不能省**：定性演进图（epoch 10 vs 50 vs 100 的输出对比）
    事后无法重建，训练完权重覆盖了就永久丢失。
    """
    model.eval()
    recs = []
    clouds_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        inp, gt = ds[idx]
        pred = model(inp.unsqueeze(0).to(device))[0].cpu().numpy()
        f = clouds_dir / f"{tag}_idx{idx:05d}.npz"
        np.savez_compressed(f, input=inp.numpy(), gt=gt.numpy(), pred=pred)
        recs.append({"index": idx, "file": str(f.name),
                     "n_input": int(len(inp)), "n_pred": int(len(pred)),
                     "n_gt": int(len(gt))})
    model.train()
    return recs


# =============================================================================
# 主训练循环
# =============================================================================

def train(cfg: dict) -> dict:
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(cfg["out_dir"])
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    cloud_dir = out_dir / "clouds"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- 复现性产物：配置快照 + 环境指纹（先写，训练崩了也留得下） ---
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    env = env_fingerprint()
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    # --- 数据 ---
    dcfg = dict(cfg["data"])
    val_ratio = float(dcfg.pop("val_ratio", 0.05))
    ds_full = PUTrainDataset(seed=seed, **dcfg)
    n_all = len(ds_full)
    n_val = max(1, int(n_all * val_ratio))
    # 划分固定：用固定 seed 的 permutation，保证跨 run 完全一致
    perm = np.random.default_rng(12345).permutation(n_all)
    val_idx = perm[:n_val].tolist()
    tr_idx = perm[n_val:].tolist()

    # 训练集不做增强关闭，验证集必须关增强+关噪声（否则监控指标带随机性不可比）
    dcfg_val = dict(dcfg)
    dcfg_val["augment"] = False
    dcfg_val["noise_beta"] = 0.0
    ds_val_base = PUTrainDataset(seed=seed, **dcfg_val)

    ds_tr = Subset(ds_full, tr_idx)
    ds_va = Subset(ds_val_base, val_idx)

    # dataloader 配置从 cfg["loader"] 读, 默认全 0 保持向后兼容 (2026-08-12 优化
    # 见 commits: 5090 上加 workers/pin_memory/persistent, 本机 3090 不动)。
    # 安全前提: PUTrainDataset.__getitem__ 用 self.seed+i 作 RNG, 不依赖全局随机,
    # 多 worker 不会让 augment 结果发散, 跨 run 跨机器可复现。
    lcfg = cfg.get("loader", {}) or {}
    loader_kwargs_tr = dict(
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=int(lcfg.get("num_workers", 0)),
        pin_memory=bool(lcfg.get("pin_memory", False)),
    )
    loader_kwargs_va = dict(
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=int(lcfg.get("num_workers", 0)),
        pin_memory=bool(lcfg.get("pin_memory", False)),
    )
    if loader_kwargs_tr["num_workers"] > 0:
        loader_kwargs_tr["prefetch_factor"] = int(lcfg.get("prefetch_factor", 2))
        loader_kwargs_tr["persistent_workers"] = bool(
            lcfg.get("persistent_workers", True))
    if loader_kwargs_va["num_workers"] > 0:
        loader_kwargs_va["prefetch_factor"] = int(lcfg.get("prefetch_factor", 2))
        loader_kwargs_va["persistent_workers"] = bool(
            lcfg.get("persistent_workers", True))
    dl_tr = DataLoader(ds_tr, **loader_kwargs_tr)
    dl_va = DataLoader(ds_va, **loader_kwargs_va)

    # 固定样本：取验证集划分里的前 K 个（索引来自固定 permutation，故跨 run 一致）
    n_fixed = int(cfg.get("n_fixed_samples", 3))
    fixed_idx = val_idx[:n_fixed]

    # --- 模型与损失 ---
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    pc = model.count_parameters()
    print(f"[模型] G={pc['generator']:,} D={pc['discriminator']:,} "
          f"total={pc['total']:,}")
    print(f"[数据] train={len(ds_tr)} val={len(ds_va)} "
          f"fixed_samples={fixed_idx}")
    print(f"[损失] {loss_fn.config()}")
    print(f"[环境] {env.get('gpu_name', 'CPU')} torch={env['torch']} "
          f"cuda={env['cuda_version']}")
    print(f"[loader] num_workers={loader_kwargs_tr['num_workers']} "
          f"pin_memory={loader_kwargs_tr['pin_memory']} "
          f"persistent={loader_kwargs_tr.get('persistent_workers', False)} "
          f"prefetch={loader_kwargs_tr.get('prefetch_factor', '-')}")

    g_params = [p for p in model.generator.parameters() if p.requires_grad]
    opt_g = torch.optim.Adam(g_params, lr=cfg["lr"],
                             betas=tuple(cfg.get("betas", (0.9, 0.999))))
    n_ep = int(cfg["epochs"])
    sched_g = torch.optim.lr_scheduler.StepLR(
        opt_g, step_size=int(cfg.get("lr_step", 20)),
        gamma=float(cfg.get("lr_decay", 0.7)))

    opt_d = None
    if loss_fn.needs_gan:
        opt_d = torch.optim.Adam(model.discriminator.parameters(),
                                 lr=float(cfg.get("lr_d", cfg["lr"])),
                                 betas=tuple(cfg.get("betas", (0.9, 0.999))))
    d_steps = int(cfg.get("d_steps", 1))          # M2 开关

    # --- 循环 ---
    history: list[dict] = []
    metrics_log: list[dict] = []
    cloud_manifest: list[dict] = []
    best_cd = float("inf")            # 影子对照：cd-only 若选点会选谁（零成本，仅记录）
    best_cd_epoch: int | None = None
    # 多指标综合选点（定案 2026-08-11）。依据与复盘见 docs/EXPERIMENT_LOG.md
    # 「选点复盘」一节：在 B-001 上与 cd-only 等价，但对改进 B/C 组是必要防护。
    selector = CompositeSelector(weights=cfg.get("select_weights"),
                                 warmup=int(cfg.get("select_warmup", 5)))
    eval_every = int(cfg.get("eval_every", 1))
    dump_every = int(cfg.get("dump_cloud_every", 10))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for ep in range(n_ep):
        t0 = time.time()
        agg: dict[str, float] = {}
        nb = 0
        for inp, gt in dl_tr:
            inp, gt = inp.to(device), gt.to(device)

            # ---- 判别器（M2：每 d_steps 个 G step 更新 1 次 D）----
            if opt_d is not None and nb % d_steps == 0:
                with torch.no_grad():
                    fake = model(inp)
                d_loss = model.d_loss(gt, fake)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                agg["d_loss"] = agg.get("d_loss", 0.0) + float(d_loss.item())

            # ---- 生成器 ----
            pred = model(inp)
            loss, logs = loss_fn(pred, gt,
                                 model=model if loss_fn.needs_gan else None,
                                 epoch=ep)
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(g_params, float(cfg["grad_clip"]))
            opt_g.step()

            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + float(v)
                else:
                    # 非数值（如 adv_adaptive_error）必须留痕，不静默丢弃
                    agg.setdefault("_notes", [])
                    if v not in agg["_notes"]:
                        agg["_notes"].append(v)
            nb += 1

        sched_g.step()
        tr = {f"train_{k}": (v / nb if isinstance(v, (int, float)) else v)
              for k, v in agg.items() if k != "_notes"}
        rec = {"epoch": ep, **tr, "lr": sched_g.get_last_lr()[0],
               "sec": round(time.time() - t0, 2)}
        if agg.get("_notes"):
            rec["notes"] = agg["_notes"]
        if torch.cuda.is_available():
            rec["gpu_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 3)

        # ---- 验证 ----
        if (ep + 1) % eval_every == 0 or ep == n_ep - 1:
            mv = monitor_eval(model, dl_va, loss_fn, device, ep)
            rec.update(mv)
            metrics_log.append({"epoch": ep, **mv})

            # 影子对照：只记录 cd-only 会选哪个 epoch，不据此存权重
            cd_now = mv.get("monitor_cd")
            if cd_now is not None and cd_now < best_cd:
                best_cd = cd_now
                best_cd_epoch = ep

            # 正式选点：多指标综合分
            score = selector.update(epoch=ep, cd=mv.get("monitor_cd"),
                                    hd=mv.get("monitor_hd"),
                                    nuc=mv.get("monitor_nuc"))
            rec["select_score"] = score
            metrics_log[-1]["select_score"] = score
            if selector.is_best:
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "epoch": ep, "select_score": score,
                            "select_weights": selector.weights,
                            "monitor_cd": mv.get("monitor_cd"),
                            "monitor_hd": mv.get("monitor_hd"),
                            "monitor_nuc": mv.get("monitor_nuc")},
                           out_dir / "ckpt" / "best.pt")
            print(f"  ep{ep:03d} total={rec.get('train_total', 0):.6f} "
                  f"cd={rec.get('train_cd', 0):.6f} "
                  f"| mon_cd={_fmt(mv.get('monitor_cd'))} "
                  f"mon_hd={_fmt(mv.get('monitor_hd'))} "
                  f"mon_nuc={_fmt(mv.get('monitor_nuc'))} "
                  f"score={_fmt(score)}{'*' if selector.is_best else ''} "
                  f"({rec['sec']}s"
                  + (f", {rec['gpu_peak_gb']}GB)" if 'gpu_peak_gb' in rec else ")"))
        else:
            print(f"  ep{ep:03d} total={rec.get('train_total', 0):.6f} "
                  f"cd={rec.get('train_cd', 0):.6f} ({rec['sec']}s)")

        history.append(rec)

        # ---- 固定样本点云落盘（事后不可重建，必须当场存）----
        if (ep + 1) % dump_every == 0 or ep == n_ep - 1:
            recs = dump_fixed_samples(model, ds_val_base, fixed_idx, device,
                                      cloud_dir, f"epoch{ep:03d}")
            cloud_manifest.append({"epoch": ep, "samples": recs})

        # 每 epoch 都覆盖 last.pt（云端断点续跑靠它）
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                    "opt_g": opt_g.state_dict()},
                   out_dir / "ckpt" / "last.pt")

        # 标量每 epoch 都刷盘，中途崩溃也不丢已跑数据
        _dump_json(out_dir / "history.json", history)
        _dump_json(out_dir / "metrics.json",
                   {"note": "训练期监控指标，非论文数字；论文主表见 evaluate.py",
                    "records": metrics_log})
        _dump_json(out_dir / "clouds_manifest.json", cloud_manifest)
        # 选点过程同样每 epoch 刷盘：best.pt 是怎么选出来的必须可审计
        _dump_json(out_dir / "selection.json",
                   {"note": "多指标综合选点过程；影子对照为 cd-only 的选择",
                    "summary": selector.summary(),
                    "shadow_cd_only": {"best_cd": (best_cd if best_cd < float("inf")
                                                   else None),
                                       "best_cd_epoch": best_cd_epoch},
                    "records": selector.records})

    # --- 收尾统计：平台区均值±σ（论文主表出口）+ 收敛检查 ---
    plateau = plateau_stats(metrics_log, frac=float(cfg.get("plateau_frac", 0.5)))
    conv = {f"w{w}": convergence_check(metrics_log, window=w)
            for w in (5, 10, 15, 20, 25)}
    _dump_json(out_dir / "summary_stats.json",
               {"note": ("平台区均值±σ 为论文主表报数方式（定案 2026-08-11）；"
                         "收敛判据须多窗口同看，单窗口结论会翻转"),
                "plateau": plateau, "convergence_multi_window": conv,
                "selection": selector.summary(),
                "shadow_cd_only_epoch": best_cd_epoch})

    # --- 过程图（全部由落盘数据重绘，不手填数字）---
    figs = make_figures(history, metrics_log, cloud_dir, fixed_idx, fig_dir)

    print(f"\n[完成] 产物目录 {out_dir}")
    print(f"       综合选点 best = ep{selector.best_epoch} "
          f"(score={_fmt(selector.best_score)})")
    print(f"       影子对照 cd-only 会选 = ep{best_cd_epoch} "
          f"(cd={_fmt(best_cd)})"
          + ("  ← 两准则一致" if best_cd_epoch == selector.best_epoch
             else "  ← ★两准则分歧，须在论文中说明"))
    for k in ("cd", "hd", "nuc"):
        s = plateau.get(k, {})
        if s.get("plateau_mean") is not None:
            print(f"       平台区 {k}: {_fmt(s['plateau_mean'])} ± "
                  f"{_fmt(s['plateau_std'])}  (最优 {_fmt(s['best'])} @ep{s['best_epoch']})")
    print(f"       图 {len(figs)} 张: {[f.name for f in figs]}")
    return {"out_dir": str(out_dir),
            "best_epoch": selector.best_epoch,
            "best_score": selector.best_score,
            "shadow_cd_only_epoch": best_cd_epoch,
            "best_monitor_cd": best_cd,
            "plateau": plateau,
            "history": history, "figures": [str(f) for f in figs]}


def _fmt(v) -> str:
    return "n/a" if v is None or v == float("inf") else f"{v:.6f}"


def _dump_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_figures(history: list[dict], metrics_log: list[dict],
                 cloud_dir: Path, fixed_idx: list[int],
                 fig_dir: Path) -> list[Path]:
    """由落盘数据重绘所有过程图。ARTIFACT_POLICY §4 的对应关系在此实现。"""
    figs: list[Path] = []

    # F_loss：训练曲线
    keys = [k for k in history[0] if k.startswith("train_")]
    hist = {"epoch": [r["epoch"] for r in history], "lr": [r["lr"] for r in history]}
    for k in keys:
        vals = [r.get(k) for r in history]
        if all(isinstance(v, (int, float)) for v in vals):
            hist[k] = vals
    figs.append(plot_training_curves(hist, fig_dir / "F_loss.png",
                                     title="训练损失分项"))

    # F_metric：监控指标曲线
    if metrics_log:
        eps = [r["epoch"] for r in metrics_log]
        mk = {}
        for name in ("monitor_cd", "monitor_hd"):
            vals = [r.get(name) for r in metrics_log]
            if all(isinstance(v, (int, float)) for v in vals):
                mk[name] = vals
        if mk:
            figs.append(plot_metric_curves(eps, mk, fig_dir / "F_metric.png"))

    # F_cloud / F_hist：最后一次固定样本
    last_files = sorted(cloud_dir.glob("epoch*_idx*.npz"))
    if last_files:
        latest_tag = last_files[-1].name.split("_idx")[0]
        for idx in fixed_idx:
            f = cloud_dir / f"{latest_tag}_idx{idx:05d}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            clouds = {"input": d["input"], "pred": d["pred"], "gt": d["gt"]}
            figs.append(plot_point_clouds(
                clouds, fig_dir / f"F_cloud_idx{idx:05d}.png"))
            figs.append(plot_nn_histogram(
                {"pred": d["pred"], "gt": d["gt"]},
                fig_dir / f"F_hist_idx{idx:05d}.png"))
    return figs


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="点分路径覆盖，如 epochs=3 loss.w_adv=0.1")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for kv in a.override:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v

    train(cfg)


if __name__ == "__main__":
    main()
