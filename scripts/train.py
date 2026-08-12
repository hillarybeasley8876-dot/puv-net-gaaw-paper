"""训练脚本。

支持 H5（分离训练 vs 端到端）的三种 stage：
  joint     表面 + 内部 联合训练（端到端）
  surface   只训表面分支
  interior  冻结编码器与表面头，只训内部头

H5 的验证方式：surface → interior 两阶段 与 joint 单阶段，
在相同总 epoch 预算下比较最终指标。这才是公平对比 ——
原稿声称分离训练更好但没有控制总计算量，那样的比较无效。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from puvnet.data.dataset import VolumetricPCDataset, collate
from puvnet.losses.reconstruction import total_loss
from puvnet.models.puvnet import PUVNet


def set_stage(model: PUVNet, stage: str) -> None:
    """按训练阶段冻结/解冻参数（H5）。"""
    for p in model.parameters():
        p.requires_grad = True
    if stage == "interior":
        for m in (model.encoder, model.up_head):
            for p in m.parameters():
                p.requires_grad = False
    elif stage == "surface":
        if model.vol_head is not None:
            for p in model.vol_head.parameters():
                p.requires_grad = False


@torch.no_grad()
def quick_eval(model, loader, weights, device) -> dict:
    """训练中的快速验证，用可微 loss 而非完整指标（完整指标走 evaluate.py）。"""
    model.eval()
    agg, n = {}, 0
    for b in loader:
        batch = {k: v.to(device) for k, v in b.items() if k != "name"}
        out = model(batch["input"],
                    batch["input_normals"] if model.use_normals else None)
        _, logs = total_loss(out, batch, weights)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(cfg: dict) -> dict:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    ds_tr = VolumetricPCDataset(cfg["data_root"], "train",
                                augment=cfg.get("augment", False))
    ds_va = VolumetricPCDataset(cfg["data_root"], "val")
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                       collate_fn=collate, num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                       collate_fn=collate, num_workers=0)

    model = PUVNet(**cfg["model"]).to(device)
    print(f"参数量 {model.n_params()/1e6:.3f}M | train={len(ds_tr)} val={len(ds_va)}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    stages = cfg.get("stages", [{"stage": "joint", "epochs": cfg["epochs"]}])

    for si, st in enumerate(stages):
        stage, n_ep = st["stage"], st["epochs"]
        set_stage(model, stage)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 1e-4))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_ep, 1))
        print(f"\n=== stage {si+1}/{len(stages)}: {stage} | "
              f"{n_ep} epochs | 可训练参数 {sum(p.numel() for p in params)/1e6:.3f}M ===")

        w = dict(cfg["loss_weights"])
        if stage == "surface":
            w["interior"] = 0.0
        elif stage == "interior":
            w["cd"] = 0.0
            w["repulsion"] = 0.0

        for ep in range(n_ep):
            t0 = time.time()
            agg, nb = {}, 0
            for b in dl_tr:
                batch = {k: v.to(device) for k, v in b.items() if k != "name"}
                out = model(batch["input"],
                            batch["input_normals"] if model.use_normals else None)
                loss, logs = total_loss(out, batch, w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                opt.step()
                for k, v in logs.items():
                    agg[k] = agg.get(k, 0.0) + v
                nb += 1
            sched.step()

            tr = {k: v / max(nb, 1) for k, v in agg.items()}
            va = quick_eval(model, dl_va, w, device)
            rec = {"stage": stage, "epoch": ep, "train": tr, "val": va,
                   "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
            history.append(rec)
            print(f"  ep{ep:03d} train_total={tr.get('total',0):.5f} "
                  f"cd={tr.get('cd',0):.5f} int={tr.get('interior',0):.5f} | "
                  f"val_total={va.get('total',0):.5f} "
                  f"val_cd={va.get('cd',0):.5f} "
                  f"({rec['sec']:.1f}s)")

    torch.save({"model": model.state_dict(), "cfg": cfg},
               out_dir / "checkpoint.pt")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n已保存到 {out_dir}")
    return {"history": history, "out_dir": str(out_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="覆盖配置，如 epochs=5 model.attention=False")
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
