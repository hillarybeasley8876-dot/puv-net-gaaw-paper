# -*- coding: utf-8 -*-
"""GPU 加速比基准：本机 3090 vs 云端 5090，跑真实模型的真实训练步。

设计要点（为什么不测矩阵乘）：
  本模型（PU-Transformer + PU-GAN 判别器）含大量 kNN gather / index_select，
  可能偏访存瓶颈而非算力瓶颈。合成的 matmul 基准会严重高估 5090 的收益。
  因此必须用**真实 forward + loss + backward + optimizer.step** 的完整训练步。

可比性纪律（不满足则数字无意义）：
  - 同一份模型/loss 代码（本文件通过 import 复用，不复制实现）
  - 同 batch_size / 同 up_ratio / 同输入点数
  - 同 warmup 步数后再计时，避免 cudnn autotune 与 lazy init 污染
  - 同 SEED，输入数据由固定种子在设备上直接生成（避免磁盘 IO 差异干扰）
  - 记录 torch/CUDA/设备指纹，跨机器对比时必须一并报告

输出：JSON，含每步耗时序列、中位数、峰值显存、环境指纹。
用法：
    python scripts/bench_gpu.py --tag local_3090
    python scripts/bench_gpu.py --tag cloud_5090 --out /root/bench_5090.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from puvnet.models.pu_transformer import PUTransformer  # noqa: E402

SEED = 20260811


def env_fingerprint() -> dict:
    d = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update({
            "device_name": p.name,
            "capability": f"{p.major}.{p.minor}",
            "total_memory_GB": round(p.total_memory / 1024 ** 3, 2),
            "multi_processor_count": p.multi_processor_count,
            "arch_list": torch.cuda.get_arch_list(),
        })
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="标识本次运行的机器，如 local_3090")
    ap.add_argument("--batch", type=int, default=64, help="与 B-001 一致，默认 64")
    ap.add_argument("--n-in", type=int, default=256, help="输入点数（PU1K patch 为 256）")
    ap.add_argument("--up-ratio", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用")
        return 1

    env = env_fingerprint()
    print("=" * 74)
    print(f"GPU 基准  tag={args.tag}")
    print("=" * 74)
    for k, v in env.items():
        print(f"  {k:24s} = {v}")
    print()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device("cuda")

    model = PUTransformer(up_ratio=args.up_ratio).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量 = {n_params:,}")
    # 与之前实测一致性检查（PU-Transformer 生成器 1,152,803）
    print(f"  参数量一致性(=1,152,803): {n_params == 1_152_803}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 固定输入：在设备上直接生成，避免 IO / DataLoader 差异干扰纯计算对比
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = (torch.rand(args.batch, args.n_in, 3, generator=g) - 0.5).to(dev)
    gt = (torch.rand(args.batch, args.n_in * args.up_ratio, 3,
                     generator=g) - 0.5).to(dev)

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        # 用与训练同构的双向 CD（平方距离，mean 双向相加），不引入额外依赖
        d = torch.cdist(pred, gt) ** 2
        loss = d.min(dim=2)[0].mean() + d.min(dim=1)[0].mean()
        loss.backward()
        opt.step()

    print(f"\n  warmup {args.warmup} 步 ...")
    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    print(f"  计时 {args.steps} 步 ...")
    times = []
    for i in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{args.steps}  最近一步 {times[-1] * 1000:.1f} ms")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    med = statistics.median(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0

    print()
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  中位数单步    = {med * 1000:.2f} ms")
    print(f"  均值 ± 标准差 = {mean * 1000:.2f} ± {std * 1000:.2f} ms")
    print(f"  最快 / 最慢   = {min(times) * 1000:.2f} / {max(times) * 1000:.2f} ms")
    print(f"  峰值显存      = {peak:.3f} GB")
    # B-001 全量：69000 样本 / batch -> 每 epoch 步数
    steps_per_epoch = 69000 // args.batch
    epoch_s = med * steps_per_epoch
    print(f"  推算单 epoch  = {epoch_s:.1f} s = {epoch_s / 60:.2f} min "
          f"（{steps_per_epoch} 步 @ batch{args.batch}）")
    print(f"  推算 100 epoch= {epoch_s * 100 / 3600:.2f} h")

    result = {
        "tag": args.tag,
        "env": env,
        "config": {"batch": args.batch, "n_in": args.n_in,
                   "up_ratio": args.up_ratio, "warmup": args.warmup,
                   "steps": args.steps, "seed": SEED},
        "n_params": n_params,
        "times_s": times,
        "median_s": med, "mean_s": mean, "std_s": std,
        "min_s": min(times), "max_s": max(times),
        "peak_mem_GB": peak,
        "steps_per_epoch": steps_per_epoch,
        "epoch_seconds": epoch_s,
        "est_100epoch_hours": epoch_s * 100 / 3600,
    }
    out = Path(args.out) if args.out else ROOT / "runs" / f"bench_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n存档: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
