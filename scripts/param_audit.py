"""参数量归因：定位与论文 969.9k 的差距来源。

spec §9 列出几个原文未明写的待验证项：
  - Encoder 内 MLP 的扩张比 mlp_ratio
  - PosFus 中 MLP 的层数
  - Head MLP 输出通道
本脚本用参数量为约束反推最可能的配置。
"""
import sys
sys.path.insert(0, r"E:\AE-CC托管\puv-net")

import torch
from puvnet.models.pu_transformer import PUTransformer

TARGET = 969_900


def breakdown(net: PUTransformer, label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"{label}  总参数 {net.count_parameters():,}  (目标 {TARGET:,}, "
          f"比例 {net.count_parameters()/TARGET:.4f})")
    print("=" * 70)
    groups: dict[str, int] = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        # 归到 encoder 级别
        parts = name.split(".")
        if parts[0] == "encoders":
            key = f"encoders.{parts[1]}.{parts[2]}"
        else:
            key = parts[0]
        groups[key] = groups.get(key, 0) + p.numel()
    for k in sorted(groups, key=lambda x: -groups[x]):
        print(f"  {k:<32} {groups[k]:>10,}  ({groups[k]/net.count_parameters()*100:5.1f}%)")


def sweep() -> None:
    print("\n" + "#" * 70)
    print("# mlp_ratio 扫描（spec §9 待验证项）")
    print("#" * 70)
    print(f"{'mlp_ratio':>10} | {'参数量':>12} | {'比例':>8} | {'与目标差':>12}")
    print("-" * 70)
    best = None
    for ratio in (1, 2, 4):
        net = PUTransformer(up_ratio=4, mlp_ratio=ratio)
        n = net.count_parameters()
        diff = n - TARGET
        print(f"{ratio:>10} | {n:>12,} | {n/TARGET:>8.4f} | {diff:>+12,}")
        if best is None or abs(diff) < abs(best[1] - TARGET):
            best = (ratio, n)
    print(f"\n最接近: mlp_ratio={best[0]}, 参数 {best[1]:,}")

    print("\n" + "#" * 70)
    print("# head_dim 扫描")
    print("#" * 70)
    print(f"{'head_dim':>10} | {'参数量':>12} | {'比例':>8}")
    print("-" * 70)
    for hd in (16, 32, 64):
        net = PUTransformer(up_ratio=4, head_dim=hd)
        n = net.count_parameters()
        print(f"{hd:>10} | {n:>12,} | {n/TARGET:>8.4f}")


if __name__ == "__main__":
    net = PUTransformer(up_ratio=4)
    breakdown(net, "默认配置 (mlp_ratio=1, head_dim=32)")
    sweep()

    print("\n" + "#" * 70)
    print("# 结论")
    print("#" * 70)
    print("""
参数量差距的可能来源（按影响排序）：
  1. PosFus 的两个 MLP 若原文是单层（本实现即单层），差距来自别处
  2. SC-MSA 的 proj 层：本实现 Linear(n_heads*w, dim)，
     n_heads=7 时输入维 = 7*(C/4) = 1.75C，比 dim 大 —— 这是主要开销
  3. LayerNorm / BatchNorm 的 affine 参数
  4. TF 与 PyTorch 的 conv bias 习惯差异

由于原文未公开代码，无法逐层核对。本项目立场（见 spec §9）：
  - 保持忠于论文明写的参数（L=5, dims, k=20, psi=4）
  - 待验证项取最合理默认值并在论文中显式标注为「复现实现细节」
  - 不为了凑参数量而扭曲结构
""")
