# -*- coding: utf-8 -*-
"""
探测 PU-Transformer / PU-GAN 判别器的真实逐层张量形状与参数量,
输出 JSON, 供画架构图时使用。图上每个数字都必须来自本脚本实测, 不允许凭印象。
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from puvnet.models.pu_transformer import PUTransformer          # noqa: E402
from puvnet.models.pu_gan import PUGANDiscriminator              # noqa: E402

OUT = os.path.join(ROOT, "docs", "_arch_probe.json")

info = {}

# ---------------- 生成器 ----------------
G = PUTransformer(up_ratio=4)
G.eval()
info["generator"] = {
    "class": type(G).__name__,
    "up_ratio": G.up_ratio,
    "dims": list(G.dims),
    "tail_mode": getattr(G, "tail_mode", None),
    "n_params": G.count_parameters(),
}

trace = []
hooks = []


def mk_hook(name):
    def hook(mod, inp, out):
        def shp(t):
            if isinstance(t, torch.Tensor):
                return list(t.shape)
            if isinstance(t, (tuple, list)):
                return [shp(x) for x in t]
            return None
        trace.append({
            "name": name,
            "type": type(mod).__name__,
            "in": shp(inp[0]) if inp else None,
            "out": shp(out),
            "n_params": sum(p.numel() for p in mod.parameters(recurse=False)),
        })
    return hook


for n, m in G.named_modules():
    # 只挂顶层有意义的块, 避免噪声
    if n and n.count(".") <= 1:
        hooks.append(m.register_forward_hook(mk_hook(n)))

x = torch.randn(2, 256, 3)
with torch.no_grad():
    y = G(x)
for h in hooks:
    h.remove()

info["generator"]["input_shape"] = list(x.shape)
info["generator"]["output_shape"] = list(y.shape)
info["generator"]["trace"] = trace

# 各 Encoder 的子模块参数量(用于图上标注块规模)
enc_params = {}
for n, m in G.named_modules():
    if type(m).__name__ in ("TransformerEncoder", "PositionalFusion", "SCMSA"):
        enc_params.setdefault(type(m).__name__, []).append(
            {"name": n, "n_params": sum(p.numel() for p in m.parameters())})
info["generator"]["blocks"] = enc_params

# ---------------- 判别器 ----------------
D = PUGANDiscriminator()
D.eval()
dtrace = []
hooks = []
trace = dtrace
for n, m in D.named_modules():
    if n and n.count(".") <= 1:
        hooks.append(m.register_forward_hook(mk_hook(n)))
with torch.no_grad():
    dy = D(torch.randn(2, 1024, 3))
for h in hooks:
    h.remove()

info["discriminator"] = {
    "class": type(D).__name__,
    "n_params": sum(p.numel() for p in D.parameters()),
    "input_shape": [2, 1024, 3],
    "output_shape": list(dy.shape) if isinstance(dy, torch.Tensor) else None,
    "trace": dtrace,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("G params : %,d" % info["generator"]["n_params"] if False else
      "G params : {:,}".format(info["generator"]["n_params"]))
print("G in/out : %s -> %s" % (info["generator"]["input_shape"],
                               info["generator"]["output_shape"]))
print("D params : {:,}".format(info["discriminator"]["n_params"]))
print("D in/out : %s -> %s" % (info["discriminator"]["input_shape"],
                               info["discriminator"]["output_shape"]))
print()
print("--- G 顶层 trace ---")
for t in info["generator"]["trace"]:
    print("  %-22s %-22s %s -> %s  (p=%d)"
          % (t["name"], t["type"], t["in"], t["out"], t["n_params"]))
print()
print("--- D 顶层 trace ---")
for t in info["discriminator"]["trace"]:
    print("  %-22s %-22s %s -> %s  (p=%d)"
          % (t["name"], t["type"], t["in"], t["out"], t["n_params"]))
print()
print("[written] %s" % OUT)
