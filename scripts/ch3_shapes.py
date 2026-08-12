# -*- coding: utf-8 -*-
"""第 3 章 3.4.2 证据：逐层张量形状与参数量分解（真实前向 hook 抓取）。

产物 docs/_ch3_shapes.json —— 表 3-1/表 3-2 与图 3-1 的唯一数字来源。
正文中任何张量形状/参数量必须能在此文件查到。
"""
import json, io, sys, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from puvnet.models.pu_transformer import PUTransformer
from puvnet.models.pu_gan import PUGANDiscriminator

B, N, R = 2, 256, 4
torch.manual_seed(20260811)

net = PUTransformer(up_ratio=R)
net.eval()

records = []

def mk_hook(name):
    def hook(mod, inp, out):
        def shp(t):
            return list(t.shape) if torch.is_tensor(t) else None
        ins = [shp(t) for t in inp if torch.is_tensor(t)]
        records.append({
            'name': name,
            'type': type(mod).__name__,
            'in_shapes': ins,
            'out_shape': shp(out),
            'n_params': sum(p.numel() for p in mod.parameters(recurse=True) if p.requires_grad),
        })
    return hook

handles = [net.head.register_forward_hook(mk_hook('head'))]
for i, enc in enumerate(net.encoders):
    handles.append(enc.register_forward_hook(mk_hook(f'encoders.{i}')))
    for sub in ('posfus', 'attn', 'mlp'):
        if hasattr(enc, sub):
            handles.append(getattr(enc, sub).register_forward_hook(mk_hook(f'encoders.{i}.{sub}')))
handles.append(net.tail.register_forward_hook(mk_hook('tail')))

xyz = torch.rand(B, N, 3)
with torch.no_grad():
    out = net(xyz)
for h in handles:
    h.remove()

# --- 参数量按模块分组 ---
groups = {}
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    parts = name.split('.')
    key = f'encoders.{parts[1]}' if parts[0] == 'encoders' else parts[0]
    groups[key] = groups.get(key, 0) + p.numel()
total = net.count_parameters()

# --- 判别器 ---
disc_info = None
try:
    d = PUGANDiscriminator()
    dn = sum(p.numel() for p in d.parameters() if p.requires_grad)
    with torch.no_grad():
        dout = d(torch.rand(B, N * R, 3))
    disc_info = {'class': type(d).__name__, 'n_params': dn,
                 'in_shape': [B, N * R, 3], 'out_shape': list(dout.shape)}
except Exception as e:  # noqa: BLE001
    disc_info = {'error': f'{type(e).__name__}: {e}'}

res = {
    'note': '真实 forward hook 抓取，B/N/r 为核验用取值，不代表训练 batch',
    'setting': {'B': B, 'N': N, 'up_ratio': R,
                'dims': list(net.dims), 'k': net.k, 'tail_mode': net.tail_mode},
    'generator': {
        'class': 'PUTransformer',
        'total_params': total,
        'paper_reported_params': 969900,
        'ratio_to_paper': total / 969900,
        'input_shape': [B, N, 3],
        'output_shape': list(out.shape),
        'group_params': groups,
        'layer_trace': records,
    },
    'discriminator': disc_info,
}
assert list(out.shape) == [B, N * R, 3], f'输出形状异常 {out.shape}'
assert total == 1_152_803, f'参数量与既有存档不一致：{total}'

dst = os.path.join(ROOT, 'docs', '_ch3_shapes.json')
json.dump(res, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('written', dst)
print(json.dumps({k: v for k, v in res.items() if k != 'generator'}, ensure_ascii=False, indent=1))
print('total_params', total, 'ratio', total / 969900)
for r in records:
    print(f"  {r['name']:<24} {r['type']:<20} in={r['in_shapes']} out={r['out_shape']} p={r['n_params']:,}")
