# -*- coding: utf-8 -*-
"""第 3 章 3.5.2 / 3.5.3 证据：基线瓶颈的实证诊断。

只用 B-002 baseline150 的 best.pt 做前向推理，不训练、不改权重。
在 CPU 上跑（GPU 让给正在排队的消融组），256->1024 的 patch 规模足够。

诊断三件事：
  D-a 3.5.2 预测点云 vs 真实点云的最近邻间距分布差异（均匀性瓶颈的直接证据）
  D-b 3.5.3 局部误差随「输入局部密度」的分布（稀疏区是否更差）
  D-c 双向 CD 分量在验证集上的不对称性（与训练期监控口径互相印证）

产物 docs/_ch3_diag.json。正文任何数字必须能在此文件查到。
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from puvnet.data.pu_dataset import PUTrainDataset
from puvnet.models.pu_transformer import PUTransformer

RUN = ROOT / 'runs' / 'B002_baseline150'
SMOKE = '--smoke' in sys.argv
N_SAMPLE = 8 if SMOKE else 200   # 诊断样本数，跑前定死
SEED = 20260811
VAL_RATIO = 0.05         # 与 B-002 config 一致，取同一尾部切片作为验证集


# ---------------------------------------------------------------- 工具
def nn_dist(p: np.ndarray) -> np.ndarray:
    """每点到最近邻的距离（欧氏，排除自身）。"""
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def cross_nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 中每点到 b 的最近距离（欧氏）。"""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return d.min(axis=1)


def desc(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {
        'mean': float(v.mean()), 'std': float(v.std()),
        'cv': float(v.std() / v.mean()) if v.mean() > 0 else 0.0,
        'p05': float(np.percentile(v, 5)), 'p50': float(np.percentile(v, 50)),
        'p95': float(np.percentile(v, 95)), 'max': float(v.max()),
    }


def spearman(a, b) -> float:
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    ra, rb = rank(list(a)), rank(list(b))
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


# ---------------------------------------------------------------- 载入
ckpt = torch.load(RUN / 'ckpt' / 'best.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt.get('gen_state', ckpt.get('model_state', ckpt)))
# baseline 的 ckpt 存的是 PUTransGAN 包装后的 state_dict，键带 generator. 前缀
if any(k.startswith('generator.') for k in sd):
    sd = {k[len('generator.'):]: v for k, v in sd.items() if k.startswith('generator.')}
if not isinstance(sd, dict) or not any(k.endswith('weight') for k in sd):
    raise SystemExit(f'无法识别 checkpoint 结构，顶层键={list(ckpt)[:10]}')

net = PUTransformer(up_ratio=4)
missing, unexpected = net.load_state_dict(sd, strict=False)
if missing or unexpected:
    raise SystemExit(f'权重不匹配 missing={missing[:5]} unexpected={unexpected[:5]}')
net.eval()

# 验证集：与训练脚本同口径取尾部 val_ratio，且不做增广
ds = PUTrainDataset(source='pu1k', up_ratio=4, noise_beta=0.0, augment=False)
n_total = len(ds)
n_val = int(n_total * VAL_RATIO)
val_start = n_total - n_val
rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(np.arange(val_start, n_total), size=min(N_SAMPLE, n_val),
                          replace=False))

print(f'dataset={n_total} val=[{val_start},{n_total}) 抽样 {len(pick)} 个')

# ---------------------------------------------------------------- 逐样本诊断
per = []
with torch.no_grad():
    for c, i in enumerate(pick):
        inp, gt = ds[int(i)]
        inp_t = inp.unsqueeze(0) if torch.is_tensor(inp) else torch.tensor(inp)[None]
        gt_np = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
        pred = net(inp_t.float())[0].numpy()
        inp_np = inp_t[0].numpy()

        nn_pred, nn_gt = nn_dist(pred), nn_dist(gt_np)
        fwd = cross_nn(pred, gt_np)          # pred -> gt，精度项
        bwd = cross_nn(gt_np, pred)          # gt -> pred，覆盖项
        f2, b2 = float((fwd ** 2).mean()), float((bwd ** 2).mean())

        # 局部密度：gt 点到输入点云的最近距离，越大 = 输入越稀疏
        d_to_in = cross_nn(gt_np, inp_np)

        per.append({
            'idx': int(i),
            'nn_pred_mean': float(nn_pred.mean()), 'nn_pred_cv': float(nn_pred.std() / nn_pred.mean()),
            'nn_gt_mean': float(nn_gt.mean()), 'nn_gt_cv': float(nn_gt.std() / nn_gt.mean()),
            'cd_fwd': f2, 'cd_bwd': b2, 'cd': f2 + b2,
            'bwd_share': b2 / (f2 + b2),
            'hd': float(max(fwd.max(), bwd.max()) ** 2),
            # 分位分层：把 gt 点按到输入的距离分 4 档，看 bwd 误差随稀疏度的变化
            'bwd_by_sparsity': [
                float((bwd[(d_to_in >= lo) & (d_to_in < hi)] ** 2).mean())
                for lo, hi in zip(
                    np.percentile(d_to_in, [0, 25, 50, 75]),
                    list(np.percentile(d_to_in, [25, 50, 75])) + [np.inf])
            ],
            'sparsity_quartile_edges': [float(x) for x in np.percentile(d_to_in, [0, 25, 50, 75, 100])],
        })
        if (c + 1) % 50 == 0:
            print(f'  {c + 1}/{len(pick)}')

# ---------------------------------------------------------------- 汇总
def col(k):
    return [r[k] for r in per]

res = {
    'note': 'B-002 baseline150 best.pt 前向推理（CPU，无训练）；样本数与随机种子跑前定死',
    'source': {'ckpt': 'runs/B002_baseline150/ckpt/best.pt',
               'best_epoch': ckpt.get('epoch'),
               'n_sample': len(pick), 'seed': SEED,
               'val_range': [val_start, n_total], 'augment': False},
    # --- 3.5.2 间距分布 ---
    'spacing': {
        'nn_pred_cv': desc(np.array(col('nn_pred_cv'))),
        'nn_gt_cv': desc(np.array(col('nn_gt_cv'))),
        'cv_ratio_mean': float(mean(col('nn_pred_cv')) / mean(col('nn_gt_cv'))),
        'n_pred_cv_gt_gt_cv': sum(1 for r in per if r['nn_pred_cv'] > r['nn_gt_cv']),
        'nn_pred_mean': desc(np.array(col('nn_pred_mean'))),
        'nn_gt_mean': desc(np.array(col('nn_gt_mean'))),
    },
    # --- 3.5.1 双向不对称 ---
    'cd_split': {
        'bwd_share': desc(np.array(col('bwd_share'))),
        'n_bwd_share_gt_half': sum(1 for r in per if r['bwd_share'] > 0.5),
        'cd_fwd_mean': float(mean(col('cd_fwd'))),
        'cd_bwd_mean': float(mean(col('cd_bwd'))),
        'cd_mean': float(mean(col('cd'))),
    },
    # --- 3.5.3 稀疏区误差分层 ---
    'sparsity_strata': {
        'q_labels': ['Q1(最密)', 'Q2', 'Q3', 'Q4(最疏)'],
        'bwd_mean_by_quartile': [
            float(mean([r['bwd_by_sparsity'][q] for r in per])) for q in range(4)
        ],
    },
    # --- 相关性：均匀性差的样本是否 CD 也差 ---
    'corr': {
        'cv_pred_vs_cd': spearman(col('nn_pred_cv'), col('cd')),
        'cv_pred_vs_bwd_share': spearman(col('nn_pred_cv'), col('bwd_share')),
        'cd_vs_hd': spearman(col('cd'), col('hd')),
    },
    'per_sample': per,
}
q = res['sparsity_strata']['bwd_mean_by_quartile']
res['sparsity_strata']['q4_over_q1'] = q[3] / q[1 - 1] if q[0] > 0 else None

dst = ROOT / 'docs' / ('_ch3_diag_SMOKE.json' if SMOKE else '_ch3_diag.json')
dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', dst)
print(json.dumps({k: v for k, v in res.items() if k != 'per_sample'},
                 ensure_ascii=False, indent=1))
