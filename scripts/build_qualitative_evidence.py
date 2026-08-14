# -*- coding: utf-8 -*-
"""生成 §6.5.3 定性小节所需的汇总层证据，落盘 docs/_qualitative_panel_67133.json。

为什么必须落盘：正文引用的「67133 劣化量 3.2627e-3」「最严重变差样本 66013
的 3.1829e-2」是从 per_sample 现算的，而审计器的权威池**有意排除
per_sample**（防逐样本噪声污染回溯池）。若不落成汇总层，这两个数在正文里
就是悬空数字，审计器判 C2 无法回溯 —— 实测确认它确实报了这 2 个 issue。

本脚本把这些量算好并写入汇总 json，使其进入权威池，同时留下可复核的口径。
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 本脚本位于 scripts/ 下，故 ROOT 为 2 层 dirname。
# （scripts/_tmp/ 下的脚本才需要 3 层——两者不可混用。）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEAS = os.path.join(ROOT, 'docs', '_cv_nn_measure.json')
OUT = os.path.join(ROOT, 'docs', '_qualitative_panel_67133.json')
FIGMETA = os.path.join(ROOT, 'paper_assets_TRIAL', 'figures_ch5',
                       'F5_1_qualitative_pointclouds.meta.json')

B1, B2 = 'ABL_B1_adv_fixed', 'ABL_B2_adv_adaptive'
TARGET = 67133

meas = json.load(io.open(MEAS, encoding='utf-8'))


def ps(run):
    out = {}
    for r in meas['runs'][run]['per_sample']:
        out[r['idx']] = r
    return out


a, b = ps(B1), ps(B2)
common = sorted(set(a) & set(b))
assert len(common) == 200, f'配对样本数异常: {len(common)}'

# 目标样本三项指标
tgt = {}
for k in ('nn_pred_cv', 'cd', 'hd'):
    v1, v2 = a[TARGET][k], b[TARGET][k]
    tgt[k] = {
        'b1': v1, 'b2': v2, 'delta': v2 - v1,
        'rel_pct': (v2 - v1) / v1 * 100.0,
        'direction': 'B2_worse' if v2 > v1 else 'B2_better',
    }

# cv_nn 变差样本排位
worse = sorted(((b[i]['nn_pred_cv'] - a[i]['nn_pred_cv'], i)
                for i in common
                if b[i]['nn_pred_cv'] > a[i]['nn_pred_cv']), reverse=True)
rank = [i for _, i in worse].index(TARGET) + 1
tgt_delta = dict((i, d) for d, i in worse)[TARGET]
worst_d, worst_i = worse[0]

# CD 改善降序（复核选样准则）
cd_desc = sorted(((a[i]['cd'] - b[i]['cd'], i) for i in common), reverse=True)
picked = [i for _, i in cd_desc[:3]]

figmeta = json.load(io.open(FIGMETA, encoding='utf-8'))
meta_picked = figmeta['picked_idx']

doc = {
    'schema': 'qualitative_panel_evidence_v1',
    'purpose': ('为 ch6 §6.5.3 的定性小节提供汇总层证据；'
                '正文引用的每个数字均须能在本文件中回溯'),
    'source': 'docs/_cv_nn_measure.json (per_sample)',
    'note': ('per_sample 被审计器权威池有意排除，故此处将正文所需量'
             '预先聚合为汇总层字段'),
    'comparison': f'{B2} vs {B1}',
    'n_paired': len(common),
    'target_idx': TARGET,
    'target_metrics': tgt,
    'cv_nn_worse_stats': {
        'n_worse': len(worse),
        'target_rank_desc': rank,
        'rank_note': '1 = 劣化最严重',
        'target_delta': tgt_delta,
        'worst_idx': worst_i,
        'worst_delta': worst_d,
        'worst_over_target': worst_d / tgt_delta,
    },
    'pick_criterion': {
        'rule': 'argsort desc of cd(B1) - cd(B2), take top 3',
        'computed_top3': picked,
        'figure_meta_picked_idx': meta_picked,
        'reproduced': picked == list(meta_picked),
        'cd_gain_top3': [d for d, _ in cd_desc[:3]],
    },
}

io.open(OUT, 'w', encoding='utf-8', newline='').write(
    json.dumps(doc, ensure_ascii=False, indent=1))

print('已写入', os.path.relpath(OUT, ROOT))
print()
print(f'样本 {TARGET}:')
for k, v in tgt.items():
    print(f'  {k:11s} B1={v["b1"]:.8g} B2={v["b2"]:.8g} '
          f'rel={v["rel_pct"]:+.4f}% {v["direction"]}')
print()
print(f'cv_nn 变差样本 {len(worse)} 个；{TARGET} 排第 {rank}；'
      f'delta={tgt_delta:.6e}')
print(f'最严重 idx={worst_i} delta={worst_d:.6e} '
      f'倍数={worst_d / tgt_delta:.4f}')
print()
print('选样准则复现:', picked, '== meta', list(meta_picked),
      '->', doc['pick_criterion']['reproduced'])
