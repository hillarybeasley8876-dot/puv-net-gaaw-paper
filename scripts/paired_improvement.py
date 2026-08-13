"""逐样本配对改善占比（B2 GAAW vs B1 固定权重）。

目的：给 §5.3.3 提供一个**不需要显著性门槛**的描述性统计。
「200 个样本里有 N 个变好」是事实陈述，不是假设检验，故无需事后补门槛。

红线遵守：
- 只用 5090 组同机器的 B1 / B2，不跨机器；
- 严格按 idx 配对，配对失败即报错退出，不做"按位置对齐"的默认假设；
- 不设任何通过/不通过门槛，只报占比与分布；
- 结果落盘 JSON，供正文与审计器回查。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURE = os.path.join(ROOT, 'docs', '_cv_nn_measure.json')
OUT = os.path.join(ROOT, 'docs', '_paired_improvement_B2_vs_B1.json')

B1 = 'ABL_B1_adv_fixed'
B2 = 'ABL_B2_adv_adaptive'
# 指标 -> per_sample 字段名（越小越好）
METRICS = {'cv_nn': 'nn_pred_cv', 'cd': 'cd', 'hd': 'hd'}


def main():
    d = json.load(open(MEASURE, encoding='utf-8'))
    runs = d['runs']
    for k in (B1, B2):
        if k not in runs:
            raise SystemExit('缺少 run: %s' % k)

    # 同机器核验：配对比较必须同 host
    g1, g2 = runs[B1].get('gpu'), runs[B2].get('gpu')
    if g1 != g2:
        raise SystemExit('跨机器配对被拒绝: %s=%s vs %s=%s' % (B1, g1, B2, g2))

    m1 = {p['idx']: p for p in runs[B1]['per_sample']}
    m2 = {p['idx']: p for p in runs[B2]['per_sample']}
    common = sorted(set(m1) & set(m2))
    if len(common) != len(m1) or len(common) != len(m2):
        raise SystemExit('样本索引不完全一致: B1=%d B2=%d 交集=%d'
                         % (len(m1), len(m2), len(common)))
    n = len(common)

    out = {
        'purpose': '逐样本配对改善占比（描述性统计，无显著性门槛）',
        'comparison': '%s (B2, GAAW) vs %s (B1, fixed w_adv)' % (B2, B1),
        'host': g1,
        'n_paired': n,
        'pairing': '按 per_sample.idx 严格配对；索引集合完全一致已核验',
        'convention': '三项指标均越小越好；improved = B2 < B1',
        'no_threshold_note': ('本量为事实陈述（N/200 个样本方向变好），'
                             '不设接收门槛，不作显著性裁定。'),
        'source': 'docs/_cv_nn_measure.json',
        'metrics': {},
    }

    for name, field in METRICS.items():
        imp = tie = wor = 0
        deltas = []
        for i in common:
            a, b = m1[i][field], m2[i][field]
            dlt = b - a          # B2 - B1，负=改善
            deltas.append(dlt)
            if b < a:
                imp += 1
            elif b > a:
                wor += 1
            else:
                tie += 1
        deltas.sort()
        mid = n // 2
        median = (deltas[mid] if n % 2 else
                  0.5 * (deltas[mid - 1] + deltas[mid]))
        out['metrics'][name] = {
            'field': field,
            'n_improved': imp,
            'n_worse': wor,
            'n_tie': tie,
            'pct_improved': 100.0 * imp / n,
            'median_delta': median,
            'delta_desc': 'B2 - B1（负值=B2 更优）',
        }

    json.dump(out, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print('=' * 70)
    print('逐样本配对改善占比  B2 (GAAW) vs B1 (fixed)   host=%s  n=%d' % (g1, n))
    print('=' * 70)
    for name, r in out['metrics'].items():
        print('%-7s 改善 %3d / %d = %6.2f%%   变差 %3d   持平 %d   '
              'median Δ = %+.6e'
              % (name, r['n_improved'], n, r['pct_improved'],
                 r['n_worse'], r['n_tie'], r['median_delta']))
    print()
    print('落盘 ->', os.path.relpath(OUT, ROOT).replace(os.sep, '/'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
