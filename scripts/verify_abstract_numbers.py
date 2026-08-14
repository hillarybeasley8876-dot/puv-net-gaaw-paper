# -*- coding: utf-8 -*-
"""独立验算摘要里 4 个"散文百分比"（审计器只判 warn，不验数值）。

摘要是全文数字最密集处，warn 意味着"无法自动回溯"而非"已核对"，
故这 4 个数必须逐个由存档独立算出，确认不是笔误。

键名均经实测确认（首版全部猜错，此处记录正确路径）：
  rho 轨迹      runs/rho_trace_B2.json -> rows[].rho_hat
                固定值 doc_calibration.rho（另有现成 b2_frac_below）
  cv_nn 相对量  docs/_thesis_results.json
                -> groups[<g>].runs[<run>].metrics.cv_nn.rel_pct
  每 epoch 耗时 docs/_train_cost.json -> groups[<g>].runs[<run>].sec_median
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 本脚本位于 scripts/ 下（由 scripts/_tmp/ 转正），ROOT 为 2 层 dirname。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    return json.load(io.open(os.path.join(ROOT, rel), encoding='utf-8'))


fails = []


def check(name, got, want, tol=0.02):
    ok = got is not None and abs(got - want) <= tol
    print(f'  {name:32s} 摘要={want:9.4f}  实算={got:9.4f}  '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        fails.append(name)


print('=== 1. 91.3%  固定权重偏离实际梯度环境的 epoch 占比 ===')
rho = load('runs/rho_trace_B2.json')
vals = [r['rho_hat'] for r in rho['rows']]
fixed = rho['doc_calibration']['rho']
above = sum(1 for v in vals if v > fixed)
print(f'  n={len(vals)}  rho_fixed={fixed}')
print(f'  rho_hat > rho_fixed: {above}  '
      f'(存档 b2_frac_below={rho["doc_calibration"]["b2_frac_below"]})')
print('  语义: rho = target_ratio / w_auto，二者成反比，'
      '故 rho_hat>rho_fixed <=> GAAW 权重 < 8.27')
check('91.3% (epoch 占比)', above / len(vals) * 100, 91.3, tol=0.05)

print()
print('=== 2. 5.44%  C1 相对 3090 基线 cv_nn 改善 ===')
th = load('docs/_thesis_results.json')
c1 = th['groups']['3090_uniform_structure']['runs']['ABL_C1_uniform']
m = c1['metrics']['cv_nn']
print(f'  baseline_mean={m["baseline_mean"]}  mean={m["mean"]}')
print(f'  存档 rel_pct={m["rel_pct"]}  verdict={m["verdict"]}  '
      f'n_se={m["n_se"]:.4f}')
check('5.44% (C1 cv_nn)', abs(m['rel_pct']), 5.44)

print()
print('=== 3/4. 训练时间增幅 ===')
tc = load('docs/_train_cost.json')
g3 = tc['groups']['3090']['runs']
g5 = tc['groups']['5090']['runs']
for base, run, want, label, runs in (
        ('B002_baseline150', 'ABL_C1_uniform', 13.04,
         '13.04% (C1 训练时间)', g3),
        ('ABL_B1_adv_fixed', 'ABL_B2_adv_adaptive', 70.58,
         '70.58% (B2 vs B1 时间)', g5)):
    a, b = runs[base]['sec_median'], runs[run]['sec_median']
    print(f'  {base} sec_median={a}   {run} sec_median={b}')
    check(label, (b - a) / a * 100, want)

print()
print('=== 5. 交叉核对：摘要与 ch7 的关键数字是否一致 ===')
mc = th['main_comparison']['metrics']
for k, want in (('cv_nn', -6.43), ('cd', -1.36), ('hd', -1.69)):
    print(f'  {k:6s} rel_pct={mc[k]["rel_pct"]:+.4f}  '
          f'n_se={mc[k]["n_se"]:.4f}  verdict={mc[k]["verdict"]}')
    check(f'{k} rel_pct', mc[k]['rel_pct'], want, tol=0.005)
for k, want in (('cv_nn', 8.48), ('cd', 0.32), ('hd', 0.35)):
    check(f'{k} n_se', mc[k]['n_se'], want, tol=0.005)

print()
print('失败项:', fails if fails else '无 —— 全部独立验算通过')
sys.exit(1 if fails else 0)
