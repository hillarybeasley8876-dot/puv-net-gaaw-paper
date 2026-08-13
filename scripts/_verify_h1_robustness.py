# -*- coding: utf-8 -*-
"""
H1 稳健性复核：3.92e6 倍极差是「阶段性演化」还是「少数异常点撑起来的」？

盲审最可能的攻击：极差 = max/min，只要有一个异常 epoch 就能刷出天文数字。
因此必须证明：即使剔除极值，主体趋势仍跨越多个数量级。

判据（跑前定死）：
  R1  剔除首尾各 10% 分位后的极差（p10..p90）仍 >= 100 倍  -> 极差非异常点驱动
  R2  前段 [0,25) 与 平台区 [75,150) 的中位数之比 >= 100 倍 -> 存在阶段性演化
  R3  单调性：分 6 段各取中位数，检查是否呈总体上升（Spearman rho > 0.7）
输出供正文引用的稳健统计量。
"""
import json
import statistics as st

d = json.load(open('runs/rho_trace_B2.json', encoding='utf-8'))
rows = d['rows']
rho = [r['rho_hat'] for r in rows]
n = len(rho)
srt = sorted(rho)


def q(p):
    """分位数（线性插值）。"""
    i = p * (len(srt) - 1)
    lo, hi = int(i), min(int(i) + 1, len(srt) - 1)
    return srt[lo] + (srt[hi] - srt[lo]) * (i - lo)


print('=' * 74)
print('H1 稳健性复核  (n=%d epoch)' % n)
print('=' * 74)
print('原始极差  max/min = %.6g / %.6g = %.6g 倍'
      % (max(rho), min(rho), max(rho) / min(rho)))

p10, p90 = q(0.10), q(0.90)
p05, p95 = q(0.05), q(0.95)
r1a = p90 / p10
r1b = p95 / p05
print()
print('R1  抗异常点极差')
print('    p10=%.6g  p90=%.6g   p90/p10 = %.6g 倍   %s'
      % (p10, p90, r1a, 'PASS' if r1a >= 100 else 'FAIL'))
print('    p05=%.6g  p95=%.6g   p95/p05 = %.6g 倍   %s'
      % (p05, p95, r1b, 'PASS' if r1b >= 100 else 'FAIL'))

early = [r['rho_hat'] for r in rows if r['epoch'] < 25]
plateau = [r['rho_hat'] for r in rows if r['epoch'] >= 75]
me, mp = st.median(early), st.median(plateau)
r2 = mp / me
print()
print('R2  阶段性演化（早期 vs 平台区中位数）')
print('    early[0,25)  n=%d  median=%.6g' % (len(early), me))
print('    plateau[75,) n=%d  median=%.6g' % (len(plateau), mp))
print('    比值 = %.6g 倍   %s' % (r2, 'PASS' if r2 >= 100 else 'FAIL'))

print()
print('R3  分段中位数单调性（6 段，每段 25 epoch）')
seg_med = []
for s in range(0, 150, 25):
    v = [r['rho_hat'] for r in rows if s <= r['epoch'] < s + 25]
    seg_med.append(st.median(v))
    print('    epoch[%3d,%3d)  n=%2d  median=%.6g' % (s, s + 25, len(v), seg_med[-1]))

# Spearman（对秩做 Pearson）
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for pos, i in enumerate(order):
        rk[i] = float(pos)
    return rk

rx, ry = rank(list(range(len(seg_med)))), rank(seg_med)
mx, my = st.mean(rx), st.mean(ry)
num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
sp = num / den if den else 0.0
print('    Spearman rho(段序, 段中位数) = %.4f   %s'
      % (sp, 'PASS' if sp > 0.7 else 'FAIL'))

print()
print('=' * 74)
print('可供正文引用的稳健统计量')
print('=' * 74)
print('  抗异常点极差 (p90/p10) : %.4g 倍' % r1a)
print('  早期→平台区中位数比值   : %.4g 倍' % r2)
print('  平台区中位数            : %.6g' % mp)
print('  段中位数单调性 Spearman : %.4f' % sp)

allpass = (r1a >= 100) and (r2 >= 100) and (sp > 0.7)
print()
print('裁定: H1 %s' % ('稳健成立（极差非异常点驱动，存在阶段性上升演化）'
                       if allpass else '不稳健，需弱化表述'))
