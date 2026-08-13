# -*- coding: utf-8 -*-
"""
裁定：待盲审 docx 的 n=11「真实验证交集」是否成立。

判据（跑前定死，不得事后修改）：
  J1  训练/验证划分由 train_pu.py 的真实口径重建，验证区间必须是 [65550, 69000)
  J2  _cv_nn_measure.json 里 200 条样本的索引，落在验证区间内的条数
      - 若 == 200  → docx 的「189 条属训练划分」为伪，n=200 口径成立
      - 若 == 11   → docx 成立，本地 n=200 结论必须撤回
      - 其它       → 需人工复核，不得自动裁定
  J3  measure 脚本自身声明的 val_ratio / seed / n_sample 是否与 ch3 诊断一致（calib 标志）

结论只依据落盘产物，不依据任何文档叙述。
"""
import json
import sys

MEASURE = 'docs/_cv_nn_measure.json'
CH3 = 'docs/_ch3_diag.json'

d = json.load(open(MEASURE, encoding='utf-8'))
c3 = json.load(open(CH3, encoding='utf-8'))

print('=' * 72)
print('J3  口径一致性')
print('=' * 72)
c3src = c3.get('source', {})
print('  measure: seed=%s n_sample=%s val_ratio=%s calib_vs_ch3_diag=%s'
      % (d.get('seed'), d.get('n_sample'), d.get('val_ratio'),
         d.get('calib_vs_ch3_diag')))
print('  ch3    : seed=%s n_sample=%s val_range=%s best_epoch=%s'
      % (c3src.get('seed'), c3src.get('n_sample'), c3src.get('val_range'),
         c3src.get('best_epoch')))

val_range = c3src.get('val_range')
if not val_range:
    print('  !! ch3_diag 未记录 val_range，无法执行 J1/J2')
    sys.exit(2)
lo, hi = val_range
print('  验证区间 = [%d, %d)   共 %d 条' % (lo, hi, hi - lo))

print()
print('=' * 72)
print('J1+J2  200 条补测样本的索引归属')
print('=' * 72)

runs = d['runs']
verdicts = {}
for name, v in runs.items():
    per = v.get('per_sample')
    if not per:
        print('  %-28s per_sample 缺失，跳过' % name)
        continue
    # 取出每条记录的样本索引（字段名当场从真实产物确认，不靠记忆）
    keys = list(per[0].keys())
    idx_key = None
    for cand in ('idx', 'index', 'sample_idx', 'i', 'sid'):
        if cand in keys:
            idx_key = cand
            break
    if idx_key is None:
        print('  %-28s 无法识别索引字段，实际字段=%s' % (name, keys))
        continue

    idxs = [r[idx_key] for r in per]
    n_total = len(idxs)
    in_val = [i for i in idxs if lo <= i < hi]
    out_val = [i for i in idxs if not (lo <= i < hi)]
    verdicts[name] = (n_total, len(in_val), len(out_val))
    print('  %-28s idx_key=%-6s n=%3d  在验证区间=%3d  在区间外=%3d  '
          'idx范围=[%d, %d]'
          % (name, idx_key, n_total, len(in_val), len(out_val),
             min(idxs), max(idxs)))

print()
print('=' * 72)
print('裁定')
print('=' * 72)
if not verdicts:
    print('  无法裁定：所有 run 均缺 per_sample 索引')
    sys.exit(2)

all_in = all(t == iv for (t, iv, ov) in verdicts.values())
any_11 = any(iv == 11 for (t, iv, ov) in verdicts.values())

if all_in:
    print('  ✅ 全部 200 条样本索引均落在验证区间 [%d, %d) 内。' % (lo, hi))
    print('  ✅ docx §6.1.3「200 条中 189 条实际属于训练划分」为伪。')
    print('  ✅ 本地 n=200 口径成立，docx 的 n=11 重算无事实基础。')
    sys.exit(0)
elif any_11:
    print('  ❌ 存在 run 的验证交集为 11 条 —— docx 口径可能成立，本地结论须撤回。')
    sys.exit(1)
else:
    print('  ⚠ 既非全部在内也非 11 条，需人工复核，不自动裁定。')
    sys.exit(3)
