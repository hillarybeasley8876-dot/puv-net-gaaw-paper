# -*- coding: utf-8 -*-
"""
_adjudicate_n11_claim.py 的负例表 —— 防止裁定器「永远判绿」。

构造 5 种合成场景，验证裁定逻辑在每种下都给出正确 rc：
  N1 全部在验证区间内           -> rc=0（docx 为伪）
  N2 交集恰为 11 条              -> rc=1（docx 成立）
  N3 部分在外但不是 11           -> rc=3（人工复核）
  N4 per_sample 缺失             -> rc=2（无法裁定）
  N5 索引字段名不认识            -> rc=2（无法裁定）
"""
import json
import os
import subprocess
import sys
import tempfile

SCRIPT = 'scripts/_adjudicate_n11_claim.py'
LO, HI = 65550, 69000


def make_pair(idxs, per_sample=True, idx_key='idx'):
    """构造一对临时 (ch3_diag, cv_nn_measure) 产物。"""
    ch3 = {'source': {'seed': 20260811, 'n_sample': len(idxs),
                      'val_range': [LO, HI], 'best_epoch': 147}}
    run = {'run': 'FAKE', 'gpu': '3090', 'best_epoch': 1,
           'cv_nn': {'mean': 0.2, 'n': len(idxs)},
           'strata': {'q4_over_q1': 1.0}}
    if per_sample:
        run['per_sample'] = [{idx_key: i, 'cd': 0.001} for i in idxs]
    meas = {'seed': 20260811, 'n_sample': len(idxs), 'val_ratio': 0.05,
            'calib_vs_ch3_diag': True, 'runs': {'FAKE': run}}
    return ch3, meas


def run_case(name, idxs, expect_rc, per_sample=True, idx_key='idx'):
    ch3, meas = make_pair(idxs, per_sample, idx_key)
    tmp = tempfile.mkdtemp(prefix='negtest_')
    docs = os.path.join(tmp, 'docs')
    os.makedirs(docs, exist_ok=True)
    json.dump(ch3, open(os.path.join(docs, '_ch3_diag.json'), 'w',
                        encoding='utf-8'))
    json.dump(meas, open(os.path.join(docs, '_cv_nn_measure.json'), 'w',
                         encoding='utf-8'))
    src = open(SCRIPT, encoding='utf-8').read()
    shim = os.path.join(tmp, 'shim.py')
    open(shim, 'w', encoding='utf-8').write(src)

    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    p = subprocess.run([sys.executable, shim], cwd=tmp, env=env,
                       capture_output=True, text=True, encoding='utf-8')
    ok = (p.returncode == expect_rc)
    print('  %-4s %-34s expect_rc=%d got_rc=%s  %s'
          % (name, '', expect_rc, p.returncode, 'PASS' if ok else 'FAIL'))
    if not ok:
        print('       stdout tail:', (p.stdout or '')[-300:].replace('\n', ' | '))
        print('       stderr tail:', (p.stderr or '')[-300:].replace('\n', ' | '))
    return ok


def main():
    print('=' * 72)
    print('_adjudicate_n11_claim.py 负例表')
    print('=' * 72)
    results = []
    # N1 全部在区间内 -> docx 为伪
    results.append(run_case('N1', list(range(LO + 10, LO + 210)), 0))
    # N2 交集恰 11 条（11 在内 + 189 在外）-> docx 成立
    idxs = list(range(LO, LO + 11)) + list(range(100, 289))
    results.append(run_case('N2', idxs, 1))
    # N3 部分在外但不是 11 -> 人工复核
    idxs = list(range(LO, LO + 50)) + list(range(100, 250))
    results.append(run_case('N3', idxs, 3))
    # N4 per_sample 缺失 -> 无法裁定
    results.append(run_case('N4', list(range(LO, LO + 20)), 2,
                            per_sample=False))
    # N5 索引字段名不认识 -> 无法裁定
    results.append(run_case('N5', list(range(LO, LO + 20)), 2,
                            idx_key='weird_field_name'))

    print()
    n_pass = sum(results)
    print('结果: %d/%d PASS' % (n_pass, len(results)))
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
