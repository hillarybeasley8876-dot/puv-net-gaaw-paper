# -*- coding: utf-8 -*-
"""
audit_thesis_numbers.py 的负例表 —— 防止审计器「永远判绿」。

构造 6 种合成章节文本，验证审计器在每种下都给出正确 rc：
  N1 干净稿（数字全部来自权威池，文件真实存在）      -> rc=0
  N2 引用不存在的 json（复刻 Codex 事故）             -> rc=1
  N3 编造的实测数字（不在权威池内）                   -> rc=1
  N4 数字被篡改一位小数（0.239709 -> 0.339709）       -> rc=1
  N5 同表行跨机器并列（红线）                          -> rc=0 但必须出 warn
  N6 空稿                                              -> rc=0
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, 'scripts', 'audit_thesis_numbers.py')
RESULTS = os.path.join(ROOT, 'docs', '_thesis_results.json')


def build_sandbox():
    """搭一个最小工程骨架：docs/_thesis_results.json + docs/chapters/。"""
    tmp = tempfile.mkdtemp(prefix='auditneg_')
    os.makedirs(os.path.join(tmp, 'docs', 'chapters'), exist_ok=True)
    os.makedirs(os.path.join(tmp, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(tmp, 'scripts'), exist_ok=True)
    shutil.copy(RESULTS, os.path.join(tmp, 'docs', '_thesis_results.json'))
    # N7 需要科学计数法真值落在池内，故一并带上 docs 白名单诊断存档。
    diag = os.path.join(ROOT, 'docs', '_ch3_diag.json')
    if os.path.exists(diag):
        shutil.copy(diag, os.path.join(tmp, 'docs', '_ch3_diag.json'))
    shutil.copy(SCRIPT, os.path.join(tmp, 'scripts',
                                     'audit_thesis_numbers.py'))
    return tmp


def run_case(name, body, expect_rc, expect_warn_kw=None):
    tmp = build_sandbox()
    ch = os.path.join(tmp, 'docs', 'chapters', 'ch9_test.md')
    open(ch, 'w', encoding='utf-8').write(body)

    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    p = subprocess.run(
        [sys.executable, os.path.join(tmp, 'scripts',
                                      'audit_thesis_numbers.py')],
        cwd=tmp, env=env, capture_output=True, text=True, encoding='utf-8')
    rc_ok = (p.returncode == expect_rc)
    warn_ok = True
    if expect_warn_kw:
        warn_ok = expect_warn_kw in (p.stdout or '')
    ok = rc_ok and warn_ok
    print('  %-4s expect_rc=%d got=%s  warn_kw=%-6s  %s'
          % (name, expect_rc, p.returncode,
             expect_warn_kw or '-', 'PASS' if ok else 'FAIL'))
    if not ok:
        print('       stdout:', (p.stdout or '')[-700:].replace('\n', ' | '))
        print('       stderr:', (p.stderr or '')[-300:].replace('\n', ' | '))
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    print('=' * 74)
    print('audit_thesis_numbers.py 负例表')
    print('=' * 74)

    # 从权威汇总取真实数字，保证 N1 必定可回溯
    R = json.load(open(RESULTS, encoding='utf-8'))
    b2 = R['groups']['5090_adversarial']['runs']['ABL_B2_adv_adaptive']
    real_cv = b2['metrics']['cv_nn']['mean']          # 0.239709...
    base = R['groups']['5090_adversarial']['runs']['B002_baseline150_5090']
    real_base_cv = base['metrics']['cv_nn']['mean']   # 0.223049...

    res = []

    # N1 干净稿
    res.append(run_case(
        'N1',
        '# 测试章\n\n'
        'B2 的 cv_nn 实测为 %.6f，本组基线为 %.6f。\n'
        '数据来源 `docs/_thesis_results.json`。\n' % (real_cv, real_base_cv),
        0))

    # N2 引用不存在的证据文件（复刻 Codex 事故）
    res.append(run_case(
        'N2',
        '# 测试章\n\n'
        'B2 的 cv_nn 实测为 %.6f。\n'
        '拆分审计见 `docs/_split_audit_and_holdout_subset.json`。\n' % real_cv,
        1))

    # N3 编造的实测数字
    res.append(run_case(
        'N3',
        '# 测试章\n\n'
        'B2 的 cv_nn 实测为 0.187654，明显优于基线。\n',
        1))

    # N4 篡改一位小数
    res.append(run_case(
        'N4',
        '# 测试章\n\nB2 的 cv_nn 实测为 0.339709。\n',
        1))

    # N5 跨机器并列（红线）—— 数字真实故 rc=0，但必须出 C3 warn
    res.append(run_case(
        'N5',
        '# 测试章\n\n'
        '| run | cv_nn |\n|---|---|\n'
        '| ABL_C1_uniform | %.6f |\n'
        '| ABL_B2_adv_adaptive | %.6f |\n'
        '\n本表混排 3090 与 5090 两组。\n'
        % (R['groups']['3090_uniform_structure']['runs']
           ['ABL_C1_uniform']['metrics']['cv_nn']['mean'], real_cv),
        0, expect_warn_kw='[C3]'))

    # N6 空稿
    res.append(run_case('N6', '# 空章\n\n暂无内容。\n', 0))

    # ---- 科学计数法量级通道（2026-08-14 补） ----
    # 背景：审计器早期只抓尾数 5.469 去查池，丢掉 \times 10^{-4}。
    # 后果双向：真值被报假红；更危险的是尾数碰巧撞上池中同名数值 -> 假绿。
    # N7 真值带指数后缀        -> rc=0（不得假红）
    # N8 尾数保留但量级篡改    -> rc=1（不得假绿）
    sci = 0.0005468675447627903     # _ch3_diag sparsity_strata Q3 后向分量
    res.append(run_case(
        'N7',
        '# 测试章\n\n'
        'Q3 区间的后向分量均值为 $%.3f \\times 10^{-4}$。\n' % (sci * 1e4),
        0))
    res.append(run_case(
        'N8',
        '# 测试章\n\n'
        'Q3 区间的后向分量均值为 $%.3f \\times 10^{-2}$。\n' % (sci * 1e4),
        1))

    print()
    n = sum(res)
    print('结果: %d/%d PASS' % (n, len(res)))
    return 0 if n == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())
