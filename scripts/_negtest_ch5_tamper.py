# -*- coding: utf-8 -*-
"""ch5 篡改负例表 —— 逐个篡改正文里最重要的数字，确认审计器每个都能检出。

为什么需要这张表：
  `issues=0` 只说明「没抓到问题」，不说明「抓得住问题」。
  只跑「未篡改应绿」永远无法暴露通道失效（正则不匹配 / 存在性判断代替
  一致性判断 / 容差过松）。实测已抓出 4 处假绿，其中 3 处正是靠本表发现。

用法：
  python scripts/_negtest_ch5_tamper.py
  退出码 0 = 全部篡改均被检出；非 0 = 有漏网，审计器需修。

原文件绝不改动：每例都在临时副本上篡改，跑完即删。
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-08-15 重定向：论文重构为七章后 ch5_experiments.md → ch6_experiments.md。
# 重定向前已用 scripts/_tmp/check_tamper_targets.py 逐条核验 11 个靶串在新
# 文件中的命中数，据此修正了 3 条（P2/P8/P11），详见各用例注释。
CH = os.path.join(ROOT, 'docs', 'chapters', 'ch6_experiments.md')
AUDIT = os.path.join(ROOT, 'scripts', 'audit_thesis_numbers.py')

# (编号, 说明, 原串, 篡改串)
# 覆盖三类靶点：① 主结果均值 ② 相对变化百分比 ③ 裸整数计数
CASES = [
    ('P1', 'cv_nn 配对改善数 158→168（裸整数计数）',
     '| $\\mathrm{cv}_{\\mathrm{nn}}$ | **158** | 42 |',
     '| $\\mathrm{cv}_{\\mathrm{nn}}$ | **168** | 42 |'),
    ('P2', 'B2 主结果 cv_nn 均值末位篡改',
     # 重构后 `0.239709` 在表 6-2 与表 6-3 各出现一次（实测 count=2）。
     # 加长为带加粗的主对比表写法以保证唯一命中。
     '**0.239709**', '**0.238709**'),
    ('P3', 'cv_nn 相对改善 -6.43%→-8.43%',
     '$\\mathbf{-6.43\\%}$', '$\\mathbf{-8.43\\%}$'),
    ('P4', 'n_se 8.48→12.48（显著性倍数夸大）',
     '$\\mathbf{8.48}$', '$\\mathbf{12.48}$'),
    ('P5', 'CD 的 SE 倍数 0.32→2.32（把 REJECT 说成过门槛）',
     '| 0.32 | REJECT_NULL |', '| 2.32 | REJECT_NULL |'),
    ('P6', 'C1 消融 cv_nn 相对变化 -5.44%→-9.44%',
     '$\\mathbf{-5.44\\%}$', '$\\mathbf{-9.44\\%}$'),
    ('P7', 'A2 劣化幅度 148.42%→48.42%（缩小负面结果）',
     '$+148.42\\%$', '$+48.42\\%$'),
    ('P8', '图 6-1 不利事实 0.256054→0.216054（掩盖变差样本）',
     # 该数字是「B2 在样本 67133 上反而更差」这一不利事实的承载值。
     # 重构中它曾一度从正文消失（只剩 EXPERIMENT_LOG 有记录），
     # 2026-08-15 随 §6.5.3 定性小节补回表 6-6，现唯一命中。
     '**0.256054**', '**0.216054**'),
    ('P9', 'cv_nn 变差数 42→32（裸整数计数，且和不再等于 200）',
     '| **158** | 42 |', '| **158** | 32 |'),
    ('P10', 'HD 配对改善数 106→116（裸整数计数）',
     '| HD | 106 | 94 |', '| HD | 116 | 94 |'),
    ('P11', 'HD 均值 0.004997338→0.004097338',
     # 重构后该值在表 6-2 与表 6-3 各出现一次（实测 count=2）。
     # 用表 6-3「HD 行」的行首上下文加长，保证只命中主对比表。
     '| HD | $5.0835×10^{-3}$ | $4.9973×10^{-3}$ |',
     '| HD | $5.0835×10^{-3}$ | $4.0973×10^{-3}$ |'),
]


def run_audit(tmp_chapters_name):
    """在 docs/chapters 下以裸文件名调用审计器（main() 的参数约定）。"""
    r = subprocess.run(
        [sys.executable, AUDIT, tmp_chapters_name],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
    )
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main():
    src = io.open(CH, encoding='utf-8').read()
    tmp_name = '_tamper_probe.md'
    tmp_path = os.path.join(ROOT, 'docs', 'chapters', tmp_name)

    # 基线：未篡改副本必须 PASS，否则后续「检出」无法归因
    io.open(tmp_path, 'w', encoding='utf-8').write(src)
    rc0, out0 = run_audit(tmp_name)
    base_ok = 'issues=0' in out0
    print(f'[BASE] 未篡改副本 issues=0 -> {base_ok}')
    if not base_ok:
        print(out0[-2000:])
        os.remove(tmp_path)
        return 2

    n_pass = 0
    fails = []
    for cid, desc, old, new in CASES:
        if old not in src:
            print(f'[{cid}] SKIP-ERR 靶串不存在，负例本身失效: {old[:40]!r}')
            fails.append((cid, desc, 'target-missing'))
            continue
        if src.count(old) < 1:
            fails.append((cid, desc, 'target-missing'))
            continue
        tampered = src.replace(old, new, 1)
        io.open(tmp_path, 'w', encoding='utf-8').write(tampered)
        rc, out = run_audit(tmp_name)
        m = re.search(r'issues=(\d+)', out)
        n_iss = int(m.group(1)) if m else -1
        # 判「检出」的标准：产生 issue，或有告警**点名了篡改后的那个值**。
        # 不能只看 warns 总数：本章本就有 1 条合法派生量 warn，
        # 单看计数无法区分「抓到篡改」与「原有告警」。
        probe = new.strip('$').strip('\\%').lstrip('+')
        named = [ln.strip() for ln in out.splitlines()
                 if ('[C2' in ln or '[C3' in ln) and probe[:6] in ln]
        caught = n_iss > 0 or bool(named)
        if caught:
            n_pass += 1
            lines = [ln.strip() for ln in out.splitlines()
                     if ln.strip().startswith(('[C2', '[C3'))]
            first = (named[0] if named else (lines[0] if lines else ''))[:110]
            kind = 'ISSUE' if n_iss > 0 else 'WARN '
            print(f'[{cid}] CAUGHT({kind}) issues={n_iss}  {desc}')
            if first:
                print(f'       └─ {first}')
        else:
            print(f'[{cid}] *** MISSED ***  {desc}')
            fails.append((cid, desc, f'issues={n_iss}'))

    os.remove(tmp_path)
    print(f'\n篡改表结果: {n_pass}/{len(CASES)} 被检出')
    if fails:
        print('漏网：')
        for cid, desc, why in fails:
            print(f'  - {cid} {desc}  ({why})')
        return 1
    print('全部篡改均被检出 -> 审计器防线有效')
    return 0


if __name__ == '__main__':
    sys.exit(main())
