# -*- coding: utf-8 -*-
"""C2b 守卫改动的专项负例表。

背景：2026-08-15 把 C2b 的受检数量守卫从
    「正文含『改善占比』 且 seen_rows == 0」
收窄为
    「存在配对表（表头含『改善样本数』） 且 seen_rows == 0」，
以消除 ch4（方法论声明）与 ch7（结论引述）两处假红。

放松守卫后必须证明防线未失效。本表覆盖 5 个用例：

  G1 未篡改的 ch6            → 应 PASS（不得自我豁免）
  G2 表 6.5 改善数 158→168   → 应 FAIL（一致性校验仍在）
  G3 表 6.5 变差数 42→32     → 应 FAIL（防「短整数存在性」误放行）
  G4 表 6.5 占比 79.00→89.00 → 应 FAIL（占比与计数自洽校验仍在）
  G5 三行结构全部破坏         → 应 FAIL 且命中守卫（正则失效必须暴露）
  G6 仅 cv_nn 一行结构破坏    → 守卫按设计不触发（另两行仍命中），
                               但该行数字因此脱离核查 —— 本例记录这一
                               **已知覆盖缺口**，期望「rc=0 且守卫不报」。

G5 是本次改动的核心回归点：它模拟「row_re 失效」的场景，
若守卫被写成复用 row_re 判定「有表」，此例会静默放行（假绿）。

G6 的期望值经复核后确定为「不触发」：首版曾误期望其 FAIL，
实测 seen_rows=2（CD/HD 仍命中）→ 守卫按设计不响应。
按纪律「自检失败先分清脚本错还是用例期望错」，此处属**期望错**，
故修用例而非改守卫。但该缺口是真实的：单行失配不会被守卫发现，
所以 G6 保留在表内作为显式记录，避免下次误以为已覆盖。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, 'docs', 'chapters', 'ch6_experiments.md')
AUDIT = os.path.join(ROOT, 'scripts', 'audit_thesis_numbers.py')


def run_audit():
    """跑审计器，返回 (rc, 全部输出)。"""
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    p = subprocess.run(
        [sys.executable, AUDIT, 'ch6_experiments.md'],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace', env=env)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def tamper(src, repl):
    """把 src 替换为 repl；返回是否替换成功（防靶串失配导致假 PASS）。"""
    t = open(CH, encoding='utf-8').read()
    if src not in t:
        return False
    open(CH, 'w', encoding='utf-8', newline='').write(t.replace(src, repl, 1))
    return True


def main():
    orig = open(CH, encoding='utf-8').read()
    bak = os.path.join(tempfile.gettempdir(), '_ch6_negtest.bak')
    shutil.copy2(CH, bak)

    # 先从原文抽出表 6.5 的三行，避免靶串凭记忆硬编码。
    #
    # 关键：必须**先把范围缩到表 6.5 区块内**再抽 CD/HD 行。
    # 首版直接用 `^\|\s*CD\s*\|` 全文搜，抓到的是表 6.2 主对比表的
    # CD 行（$1.1457×10^{-3}$ …），于是表 6.5 的 CD/HD 从未被打断，
    # G5 因 seen_rows=2 而静默通过 —— 又一次「靶串失配造成假 PASS」。
    blk = re.search(r'\*\*表 6.5[^\n]*\*\*(.*?)(?=\n\*\*表 |\n## )',
                    orig, re.S)
    if not blk:
        print('[FATAL] 未能定位表 6.5 区块 → 本负例表无效')
        return 1
    block = blk.group(1)

    row = re.search(
        r'^\|[^|\n]*cv[^|\n]*\|\s*\**158\**\s*\|\s*\**42\**\s*\|'
        r'\s*\**79\.00\**\s*\\?%[^\n]*$',
        block, re.M)
    row_cd = re.search(r'^\|\s*CD\s*\|\s*\**131\**[^\n]*$', block, re.M)
    row_hd = re.search(r'^\|\s*HD\s*\|\s*\**106\**[^\n]*$', block, re.M)
    if not (row and row_cd and row_hd):
        print('[FATAL] 未能在表 6.5 区块内定位三行，'
              '靶串失配 → 本负例表无效，请先核对表格写法')
        return 1
    row_txt = row.group(0)
    cd_txt = row_cd.group(0)
    hd_txt = row_hd.group(0)
    # 三行靶串必须互不相同且都能在全文唯一定位，否则替换会打错行。
    for nm, s in (('cv', row_txt), ('CD', cd_txt), ('HD', hd_txt)):
        if orig.count(s) != 1:
            print(f'[FATAL] 靶串 {nm} 在全文出现 {orig.count(s)} 次，'
                  '非唯一 → 本负例表无效')
            return 1

    def break_all(t):
        """把三行的竖线结构全部打断，模拟 row_re 整体失效。"""
        for s in (row_txt, cd_txt, hd_txt):
            t = t.replace(s, s.replace('|', '/'), 1)
        return t

    cases = [
        ('G1', None, None, 0, None),
        ('G2', row_txt, row_txt.replace('158', '168', 1), 1, None),
        ('G3', row_txt, row_txt.replace('42', '32', 1), 1, None),
        ('G4', row_txt, row_txt.replace('79.00', '89.00', 1), 1, None),
        # G5：保留表头「改善样本数」，把三行数据行结构全部打断，
        #     模拟 row_re 完全失效。守卫必须因「有表却零命中」而报错。
        ('G5', 'ALL', None, 1, 'C2b'),
        # G6：只打断一行 → seen_rows=2，守卫按设计不触发（已知缺口）。
        ('G6', row_txt, row_txt.replace('|', '/'), 0, None),
    ]

    results = []
    try:
        for name, src, repl, want_rc, want_kw in cases:
            shutil.copy2(bak, CH)
            if src == 'ALL':
                t = open(CH, encoding='utf-8').read()
                open(CH, 'w', encoding='utf-8',
                     newline='').write(break_all(t))
            elif src is not None:
                if not tamper(src, repl):
                    results.append((name, 'FATAL 靶串失配', False))
                    continue
            rc, out = run_audit()
            ok = (rc != 0) if want_rc else (rc == 0)
            kw_ok = True
            if want_kw:
                kw_ok = want_kw in out
                ok = ok and kw_ok
            results.append((name, f'rc={rc} kw={"hit" if kw_ok else "MISS"}',
                            ok))
    finally:
        shutil.copy2(bak, CH)
        # 还原校验：必须逐字节等于原文，否则负例表污染了正稿。
        now = open(CH, encoding='utf-8').read()
        if now != orig:
            print('[FATAL] 还原失败，ch6 已被污染！备份在', bak)
            return 2

    print('=' * 74)
    print('C2b 守卫专项负例表（ch6_experiments.md）')
    print('=' * 74)
    npass = 0
    for name, detail, ok in results:
        print(f'  {name}  {detail:28s} {"PASS" if ok else "FAIL"}')
        npass += bool(ok)
    print()
    print(f'结果: {npass}/{len(results)} PASS')
    print('（已逐字节还原 ch6_experiments.md）')
    return 0 if npass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
