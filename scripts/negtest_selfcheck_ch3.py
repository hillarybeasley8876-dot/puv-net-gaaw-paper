# -*- coding: utf-8 -*-
"""selfcheck_ch3.py 的负向测试: 人为注入 4 类违规, 确认自检能全部抓到。
不修改任何真实文稿: 在 docs/_negtest/ 下建临时副本树, 并以 monkeypatch 重定向 ROOT。
"""
import io
import os
import re
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
TMP = os.path.join(ROOT, "docs", "_negtest")

CASES = {
    "无效 cite key": ("{{cite:PUGCN}}", "{{cite:NOSUCHKEY_XYZ}}"),
    "硬编号": ("存档于 `docs/_ch3_diag.json`", "存档于 `docs/_ch3_diag.json`[42]"),
    "违规表述": ("本章据此固定了", "本方法显著提升了均匀性。本章据此固定了"),
    "数字不可溯源": ("比值 0.916", "比值 0.777"),
    "禁引文献": ("{{cite:PUNet}}", "{{cite:LOAM}}"),
}


def build(mutate_key=None):
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(os.path.join(TMP, "docs", "chapters"))
    os.makedirs(os.path.join(TMP, "scripts"))
    # 只复制自检需要读的文件。清单从 selfcheck_ch3.py 源码里抽 arch_files, 避免
    # 自检新增数据源后这里漏拷, 导致 baseline 假 FAIL。
    src = open(os.path.join(ROOT, "scripts", "selfcheck_ch3.py"), encoding="utf-8").read()
    m = re.search(r"arch_files = \[(.*?)\]", src, re.S)
    assert m, "未能从 selfcheck_ch3.py 抽出 arch_files"
    need = re.findall(r'"([^"]+\.json)"', m.group(1))
    assert need, "arch_files 抽取为空"
    for rel in ["docs/REFERENCES.json", "scripts/make_ch3_figures.py"] + need:
        dst = os.path.join(TMP, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    print("  (临时树复制数据源 %d 个)" % len(need)) if mutate_key is None else None
    shutil.copytree(os.path.join(ROOT, "docs", "figures_schematic"),
                    os.path.join(TMP, "docs", "figures_schematic"))
    with open(os.path.join(ROOT, "docs/chapters/ch3_baseline.md"), encoding="utf-8") as f:
        s = f.read()
    if mutate_key:
        old, new = CASES[mutate_key]
        assert old in s, "注入锚点不存在: %r" % old
        s = s.replace(old, new, 1)
    with open(os.path.join(TMP, "docs/chapters/ch3_baseline.md"), "w", encoding="utf-8") as f:
        f.write(s)


def run():
    """以子进程运行自检: 避免 reload 时顶层 stdout 包装与本进程冲突, 且更接近真实调用。
    通过环境变量 SELFCHECK_CH3_ROOT 重定向 ROOT。
    """
    import subprocess
    env = dict(os.environ)
    env["SELFCHECK_CH3_ROOT"] = TMP
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "selfcheck_ch3.py")],
                       capture_output=True, env=env)
    return r.returncode, r.stdout.decode("utf-8", "replace")


def main():
    bad = []
    build(None)
    rc, out = run()
    print("[baseline 未注入]      exit=%d  %s" % (rc, "OK" if rc == 0 else "!! 应为 PASS"))
    if rc != 0:
        bad.append("未注入时不应 FAIL")
        print(out[-1200:])
    for k in CASES:
        build(k)
        rc, out = run()
        tail = [l for l in out.split("\n") if l.strip().startswith("- ")]
        print("[注入 %-14s] exit=%d  %s" % (k, rc, "OK 已捕获" if rc != 0 else "!! 漏检"))
        for l in tail:
            print("      %s" % l.strip()[:110])
        if rc == 0:
            bad.append("漏检: %s" % k)
    shutil.rmtree(TMP)
    print("\n" + "=" * 60)
    if bad:
        print("负向测试 FAIL: %s" % bad)
        return 1
    print("负向测试 PASS: 5 类违规全部被捕获, 且未注入时不误报")
    return 0


if __name__ == "__main__":
    sys.exit(main())
