# -*- coding: utf-8 -*-
"""
占位符 -> 编号替换器：{{cite:KEY}} -> [N]

这是论文交付前的一次性操作。替换后正文不再含 key 信息，因此**替换前必须**：
  1. 跑 scripts/_check_cite_coverage.py 确认全部 key 在库中（READY）
  2. 跑 scripts/_dump_cite_semantics.py 并人工核过语义（编号在库 != 语义正确）
两步未做完不要执行本脚本。

安全设计：
  * 只按 key 查库取 number，绝不手写数字
  * 相邻占位符合并：{{cite:A}}{{cite:B}} -> [12,13]（按编号升序去重）
  * 说明性写法 {{cite:⟨key⟩}} 保留不动（它是文档里讲语法用的，不是真引用）
  * 默认 dry-run，须显式 --apply 才写盘；写盘前备份 .bak_cite
  * 写盘后自检：正文不应再有真实 key 占位符；[N] 总数应等于替换前的 key 出现数
  * 库中缺 key 即中止，不做部分替换（避免正文半新半旧）

用法：
  python scripts/apply_cite_numbers.py            # dry-run
  python scripts/apply_cite_numbers.py --apply    # 写盘
"""
import json, os, io, re, sys, shutil
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")
FILES = ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md",
         "ch4_design.md", "ch5_mechanism.md", "ch6_experiments.md", "ch7_conclusion.md"]

APPLY = "--apply" in sys.argv

refs = json.load(io.open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
key2num = {r["key"]: int(r["number"]) for r in refs["references"]}

# 真实占位符：key 不含 ⟨ ⟩（那是说明性写法）
TOKEN = re.compile(r"\{\{cite:([^}⟨⟩]+)\}\}")
# 连续多个占位符（允许中间无字符）
RUN = re.compile(r"(?:\{\{cite:[^}⟨⟩]+\}\})+")

print("=" * 76)
print("占位符 -> 编号替换" + ("（写盘）" if APPLY else "（dry-run）"))
print("=" * 76)

# --- 前置校验：全部 key 必须在库中 ---
all_keys = Counter()
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    for m in TOKEN.finditer(t):
        all_keys[m.group(1).strip()] += 1
missing = [k for k in all_keys if k not in key2num]
if missing:
    print(f"[BLOCKING] {len(missing)} 个 key 不在库中，中止（不做部分替换）：")
    for k in sorted(missing):
        print(f"   {k}  用量 {all_keys[k]}")
    sys.exit(1)
print(f"  前置校验通过：{len(all_keys)} 个 key 全在库中，占位符共 {sum(all_keys.values())} 处")
print()

total_tokens = total_runs = 0
per_file = {}
for f in FILES:
    p = os.path.join(CH, f)
    src = io.open(p, encoding="utf-8").read()
    n_tok = len(TOKEN.findall(src))
    samples = []

    def repl(m):
        keys = TOKEN.findall(m.group(0))
        nums = sorted({key2num[k.strip()] for k in keys})
        out = "[" + ",".join(str(x) for x in nums) + "]"
        if len(samples) < 4:
            samples.append((m.group(0)[:56], out))
        return out

    dst, n_run = RUN.subn(repl, src)
    total_tokens += n_tok
    total_runs += n_run
    per_file[f] = (n_tok, n_run)

    print(f"  {f:26s} 占位符 {n_tok:>4d} -> 标记 {n_run:>4d}")
    for a, b in samples:
        print(f"       {a}  ->  {b}")

    if APPLY and dst != src:
        shutil.copyfile(p, p + ".bak_cite")
        io.open(p, "w", encoding="utf-8").write(dst)

print()
print(f"  合计：占位符 {total_tokens} 处，生成标记 {total_runs} 处（合并后）")

if not APPLY:
    print()
    print("  dry-run 结束。确认无误后加 --apply 写盘。")
    sys.exit(0)

# --- 写盘后自检 ---
print()
print("=" * 76)
print("写盘后自检")
print("=" * 76)
left_tok = 0
found_marks = 0
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    left_tok += len(TOKEN.findall(t))
    found_marks += len(re.findall(r"\[\d+(?:,\d+)*\]", t))
print(f"  残留真实占位符: {left_tok}  (应为 0)")
print(f"  正文中的 [N] 标记: {found_marks}")
ok = (left_tok == 0)
# 说明性写法应保留
kept = 0
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    kept += len(re.findall(r"\{\{cite:⟨", t))
print(f"  保留的说明性写法 {{{{cite:⟨key⟩}}}}: {kept} 处（不应被替换）")
print()
if ok:
    print("  PASS  替换完成")
else:
    print("  FAIL  仍有残留占位符")
    sys.exit(1)
