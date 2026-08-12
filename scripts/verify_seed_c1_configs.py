# -*- coding: utf-8 -*-
"""启动前核验 SEED_C1 config 与源 config 的单一变量性。

只允许 seed / out_dir 两行不同，其余逐行逐字节必须完全一致。
任何第三处差异 -> 中止（宁可不跑，也不能让消融的单一变量失效）。

基线选择（重要）：必须用 `runs/ABL_C1_uniform/config.yaml`，即训练时实际落盘生效
的规范化版本（59 行），**不是** `configs/abl_C1_uniform.yaml`（85 行人工版带注释）。
make_seed_configs.py 也是从存档版派生的。用人工版作基线会因行数差异误判 FAIL。
2026-08-12 踩过：拿 85 行人工版比 59 行存档版，守门误报"行数不一致"。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "ABL_C1_uniform" / "config.yaml"
TARGETS = ["SEED_C1_s20260812.yaml", "SEED_C1_s20260813.yaml"]
ALLOWED_KEYS = ("seed:", "out_dir:")

if not SRC.exists():
    print(f"[FAIL] 基线存档 config 不存在: {SRC}")
    sys.exit(1)

src_lines = SRC.read_text(encoding="utf-8").splitlines()
fail = 0

for t in TARGETS:
    p = ROOT / "configs" / t
    lines = p.read_text(encoding="utf-8").splitlines()
    print("=" * 70)
    print(f"{t}  ({len(lines)} 行, 源 {len(src_lines)} 行)")
    if len(lines) != len(src_lines):
        print(f"  [FAIL] 行数不一致 {len(lines)} != {len(src_lines)}")
        fail += 1
        continue
    diffs = []
    for i, (a, b) in enumerate(zip(src_lines, lines), 1):
        if a != b:
            diffs.append((i, a.strip(), b.strip()))
    bad = [d for d in diffs
           if not any(d[1].startswith(k) or d[2].strip().startswith(k)
                      for k in ALLOWED_KEYS)]
    for i, a, b in diffs:
        tag = "OK  " if (i, a, b) not in [(x[0], x[1], x[2]) for x in bad] else "BAD "
        print(f"  [{tag}] L{i}: {a}  ->  {b}")
    # 关键参数回读复核（防跨 run 变量污染）
    for key in ("batch_size:", "epochs:", "w_uniform:", "w_adv:", "adaptive_adv:"):
        sv = [l.strip() for l in src_lines if l.strip().startswith(key)]
        tv = [l.strip() for l in lines if l.strip().startswith(key)]
        if sv != tv:
            print(f"  [FAIL] 关键参数不一致 {key}: {sv} vs {tv}")
            fail += 1
        else:
            print(f"  [keep] {key} {sv}")
    if bad:
        print(f"  [FAIL] 存在 {len(bad)} 处非允许差异")
        fail += 1
    elif len(diffs) == 0:
        print("  [FAIL] 零差异 —— seed 没被改，等于重复跑同一实验")
        fail += 1
    else:
        print(f"  [PASS] 仅 {len(diffs)} 处允许差异（seed/out_dir）")

print("=" * 70)
print("结果:", "PASS 可启动" if fail == 0 else f"FAIL({fail}) 禁止启动")
sys.exit(0 if fail == 0 else 1)
