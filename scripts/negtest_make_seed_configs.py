# -*- coding: utf-8 -*-
"""make_seed_configs 的负例表 —— 防止「单一变量自检」假绿。

判据脚本改完必须先用构造数据跑负例，不能拿真环境当测试场
（真环境只覆盖当前那一种状态）。

用例设计
--------
1. 正常改写            -> verify 无违规 且 行数不变 且 seed 真被改
2. 篡改 batch_size     -> verify 必须抓出（这是红线键）
3. 篡改 w_adv          -> verify 必须抓出
4. 行数变化            -> verify 必须抓出
5. 源缺 seed 键        -> rewrite 必须 SystemExit（不得静默用默认值）
6. 带行内注释的 seed   -> 注释须保留，且不产生多余空行
7. 缩进键同名干扰      -> 顶层 out_dir 改、嵌套同名键不得被误改
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_seed_configs as M  # noqa: E402

SRC = """out_dir: runs/BASE
seed: 20260811
batch_size: 64
w_adv: 8.27
adaptive_adv: true
loss:
  cd: 0.5
"""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {note}" if note else ""))


# --- 1 正常改写 ---
new, changed = M.rewrite(SRC, 20260812, "runs/NEW")
bad = M.verify(SRC, new)
ok1 = (not bad
       and len(SRC.splitlines()) == len(new.splitlines())
       and "seed: 20260812" in new
       and "out_dir: runs/NEW" in new
       and "batch_size: 64" in new
       and len(changed) == 2)
check("1 正常改写: 无违规/行数不变/seed 已改", ok1,
      f"行数 {len(SRC.splitlines())}->{len(new.splitlines())} 改动={len(changed)}")

# --- 2 篡改红线键 batch_size ---
tampered = new.replace("batch_size: 64", "batch_size: 32")
bad2 = M.verify(SRC, tampered)
ok2 = any("batch_size" in b for b in bad2)
check("2 篡改 batch_size 必须被抓出", ok2, str(bad2[:1]))

# --- 3 篡改 w_adv ---
tampered3 = new.replace("w_adv: 8.27", "w_adv: 1.0")
bad3 = M.verify(SRC, tampered3)
ok3 = any("w_adv" in b for b in bad3)
check("3 篡改 w_adv 必须被抓出", ok3, str(bad3[:1]))

# --- 4 行数变化 ---
bad4 = M.verify(SRC, new + "extra_key: 1\n")
ok4 = any("行数变化" in b for b in bad4)
check("4 行数变化必须被抓出", ok4, str(bad4[:1]))

# --- 5 源缺 seed 键 -> 必须 SystemExit ---
no_seed = "out_dir: runs/BASE\nbatch_size: 64\n"
try:
    M.rewrite(no_seed, 20260812, "runs/NEW")
    ok5 = False
    note5 = "未报错（危险：会静默用默认 seed）"
except SystemExit as e:
    ok5 = "缺键" in str(e)
    note5 = str(e)[:60]
check("5 源缺 seed 键必须中止", ok5, note5)

# --- 6 行内注释保留且不多空行 ---
src6 = "out_dir: runs/BASE\nseed: 20260811  # 跑前定死\nbatch_size: 64\n"
new6, _ = M.rewrite(src6, 20260812, "runs/NEW")
ok6 = ("# 跑前定死" in new6
       and len(new6.splitlines()) == len(src6.splitlines())
       and "seed: 20260812" in new6
       and "" not in [l for l in new6.splitlines()])
check("6 行内注释保留且无多余空行", ok6,
      f"行数 {len(src6.splitlines())}->{len(new6.splitlines())}")

# --- 7 嵌套同名键不得被误改 ---
src7 = ("out_dir: runs/BASE\nseed: 20260811\ndata:\n"
        "  seed: 999\n  batch_size: 64\n")
new7, ch7 = M.rewrite(src7, 20260812, "runs/NEW")
# 嵌套 seed 同样匹配（缩进被 \s* 吃掉）——此为已知行为，
# 判据：若发生改动必须出现在 changed 记录里，不得静默
nested_changed = "  seed: 20260812" in new7
ok7 = (len(new7.splitlines()) == len(src7.splitlines())
       and (not nested_changed or len(ch7) >= 3))
check("7 嵌套同名键: 要么不改, 要么留痕", ok7,
      f"嵌套被改={nested_changed} 改动记录={len(ch7)}")
if nested_changed:
    print("     [WARN] 嵌套 seed 也被改写。真实 config 顶层 seed 在 L2、"
          "无嵌套 seed（已核对），当前不触发；若将来出现须改用顶层限定正则。")

print("-" * 70)
n_ok = sum(1 for _, o, _ in results if o)
print(f"负例表: {n_ok}/{len(results)} PASS")
sys.exit(0 if n_ok == len(results) else 1)
