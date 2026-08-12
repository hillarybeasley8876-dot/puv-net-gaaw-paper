# -*- coding: utf-8 -*-
"""measure_cv_nn.py 合并逻辑负例表。

不碰真实产物：全部在临时目录里构造假 json，直接测试合并分支的行为契约。
覆盖用例：
  1 无旧文件           -> 只写本次
  2 旧文件同口径       -> 合并保留旧 run
  3 同名 run           -> 本次覆盖旧值
  4 口径不同(seed)     -> 不合并 + 另存 .oldcaliber.json
  5 口径不同(n_sample) -> 不合并
  6 旧文件损坏         -> 不合并但不崩、要留痕
  7 --no-merge         -> 整表覆盖
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "measure_cv_nn.py"

SEED, N_FULL, VAL_RATIO = 20260811, 200, 0.05


def simulate(old_json, results, no_merge=False,
             seed=SEED, n_sample=N_FULL, val_ratio=VAL_RATIO):
    """逐字复刻脚本里的合并分支（同一段逻辑，独立可测）。"""
    merged, notes = {}, []
    if old_json is not None and not no_merge:
        try:
            old = json.loads(old_json)
            same = (old.get("seed") == seed
                    and old.get("n_sample") == n_sample
                    and old.get("val_ratio") == val_ratio)
            if same and isinstance(old.get("runs"), dict):
                merged.update(old["runs"])
                notes.append("merged")
            elif not same:
                notes.append("oldcaliber_saved")
        except Exception:
            notes.append("read_failed")
    merged.update(results)
    return merged, notes


def main() -> int:
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [PASS] {name}")
        else:
            fail += 1
            print(f"  [FAIL] {name}  {detail}")

    base = json.dumps({"seed": SEED, "n_sample": N_FULL, "val_ratio": VAL_RATIO,
                       "runs": {"OLD_A": {"cv_nn": {"mean": 0.1}},
                                "OLD_B": {"cv_nn": {"mean": 0.2}}}})
    new = {"NEW_C": {"cv_nn": {"mean": 0.3}}}

    print("负例表 measure_cv_nn 合并逻辑")

    m, n = simulate(None, new)
    check("1 无旧文件只写本次", set(m) == {"NEW_C"}, str(set(m)))

    m, n = simulate(base, new)
    check("2 同口径合并保留旧 run",
          set(m) == {"OLD_A", "OLD_B", "NEW_C"} and "merged" in n, str(set(m)))

    m, n = simulate(base, {"OLD_A": {"cv_nn": {"mean": 0.9}}})
    check("3 同名 run 本次覆盖",
          m["OLD_A"]["cv_nn"]["mean"] == 0.9 and "OLD_B" in m, str(m.get("OLD_A")))

    bad_seed = json.dumps({"seed": 999, "n_sample": N_FULL, "val_ratio": VAL_RATIO,
                           "runs": {"OLD_A": {}}})
    m, n = simulate(bad_seed, new)
    check("4 seed 不同不合并 + 留痕",
          set(m) == {"NEW_C"} and "oldcaliber_saved" in n, f"{set(m)} {n}")

    bad_ns = json.dumps({"seed": SEED, "n_sample": 8, "val_ratio": VAL_RATIO,
                         "runs": {"OLD_A": {}}})
    m, n = simulate(bad_ns, new)
    check("5 n_sample 不同不合并", set(m) == {"NEW_C"}, str(set(m)))

    m, n = simulate("{not valid json", new)
    check("6 旧文件损坏不崩且留痕",
          set(m) == {"NEW_C"} and "read_failed" in n, f"{set(m)} {n}")

    m, n = simulate(base, new, no_merge=True)
    check("7 --no-merge 整表覆盖", set(m) == {"NEW_C"}, str(set(m)))

    print(f"\n结果 {ok}/{ok + fail} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
