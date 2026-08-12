# -*- coding: utf-8 -*-
"""生成多 seed 复现 config —— 兑现 3.5.5 预注册的 2SE 门槛。

背景
----
scripts/audit_promises.py 债务 [2]：全部 9 个 run 均为 seed=20260811 单次运行，
只能算「平台区跨 epoch σ」，无法算「跨 seed SE」，而 3.5.5 预注册门槛是 2SE。
用户已批准补 baseline + B2 两组 × 2 个新 seed（共 4 个 run）。

方法
----
**不手敲 config** —— 直接读取已完成 run 的 config.yaml 原文，
逐行只替换 `seed:` 与 `out_dir:` 两个键，其余字节原样保留。
这样可保证除 seed 外严格单一变量（避免漏抄某项造成不可比）。

红线
----
* 只改 seed / out_dir，其他任何键（尤其 batch_size=64）不得改动
* 新 run 全部在 5090 上跑 —— 与 5090 基线同机，数字方可并表
* loader 参数由 deploy 阶段注入，不写入本机 yaml

用法
    python scripts/make_seed_configs.py            # 生成并自检
    python scripts/make_seed_configs.py --dry-run
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

NEW_SEEDS = [20260812, 20260813]
# (源 run, 新 run 名模板) —— 源必须是已完成的 5090 run，保证同机可比
SOURCES = [
    ("B002_baseline150_5090", "SEED_baseline_s{seed}"),
    ("ABL_B2_adv_adaptive", "SEED_B2_s{seed}"),
    # C1 是 cv_nn（预注册主指标）唯一达标组，须有跨 seed SE。
    # 源取 3090 的 C1 config，但新 run 在 5090 上跑 —— 因此其对照基准
    # 只能是 5090 的 baseline seed 组，不得与 3090 的 C1 并表。
    ("ABL_C1_uniform", "SEED_C1_s{seed}"),
]
ALLOWED_CHANGES = {"seed", "out_dir"}


def rewrite(src_text: str, seed: int, out_dir: str) -> tuple[str, list[str]]:
    """逐行改写，只允许动 seed / out_dir。返回 (新文本, 改动记录)。"""
    changed: list[str] = []
    lines = src_text.splitlines(keepends=True)
    out: list[str] = []
    seen = set()
    for ln in lines:
        # 先剥离行尾换行再匹配：否则 tail 的 \s* 会吞掉 \n，
        # 拼回时又补一个，导致每替换一行多出一个空行（已实测踩中）。
        nl = "\n" if ln.endswith("\n") else ""
        body = ln[:-1] if nl else ln
        m = re.match(r"^(\s*)(seed|out_dir)\s*:\s*(.*?)([ \t]*(?:#.*)?)$", body)
        if m:
            indent, key, old, tail = m.groups()
            new = str(seed) if key == "seed" else out_dir
            out.append(f"{indent}{key}: {new}{tail}{nl}")
            changed.append(f"{key}: {old.strip()} -> {new}")
            seen.add(key)
        else:
            out.append(ln)
    missing = ALLOWED_CHANGES - seen
    if missing:
        sys.exit(f"[FAIL] 源 config 缺键 {missing}，拒绝生成（防止静默用默认值）")
    return "".join(out), changed


def verify(src_text: str, new_text: str) -> list[str]:
    """自检：除 seed/out_dir 外任何行不得变化。返回违规行。"""
    a = src_text.splitlines()
    b = new_text.splitlines()
    if len(a) != len(b):
        return [f"行数变化 {len(a)} -> {len(b)}"]
    bad = []
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x == y:
            continue
        k = re.match(r"^\s*([A-Za-z_][\w]*)\s*:", x)
        key = k.group(1) if k else "<?>"
        if key not in ALLOWED_CHANGES:
            bad.append(f"L{i} 非允许键 [{key}] 发生变化: {x.strip()!r} -> {y.strip()!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("多 seed config 生成（除 seed 外严格单一变量）")
    print("=" * 74)

    plan = []
    for src_run, tmpl in SOURCES:
        sp = ROOT / "runs" / src_run / "config.yaml"
        if not sp.exists():
            sys.exit(f"[FAIL] 缺源 config: {sp}")
        src_text = sp.read_text(encoding="utf-8")
        for seed in NEW_SEEDS:
            run = tmpl.format(seed=seed)
            new_text, changed = rewrite(src_text, seed, f"runs/{run}")
            bad = verify(src_text, new_text)
            status = "PASS" if not bad else "FAIL"
            print(f"\n  {run}")
            print(f"    源: runs/{src_run}/config.yaml")
            for c in changed:
                print(f"    改: {c}")
            print(f"    单一变量自检: {status}")
            for b in bad:
                print(f"      {b}")
            if bad:
                sys.exit("[FAIL] 自检不通过，已中止（不产出可疑 config）")
            plan.append((run, new_text))

    if args.dry_run:
        print("\n[DRY-RUN] 未写文件")
        return 0

    CFG.mkdir(exist_ok=True)
    for run, text in plan:
        p = CFG / f"{run}.yaml"
        p.write_text(text, encoding="utf-8")
        print(f"\n[OK] 写 {p.relative_to(ROOT)}")

    # 回读复核：写出的文件 seed 必须真的是新值
    print("\n" + "-" * 74)
    print("回读复核")
    for run, _ in plan:
        p = CFG / f"{run}.yaml"
        txt = p.read_text(encoding="utf-8")
        sd = re.search(r"^\s*seed\s*:\s*(\S+)", txt, re.M)
        od = re.search(r"^\s*out_dir\s*:\s*(\S+)", txt, re.M)
        bs = re.search(r"^\s*batch_size\s*:\s*(\S+)", txt, re.M)
        print(f"  {run:26s} seed={sd.group(1) if sd else '?':10s} "
              f"batch_size={bs.group(1) if bs else '?':4s} "
              f"out_dir={od.group(1) if od else '?'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
