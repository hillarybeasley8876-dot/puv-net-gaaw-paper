# -*- coding: utf-8 -*-
"""实验存档备份 —— 打包 runs/ 的判据数据 + 文档 + 代码到带时间戳的 zip。

设计原则：
1. **判据优先**：history/summary_stats/selection/config/env/metrics/clouds_manifest
   + train_*.log 是论文数字的唯一来源，必须 100% 进包。
2. **大件可选**：clouds/*.npz 和 ckpt/*.pt 体积大、可由 ckpt 重算，默认不进包
   （--full 强制包含）。
3. **自校验**：打包后重新打开 zip 逐项核对，缺一项即 FAIL，绝不静默成功。
4. 只读操作，不删除、不移动任何原文件。

用法:
    python scripts/backup_runs.py                    # 判据备份（推荐，日常）
    python scripts/backup_runs.py --full             # 含 clouds/ 与 ckpt/
    python scripts/backup_runs.py --out D:/bak       # 指定输出目录（建议异盘）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 论文 run —— 这些的判据文件缺失即 FAIL
PAPER_RUNS = [
    "B001_reproduce",
    "B002_baseline150",
    "ABL_A1_cd_balance",
    "ABL_A2_cd_boost_bwd",
    "ABL_D1_scale_qk",
    "ABL_C1_uniform",
    "B002_baseline150_5090",
    "ABL_B1_adv_fixed",
    "ABL_B2_adv_adaptive",
]
# 判据文件（B001 历史豁免 summary_stats/selection，见 EXPERIMENT_LOG）
CRIT = [
    "history.json",
    "summary_stats.json",
    "selection.json",
    "config.yaml",
    "env.json",
    "metrics.json",
]
EXEMPT = {"B001_reproduce": {"summary_stats.json", "selection.json"}}

# 大件后缀（默认排除）
BIG_SUFFIX = {".npz", ".pt", ".pth", ".ckpt"}
# 代码 / 文档目录
CODE_DIRS = ["puvnet", "scripts", "configs", "docs"]
SKIP_DIR_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def sha256(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def collect(full: bool) -> tuple[list[Path], list[str]]:
    """返回 (待打包文件列表, 缺失判据告警)。"""
    files: list[Path] = []
    issues: list[str] = []

    runs = ROOT / "runs"
    if runs.is_dir():
        for dirpath, dirnames, filenames in os.walk(runs):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if not full and p.suffix.lower() in BIG_SUFFIX:
                    continue
                files.append(p)

    for d in CODE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
            for fn in filenames:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                files.append(Path(dirpath) / fn)

    for fn in ["README.md", "AGENTS.md", "requirements.txt", "pyproject.toml"]:
        p = ROOT / fn
        if p.is_file():
            files.append(p)

    # 判据完整性预检
    for run in PAPER_RUNS:
        rd = runs / run
        if not rd.is_dir():
            issues.append(f"{run}: 目录不存在")
            continue
        for c in CRIT:
            if c in EXEMPT.get(run, set()):
                continue
            if not (rd / c).is_file():
                issues.append(f"{run}/{c}: 缺失")

    return files, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="包含 clouds/*.npz 与 ckpt/*.pt（体积大）")
    ap.add_argument("--out", default=str(ROOT / "backups"),
                    help="输出目录，建议指向另一块物理盘")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "full" if args.full else "crit"
    zpath = outdir / f"puvnet_backup_{tag}_{stamp}.zip"

    print("=" * 74)
    print(f"[1] 收集文件（模式={tag}）")
    print("=" * 74)
    files, issues = collect(args.full)
    print(f"  待打包 {len(files)} 个文件")
    if issues:
        print("  ⚠️ 判据预检告警：")
        for i in issues:
            print("     -", i)
    else:
        print("  ✅ 论文 run 判据文件齐全")

    print()
    print("=" * 74)
    print(f"[2] 打包 -> {zpath}")
    print("=" * 74)
    manifest: dict[str, dict] = {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for i, p in enumerate(files, 1):
            arc = p.relative_to(ROOT).as_posix()
            z.write(p, arc)
            manifest[arc] = {"size": p.stat().st_size}
            if i % 200 == 0:
                print(f"  ... {i}/{len(files)}")
        meta = {
            "created": stamp,
            "mode": tag,
            "root": str(ROOT),
            "n_files": len(files),
            "paper_runs": PAPER_RUNS,
            "critical_files": CRIT,
            "exempt": {k: sorted(v) for k, v in EXEMPT.items()},
            "issues": issues,
            "note": ("判据备份不含 clouds/*.npz 与 ckpt/*.pt；"
                     "论文数字仅依赖 json/yaml/log，可完整重算统计。"
                     "需要复现推理或续训时用 --full。"),
        }
        z.writestr("BACKUP_META.json",
                   json.dumps(meta, ensure_ascii=False, indent=2))

    size_mb = zpath.stat().st_size / 1024 / 1024
    print(f"  ✅ 完成，{size_mb:.1f} MB")

    print()
    print("=" * 74)
    print("[3] 自校验（重新打开 zip 逐项核对）")
    print("=" * 74)
    bad: list[str] = []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for arc, info in manifest.items():
            if arc not in names:
                bad.append(f"{arc}: 不在包内")
                continue
            zi = z.getinfo(arc)
            if zi.file_size != info["size"]:
                bad.append(f"{arc}: 大小不符 "
                           f"{zi.file_size} != {info['size']}")
        # 判据文件必须在包内
        for run in PAPER_RUNS:
            for c in CRIT:
                if c in EXEMPT.get(run, set()):
                    continue
                arc = f"runs/{run}/{c}"
                if (ROOT / arc).is_file() and arc not in names:
                    bad.append(f"{arc}: 判据文件未进包")
        crc = z.testzip()
        if crc is not None:
            bad.append(f"CRC 校验失败于 {crc}")

    if bad:
        print("  ❌ 自校验 FAIL：")
        for b in bad[:30]:
            print("     -", b)
        return 1

    print(f"  ✅ 自校验 PASS（{len(manifest)} 项全部核对一致，CRC 无误）")
    print(f"  sha256 = {sha256(zpath)}")

    print()
    print("=" * 74)
    print("[4] 结果")
    print("=" * 74)
    print(f"  备份包: {zpath}")
    print(f"  体积  : {size_mb:.1f} MB")
    if issues:
        print("  ⚠️ 存在判据告警（见上），但已备份现有内容")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
