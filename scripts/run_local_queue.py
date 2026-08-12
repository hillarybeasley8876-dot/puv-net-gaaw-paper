# -*- coding: utf-8 -*-
"""本机 3090 串行队列：等 B-002 跑完 -> 依次跑 5 组不带对抗的消融。

分工（用户 2026-08-11 拍板）：
    本机 3090 : A1 / A2 / D1 / C1 / AC   （不带判别器，1.0~2.3x 倍率）
    云端 5090 : baseline150 / B1 / B2 / BD（带判别器，4.2~7.7x 倍率）

为什么不用 & 并行：
    单卡并行会互相抢 SM，两个 run 都变慢且**耗时数据失去可比性**
    （已在 GPU 加速比基准踩过：被占用时测得虚高 3.30 倍）。
    串行虽然 wall 长，但每组的 s/ep 都干净可用。

为什么等 B-002：
    B-002 是全部消融组的对照。若与消融组并行跑，两者都被干扰，
    "平台区均值差异"里会混入 GPU 争抢造成的收敛差异。

崩溃恢复：本脚本每组结束都把状态写进 queue_state.json，
    重启后自动跳过已完成的组（按 out_dir 是否有满 150 epoch 的 metrics.json 判定）。

用法：
    python scripts/run_local_queue.py            # 前台跑（会阻塞数十小时）
    python scripts/run_local_queue.py --status   # 只看状态
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "local_queue_state.json"

# 顺序有讲究：先跑最快且最可能出结论的（A1/A2 是一对，必须相邻跑完才能判读），
# D1 单项快，C1/AC 较慢放最后。
ABL_QUEUE = [
    ("A1_cd_balance", "abl_A1_cd_balance"),
    ("A2_cd_boost_bwd", "abl_A2_cd_boost_bwd"),
    ("D1_scale_qk", "abl_D1_scale_qk"),
    ("C1_uniform", "abl_C1_uniform"),
    ("AC_combo", "abl_AC_combo"),
]

# SEED_C1 队列：兑现 3.5.5 预注册的 2SE 门槛需要跨 seed run 间 SE，
# 单 seed 只能算跨样本 SE（口径不同，不可混用）。C1 是 cv_nn 唯一达标组，
# 原生 3090，同机可比且不花钱，故两个补充 seed 放本机跑。
# config 由 make_seed_configs.py 从 runs/ABL_C1_uniform/config.yaml 派生，
# 启动前须经 verify_seed_c1_configs.py 校验（只允许 seed/out_dir 两行不同）。
SEED_C1_QUEUE = [
    ("SEED_C1_s20260812", "SEED_C1_s20260812"),
    ("SEED_C1_s20260813", "SEED_C1_s20260813"),
]

QUEUES = {"ABL": ABL_QUEUE, "SEED_C1": SEED_C1_QUEUE}
QUEUE = ABL_QUEUE  # 默认，保持既有调用行为不变
TARGET_EPOCHS = 150
BASELINE_DIR = ROOT / "runs" / "B002_baseline150"


def epochs_done(run_dir: Path) -> int:
    p = run_dir / "metrics.json"
    if not p.exists():
        return 0
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        return len(blob["records"] if isinstance(blob, dict) else blob)
    except Exception:
        return 0


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"started_at": None, "groups": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2),
                     encoding="utf-8")


def show_status() -> int:
    st = load_state()
    print("=" * 70)
    print("本机队列状态")
    print("=" * 70)
    b = epochs_done(BASELINE_DIR)
    print(f"B-002 baseline : {b}/{TARGET_EPOCHS} "
          f"{'✅ 完成' if b >= TARGET_EPOCHS else '⏳ 进行中'}")
    for name, cfg in QUEUE:
        d = ROOT / "runs" / f"ABL_{name}"
        n = epochs_done(d)
        rec = st["groups"].get(name, {})
        flag = ("✅" if n >= TARGET_EPOCHS else
                ("⏳" if n > 0 else "· "))
        extra = ""
        if rec.get("hours"):
            extra = f"  用时 {rec['hours']:.2f} h"
        if rec.get("error"):
            extra = f"  ★ {rec['error']}"
        print(f"  {flag} {name:<18} {n:>3}/{TARGET_EPOCHS}{extra}")
    return 0


def wait_for_baseline(poll: int = 300) -> None:
    n = epochs_done(BASELINE_DIR)
    if n >= TARGET_EPOCHS:
        print(f"[前置] B-002 已完成 {n}/{TARGET_EPOCHS}，直接开始队列")
        return
    print(f"[前置] 等 B-002 跑完（当前 {n}/{TARGET_EPOCHS}），每 {poll}s 查一次")
    while True:
        time.sleep(poll)
        n = epochs_done(BASELINE_DIR)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] B-002 {n}/{TARGET_EPOCHS}", flush=True)
        if n >= TARGET_EPOCHS:
            print("[前置] B-002 完成，开始队列")
            return


def run_group(name: str, cfg: str) -> dict:
    out_dir = ROOT / "runs" / f"ABL_{name}"
    done = epochs_done(out_dir)
    if done >= TARGET_EPOCHS:
        print(f"[跳过] {name} 已有 {done}/{TARGET_EPOCHS} epoch")
        return {"status": "skipped", "epochs": done}

    log = ROOT / "runs" / f"queue_{name}.log"
    print(f"\n{'='*70}\n[启动] {name}  config={cfg}.yaml\n"
          f"        日志 -> {log.name}\n{'='*70}", flush=True)
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1"})
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(
            [sys.executable, "scripts/train_pu.py", "--config",
             f"configs/{cfg}.yaml"],
            cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    hours = (time.time() - t0) / 3600
    n = epochs_done(out_dir)
    if p.returncode != 0 or n < TARGET_EPOCHS:
        tail = ""
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")[-600:]
        except Exception:
            pass
        print(f"[失败] {name} rc={p.returncode} epochs={n}/{TARGET_EPOCHS}")
        print(tail)
        return {"status": "failed", "rc": p.returncode, "epochs": n,
                "hours": hours, "error": f"rc={p.returncode} ep={n}"}
    print(f"[完成] {name}  {n}/{TARGET_EPOCHS} epoch  用时 {hours:.2f} h")
    return {"status": "done", "epochs": n, "hours": hours}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="只看状态")
    ap.add_argument("--skip-wait", action="store_true",
                    help="不等 B-002（仅在确认 baseline 已另有对照时用）")
    a = ap.parse_args()
    if a.status:
        return show_status()

    st = load_state()
    st["started_at"] = st.get("started_at") or datetime.now().isoformat()
    save_state(st)

    if not a.skip_wait:
        wait_for_baseline()

    total0 = time.time()
    for name, cfg in QUEUE:
        r = run_group(name, cfg)
        r["finished_at"] = datetime.now().isoformat()
        st["groups"][name] = r
        save_state(st)
        if r["status"] == "failed":
            print(f"\n★ {name} 失败，队列停止 —— 不带着失败继续跑后续组，"
                  f"避免浪费机时。修好后重跑本脚本会自动跳过已完成组。")
            return 1

    print(f"\n{'='*70}")
    print(f"[队列完成] 5 组合计 {(time.time()-total0)/3600:.2f} h")
    for name, _ in QUEUE:
        r = st["groups"].get(name, {})
        print(f"  {name:<18} {r.get('status'):<8} "
              f"{r.get('hours', 0):.2f} h")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
