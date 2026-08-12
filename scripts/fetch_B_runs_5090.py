# -*- coding: utf-8 -*-
"""回传 5090 上三个 B 组 run 的存档到本地 runs/<RUN>/。

背景（2026-08-12 事故）：cron 指令是「ALL_DONE 立刻关机 → 然后回传」，
这两步顺序矛盾 —— 回传走 SSH，关机后连不上。当时按硬要求先关了机，
导致存档留在远端。本脚本用于开机后补回传。

用法:
    python scripts/fetch_B_runs_5090.py            # 只拉取（需实例已 Running）
    python scripts/fetch_B_runs_5090.py --poweroff # 拉完立刻关机止费
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT, SSH_USER = 28870, "root"
RWORK = "/root/puv-net"

RUNS = [
    "B002_baseline150_5090",
    "ABL_B1_adv_fixed",
    "ABL_B2_adv_adaptive",
]
# 必拉文件（cron 指定 5 项 + 审计闸门需要的 metrics/clouds_manifest）
FILES = [
    "history.json",
    "summary_stats.json",
    "selection.json",
    "config.yaml",
    "env.json",
    "metrics.json",
    "clouds_manifest.json",
]
# 训练日志（在 RWORK 根目录，不在 run 目录里）
LOGS = {r: f"train_{r}.log" for r in RUNS}
LOGS_EXTRA = ["queue_B.log"]


def get_password() -> str:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            raw = h.get("Password") or ""
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return raw
    raise RuntimeError("未找到实例 " + TARGET_ID)


def wait_running(timeout: int = 300) -> bool:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = cli.call("DescribeCompShareInstance", Limit=20)
        for h in r.get("UHostSet") or []:
            if h.get("UHostId") == TARGET_ID:
                st = h.get("State")
                print(f"  State={st}  ({int(time.time()-t0)}s)")
                if st == "Running":
                    return True
        time.sleep(10)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poweroff", action="store_true",
                    help="拉完立刻关机止费")
    args = ap.parse_args()

    print("=" * 74)
    print("[1] 等待实例 Running")
    print("=" * 74)
    if not wait_running():
        print("❌ 实例未在 300s 内进入 Running，放弃（未产生 SSH 连接）")
        return 2

    print()
    print("=" * 74)
    print("[2] SSH 连接 + 拉取存档")
    print("=" * 74)
    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 关机后首次开机 sshd 可能还没起，重试几次
    for attempt in range(1, 7):
        try:
            cli.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER,
                        password=pwd, timeout=30, banner_timeout=30,
                        auth_timeout=30)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  SSH 第 {attempt} 次失败: {type(e).__name__}: {e}")
            if attempt == 6:
                print("❌ SSH 始终连不上，放弃")
                return 3
            time.sleep(20)

    sftp = cli.open_sftp()
    report: dict[str, dict[str, int | str]] = {}
    missing: list[str] = []

    for run in RUNS:
        ldir = ROOT / "runs" / run
        ldir.mkdir(parents=True, exist_ok=True)
        got: dict[str, int | str] = {}
        for fn in FILES:
            rp = f"{RWORK}/runs/{run}/{fn}"
            lp = ldir / fn
            try:
                sftp.get(rp, str(lp))
                got[fn] = lp.stat().st_size
            except Exception as e:  # noqa: BLE001
                got[fn] = f"MISS ({type(e).__name__})"
                missing.append(f"{run}/{fn}")
        # 训练日志
        for rl in [LOGS[run]]:
            rp = f"{RWORK}/{rl}"
            lp = ldir / rl
            try:
                sftp.get(rp, str(lp))
                got[rl] = lp.stat().st_size
            except Exception as e:  # noqa: BLE001
                got[rl] = f"MISS ({type(e).__name__})"
        report[run] = got
        print(f"\n--- {run} ---")
        for k, v in got.items():
            flag = "✅" if isinstance(v, int) and v > 0 else "❌"
            print(f"  {flag} {k:26s} {v}")

    # 队列日志放 runs/ 根
    for rl in LOGS_EXTRA:
        try:
            sftp.get(f"{RWORK}/{rl}", str(ROOT / "runs" / rl))
            print(f"\n✅ {rl} -> runs/{rl}")
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ {rl}: {type(e).__name__}")

    sftp.close()
    cli.close()

    print()
    print("=" * 74)
    print("[3] 回传结果")
    print("=" * 74)
    if missing:
        print("❌ 缺失文件：")
        for m in missing:
            print("   -", m)
    else:
        print("✅ 三个 run 的 7 项存档全部到位")

    (ROOT / "runs" / "fetch_B_runs_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告 -> runs/fetch_B_runs_report.json")

    if args.poweroff:
        print()
        print("=" * 74)
        print("[4] 关机止费")
        print("=" * 74)
        c = CompShareClient()
        c.region, c.zone = REGION, ZONE
        r = c.call("StopCompShareInstance", UHostId=TARGET_ID)
        print("  StopCompShareInstance RetCode =", r.get("RetCode"))
        for _ in range(18):
            time.sleep(10)
            d = c.call("DescribeCompShareInstance", Limit=20)
            for h in d.get("UHostSet") or []:
                if h.get("UHostId") == TARGET_ID and h.get("State") == "Stopped":
                    print("✅ 已关机，State=Stopped")
                    return 0 if not missing else 1
        print("⚠️ 未确认 Stopped，请手工复核 status_5090.py")
        return 1

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
