# -*- coding: utf-8 -*-
"""回传 5090 三个 B 组 run 的 clouds/ + ckpt/ + figures/（大件补拉）。

背景：fetch_B_runs_5090.py 只拉了判据 json/log（论文数字来源）。
本脚本补拉可视化与权重资产 —— 实测远端 runs/ 仅 88 MB，可一次拉全。

用法:
    python scripts/fetch_B_assets_5090.py            # 只拉
    python scripts/fetch_B_assets_5090.py --poweroff # 拉完立刻关机止费
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import stat as statmod
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

RUNS = ["B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"]
SUBDIRS = ["clouds", "ckpt", "figures"]


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


def pull_dir(sftp, rdir: str, ldir: Path) -> tuple[int, int, list[str]]:
    """递归拉取远端目录，返回 (文件数, 字节数, 失败项)。"""
    n = sz = 0
    fails: list[str] = []
    try:
        entries = sftp.listdir_attr(rdir)
    except FileNotFoundError:
        return 0, 0, [f"{rdir}: 远端不存在"]
    ldir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        rp = f"{rdir}/{e.filename}"
        lp = ldir / e.filename
        if statmod.S_ISDIR(e.st_mode or 0):
            a, b, f = pull_dir(sftp, rp, lp)
            n += a
            sz += b
            fails += f
            continue
        # 已存在且大小一致则跳过（可重复执行、断点续拉）
        if lp.is_file() and lp.stat().st_size == (e.st_size or -1):
            n += 1
            sz += lp.stat().st_size
            continue
        try:
            sftp.get(rp, str(lp))
            got = lp.stat().st_size
            if e.st_size is not None and got != e.st_size:
                fails.append(f"{rp}: 大小不符 {got} != {e.st_size}")
            else:
                n += 1
                sz += got
        except Exception as ex:  # noqa: BLE001
            fails.append(f"{rp}: {type(ex).__name__}")
    return n, sz, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poweroff", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("[1] SSH 连接")
    print("=" * 74)
    pwd = get_password()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, 7):
        try:
            c.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER,
                      password=pwd, timeout=30, banner_timeout=30,
                      auth_timeout=30)
            print("  ✅ 已连接")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  第 {attempt} 次失败: {type(e).__name__}")
            if attempt == 6:
                return 3
            time.sleep(20)

    sftp = c.open_sftp()
    report: dict[str, dict] = {}
    all_fails: list[str] = []

    print()
    print("=" * 74)
    print("[2] 拉取 clouds/ ckpt/ figures/")
    print("=" * 74)
    for run in RUNS:
        report[run] = {}
        print(f"\n--- {run} ---")
        for sub in SUBDIRS:
            n, sz, fails = pull_dir(
                sftp, f"{RWORK}/runs/{run}/{sub}",
                ROOT / "runs" / run / sub)
            report[run][sub] = {"files": n, "bytes": sz,
                                "fails": len(fails)}
            all_fails += fails
            flag = "✅" if not fails else "❌"
            print(f"  {flag} {sub:9s} {n:4d} 文件  {sz/1024/1024:7.2f} MB")

    sftp.close()
    c.close()

    print()
    print("=" * 74)
    print("[3] 结果")
    print("=" * 74)
    if all_fails:
        print("❌ 失败项：")
        for f in all_fails[:30]:
            print("   -", f)
    else:
        tot = sum(v[s]["bytes"] for v in report.values() for s in SUBDIRS)
        print(f"✅ 全部到位，共 {tot/1024/1024:.1f} MB")

    (ROOT / "runs" / "fetch_B_assets_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告 -> runs/fetch_B_assets_report.json")

    if args.poweroff:
        print()
        print("=" * 74)
        print("[4] 关机止费（轮询确认 Stopped）")
        print("=" * 74)
        cl = CompShareClient()
        cl.region, cl.zone = REGION, ZONE
        r = cl.call("StopCompShareInstance", UHostId=TARGET_ID)
        print("  StopCompShareInstance RetCode =", r.get("RetCode"))
        for i in range(18):
            time.sleep(10)
            d = cl.call("DescribeCompShareInstance", Limit=20)
            for h in d.get("UHostSet") or []:
                if h.get("UHostId") == TARGET_ID:
                    st = h.get("State")
                    print(f"  [{i+1:02d}] State={st}")
                    if st == "Stopped":
                        print("✅ 已关机，停止 GPU 计费")
                        return 1 if all_fails else 0
        print("⚠️ 未确认 Stopped —— 立即手工跑 poweroff_5090.py --execute")
        return 1

    return 1 if all_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
