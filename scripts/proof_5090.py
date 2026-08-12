# -*- coding: utf-8 -*-
"""硬证据检查: 5090 是否真在推进训练 (看 epoch 是否递增, 不看瞬时利用率)。

为什么不看 nvidia-smi 利用率:
    利用率是瞬时采样, 且云平台面板有分钟级延迟/缓存, 两者都可能骗人。
    唯一不可伪造的证据是 **history.json 的 epoch 数随时间增长**。

用法: python scripts/proof_5090.py [间隔秒数, 默认 90]
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import sys
import time
from datetime import datetime
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
RUN = "B002_baseline150_5090"


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
    raise RuntimeError("未找到实例")


def snap(cli):
    """取一次快照: epoch 数 + 最新 epoch 的 sec + GPU 瞬时。"""
    cmd = (
        "cat %s/runs/%s/history.json 2>/dev/null | python3 -c \""
        "import sys,json;"
        "d=json.load(sys.stdin);"
        "print(len(d), d[-1]['epoch'], d[-1]['sec'], d[-1].get('monitor_cd',''))"
        "\" 2>/dev/null || echo 'NO_HISTORY'" % (RWORK, RUN)
    )
    _i, o, _e = cli.exec_command(cmd, timeout=45)
    hist = o.read().decode("utf-8", errors="replace").strip()
    _i2, o2, _e2 = cli.exec_command(
        "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used "
        "--format=csv,noheader", timeout=30)
    gpu = o2.read().decode("utf-8", errors="replace").strip()
    _i3, o3, _e3 = cli.exec_command(
        "cat /proc/619/stat 2>/dev/null | awk '{print $14+$15}' "
        "|| echo NA", timeout=30)
    cputicks = o3.read().decode("utf-8", errors="replace").strip()
    return hist, gpu, cputicks


def main() -> int:
    gap = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                password=pwd, timeout=30, banner_timeout=30, auth_timeout=30,
                look_for_keys=False, allow_agent=False)
    del pwd

    print("=" * 70)
    print("硬证据: 间隔 %d 秒取两次快照, 看 epoch / CPU 时间是否推进" % gap)
    print("=" * 70)

    h1, g1, c1 = snap(cli)
    t1 = datetime.now().strftime("%H:%M:%S")
    print("\n[%s] 第一次快照" % t1)
    print("  history (条数 最新epoch sec monitor_cd): %s" % h1)
    print("  GPU 瞬时: %s" % g1)
    print("  进程 CPU ticks: %s" % c1)

    print("\n  等 %d 秒 ..." % gap)
    time.sleep(gap)

    h2, g2, c2 = snap(cli)
    t2 = datetime.now().strftime("%H:%M:%S")
    print("\n[%s] 第二次快照" % t2)
    print("  history (条数 最新epoch sec monitor_cd): %s" % h2)
    print("  GPU 瞬时: %s" % g2)
    print("  进程 CPU ticks: %s" % c2)

    print("\n" + "=" * 70)
    print("判定")
    print("=" * 70)
    verdict_ok = False
    if h1 != "NO_HISTORY" and h2 != "NO_HISTORY":
        n1 = int(h1.split()[0])
        n2 = int(h2.split()[0])
        print("  epoch 条数: %d -> %d  (%s)"
              % (n1, n2, "推进 +%d" % (n2 - n1) if n2 > n1 else "未变"))
        if n2 > n1:
            verdict_ok = True
    else:
        print("  history.json 还没生成 —— 第一个 epoch 尚未跑完")
        print("  (150 epoch 的 run, 首个 epoch 约 90-100 s 才落盘)")
    try:
        d = int(c2) - int(c1)
        print("  CPU ticks 增量: %d  (>0 说明进程在真实消耗 CPU)" % d)
        if d > 0:
            verdict_ok = True
    except ValueError:
        print("  CPU ticks: 无法解析 (%s -> %s)" % (c1, c2))

    print("\n  结论: %s" % ("✅ 在真跑" if verdict_ok
                          else "★ 无推进证据, 需进一步排查"))
    cli.close()
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
