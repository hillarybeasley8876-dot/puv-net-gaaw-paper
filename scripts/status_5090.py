# -*- coding: utf-8 -*-
"""查 5090 远端当前实际状态（只读，秒回，不阻塞）。

用法: python scripts/status_5090.py
"""
from __future__ import annotations

import base64
import binascii
import io
import sys
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

CHECKS = [
    # pgrep -f 会匹配到 paramiko 的 `bash -c pgrep -af 'train_pu.py'` 包装行,
    # 造成"看起来有进程在跑"的假象 (2026-08-11 连续误判两次)。必须剔除。
    ("训练进程", "pgrep -af 'train_pu.py' | grep -v 'bash -c' | grep -v pgrep "
             "|| echo '(无 train_pu.py 进程)'"),
    ("队列进程", "pgrep -af 'run_B_queue' | grep -v 'bash -c' | grep -v pgrep "
             "|| echo '(无队列进程)'"),
    ("GPU", "nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw "
            "--format=csv,noheader"),
    ("GPU 计算进程", "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                 "--format=csv,noheader || echo '(无)'"),
    ("smoke run 目录", "ls -la %s/runs/SMOKE_5090_B1/ 2>/dev/null "
                   "|| echo '(SMOKE_5090_B1 不存在)'" % RWORK),
    ("smoke history", "cat %s/runs/SMOKE_5090_B1/history.json 2>/dev/null "
                      "| head -c 800 || echo '(无 history)'" % RWORK),
    ("所有 run", "ls -d %s/runs/*/ 2>/dev/null || echo '(runs 空)'" % RWORK),
    ("训练日志尾部", "tail -25 %s/train_*.log 2>/dev/null "
                "|| echo '(无 train_*.log)'" % RWORK),
    ("队列日志", "cat %s/queue_B.log 2>/dev/null || echo '(无 queue_B.log)'" % RWORK),
]


def get_password() -> str:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            print("实例状态: %s" % h.get("State"))
            raw = h.get("Password") or ""
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return raw
    raise RuntimeError("未找到实例")


def main() -> int:
    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                password=pwd, timeout=30, banner_timeout=30, auth_timeout=30,
                look_for_keys=False, allow_agent=False)
    del pwd
    for label, cmd in CHECKS:
        print("\n" + "=" * 66)
        print("[%s]" % label)
        print("=" * 66)
        _i, o, e = cli.exec_command(cmd, timeout=45)
        so = o.read().decode("utf-8", errors="replace").rstrip()
        se = e.read().decode("utf-8", errors="replace").rstrip()
        if so:
            print(so[:1800])
        if se:
            print("  [stderr] %s" % se[:400])
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
