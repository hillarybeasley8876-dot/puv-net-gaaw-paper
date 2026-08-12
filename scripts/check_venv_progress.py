# -*- coding: utf-8 -*-
"""查询远端 venv 安装进度（独立连接，不受本地后台任务 stdout 缓冲影响）。"""
from __future__ import annotations

import base64
import binascii
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT, SSH_USER = 28870, "root"
VENV = "/root/puvnet-venv"

CHECKS = [
    ("venv 是否已建", f"test -x {VENV}/bin/python && echo YES || echo NO"),
    ("venv 目录大小", f"du -sh {VENV} 2>/dev/null || echo 'not yet'"),
    ("已装包", f"{VENV}/bin/pip list 2>/dev/null | head -30 || echo 'pip 不可用'"),
    ("torch 是否就绪",
     f"{VENV}/bin/python -c \"import torch;print(torch.__version__, torch.version.cuda)\" "
     "2>&1 | tail -2"),
    ("正在跑的 pip 进程", "ps aux | grep -E 'pip|python' | grep -v grep | head -8"),
    ("pip 缓存下载量", "du -sh /root/.cache/pip 2>/dev/null || echo 'no cache'"),
    ("磁盘", "df -h / | tail -1"),
]


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


def main() -> int:
    pwd = get_password()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER, password=pwd,
              timeout=30, banner_timeout=30, auth_timeout=30,
              look_for_keys=False, allow_agent=False)
    del pwd
    for label, cmd in CHECKS:
        _i, o, e = c.exec_command(cmd, timeout=60)
        so = o.read().decode("utf-8", errors="replace").rstrip()
        se = e.read().decode("utf-8", errors="replace").rstrip()
        print(f"--- {label} ---")
        if so:
            print(so[:1500])
        if se:
            print(f"  [stderr] {se[:300]}")
        print()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
