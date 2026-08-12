# -*- coding: utf-8 -*-
"""列出 5090 上三个 B 组 run 的 clouds/ 与 ckpt/ 清单与体积（只看不拉）。

先摸清体积再决定拉什么 —— 避免盲目 sftp 几个 GB。

用法: python scripts/list_B_assets_5090.py
"""
from __future__ import annotations

import base64
import binascii
import io
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT, SSH_USER = 28870, "root"
RWORK = "/root/puv-net"
RUNS = ["B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"]


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
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER,
              password=get_password(), timeout=30,
              banner_timeout=30, auth_timeout=30)

    for run in RUNS:
        print("=" * 70)
        print(run)
        print("=" * 70)
        for sub in ["clouds", "ckpt", "figures"]:
            cmd = (f"d={RWORK}/runs/{run}/{sub}; "
                   f"if [ -d $d ]; then "
                   f"echo \"  [{sub}] n=$(ls -1 $d | wc -l) "
                   f"size=$(du -sh $d | cut -f1)\"; "
                   f"ls -1 $d | head -6 | sed 's/^/      /'; "
                   f"else echo '  [{sub}] 不存在'; fi")
            _, out, _ = c.exec_command(cmd)
            print(out.read().decode("utf-8", "replace").rstrip())
        print()

    _, out, _ = c.exec_command(f"du -sh {RWORK}/runs 2>/dev/null; "
                               f"df -h {RWORK} | tail -1")
    print("=" * 70)
    print("远端 runs/ 总量 + 磁盘")
    print("=" * 70)
    print(out.read().decode("utf-8", "replace").rstrip())
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
