# -*- coding: utf-8 -*-
"""判断远端 pip 下载是停滞还是仅仅慢：两次采样 pip 缓存大小做差。

背景：torch cu128 从 download.pytorch.org 官方源下载，国内常见极慢或停滞。
清华镜像也提供 cu128 轮子（https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128），
若判定停滞则应改源重装。
"""
from __future__ import annotations

import base64
import binascii
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT, SSH_USER = 28870, "root"

SAMPLES = 3
INTERVAL = 20  # 秒


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


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 60) -> str:
    _i, o, _e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode("utf-8", errors="replace").strip()


def main() -> int:
    pwd = get_password()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER, password=pwd,
              timeout=30, banner_timeout=30, auth_timeout=30,
              look_for_keys=False, allow_agent=False)
    del pwd

    print(f"采样 {SAMPLES} 次，间隔 {INTERVAL}s —— 判断下载是否推进\n")
    sizes = []
    for i in range(SAMPLES):
        # 用 du -sb 拿字节数，比 -sh 精确
        raw = run(c, "du -sb /root/.cache/pip 2>/dev/null | cut -f1 || echo 0")
        m = re.search(r"\d+", raw or "0")
        b = int(m.group()) if m else 0
        cpu = run(c, "ps -o cputime= -p 300 2>/dev/null | tr -d ' ' || echo NA")
        alive = run(c, "test -d /proc/300 && echo alive || echo gone")
        sizes.append(b)
        print(f"  [{i + 1}/{SAMPLES}] cache={b / 1024 ** 3:.3f} GB  "
              f"pip_cputime={cpu}  proc={alive}")
        if i < SAMPLES - 1:
            time.sleep(INTERVAL)

    delta = sizes[-1] - sizes[0]
    span = INTERVAL * (SAMPLES - 1)
    rate = delta / span if span else 0
    print()
    print(f"  {span}s 内增长 {delta / 1024 ** 2:.2f} MB "
          f"-> {rate / 1024:.1f} KB/s")
    if rate < 20 * 1024:  # < 20 KB/s 视为停滞
        print("  判定：**停滞或极慢** -> 建议换清华 pytorch-wheels 源重装")
        verdict = "stalled"
    else:
        eta = (2.8 * 1024 ** 3 - sizes[-1]) / rate if rate > 0 else -1
        print(f"  判定：正在推进，预计还需 {eta / 60:.1f} 分钟（按 torch 约 2.8 GB 估）")
        verdict = "progressing"
    c.close()
    print(f"\nVERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
