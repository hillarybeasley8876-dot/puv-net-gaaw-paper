# -*- coding: utf-8 -*-
"""直接 paramiko 查 5090 真实状态, 绕开 ssh 命令行的网络问题。"""
from __future__ import annotations

import base64
import binascii
import io
import sys
import json
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


def sh(cli, cmd, timeout=45):
    si, so, se = cli.exec_command(cmd, timeout=timeout)
    return so.read().decode("utf-8", errors="replace")


def main() -> int:
    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                password=pwd, timeout=30, banner_timeout=30, auth_timeout=30,
                look_for_keys=False, allow_agent=False)
    del pwd

    # 1) 进程 etime
    print("=" * 70)
    print("B002 进程 etime (从启动到现在的累计时间)")
    print("=" * 70)
    print(sh(cli, "ps -p 619 -o pid,etime,cmd 2>/dev/null || echo 'PID 619 已死'"))

    # 2) B002 真实 epoch 进度
    print("=" * 70)
    print("B002 真实进度 (history.json)")
    print("=" * 70)
    hist = sh(cli, "python3 -c "
                  "'import json;d=json.load(open(\"/root/puv-net/runs/B002_baseline150_5090/history.json\"));"
                  "print(\"len=\",len(d));"
                  "print(\"last_epoch=\",d[-1][\"epoch\"] if d else None);"
                  "print(\"last_sec=\",d[-1].get(\"sec\") if d else None);"
                  "print(\"last_ts_in_log=\",d[-1].get(\"monitor_cd\"))' 2>&1")
    print(hist)

    # 3) queue_B.log 全量
    print("=" * 70)
    print("queue_B.log 全部条目")
    print("=" * 70)
    print(sh(cli, "cat /root/puv-net/queue_B.log 2>/dev/null || echo '无 log'"))

    # 4) 远端 3 个 yaml 的 loader 段
    print("=" * 70)
    print("远端 remote_*.yaml 的 loader 段 (确认部署是否生效)")
    print("=" * 70)
    for n in ("B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"):
        print("--- remote_%s.yaml ---" % n)
        print(sh(cli, "grep -A4 loader /root/puv-net/configs/remote_%s.yaml 2>/dev/null || echo '★ 无 loader 段'" % n))

    # 5) 远端 train_pu.py 是否已带新版本 (看 mtime + 含 [loader] 打印)
    print("=" * 70)
    print("远端 train_pu.py 是否新版本 (含 '[loader]' 行)")
    print("=" * 70)
    print(sh(cli, "ls -la /root/puv-net/scripts/train_pu.py"))
    print(sh(cli, "grep -n '\\[loader\\]' /root/puv-net/scripts/train_pu.py | head -3"))

    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
