# -*- coding: utf-8 -*-
"""拉取 5090 smoke 的完整 history 并做时长/费用估算。

用法: python scripts/fetch_smoke_5090.py
"""
from __future__ import annotations

import base64
import binascii
import io
import json
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
OUT = ROOT / "runs" / "probe_cpod"


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
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                password=pwd, timeout=30, banner_timeout=30, auth_timeout=30,
                look_for_keys=False, allow_agent=False)
    del pwd
    _i, o, _e = cli.exec_command(
        "cat %s/runs/SMOKE_5090_B1/history.json" % RWORK, timeout=60)
    txt = o.read().decode("utf-8", errors="replace")
    _i2, o2, _e2 = cli.exec_command(
        "cat %s/runs/SMOKE_5090_B1/env.json" % RWORK, timeout=60)
    envtxt = o2.read().decode("utf-8", errors="replace")
    cli.close()

    hist = json.loads(txt)
    secs = [h["sec"] for h in hist]
    peak = max(h.get("gpu_peak_gb", 0) for h in hist)
    steady = secs[1:] if len(secs) > 1 else secs
    sep = sum(steady) / len(steady)

    print("=" * 66)
    print("5090 smoke 实测 (B1 口径, 带判别器)")
    print("=" * 66)
    try:
        env = json.loads(envtxt)
        print("  设备      : %s" % env.get("gpu_name"))
        print("  torch     : %s  cuda %s" % (env.get("torch"), env.get("cuda")))
    except Exception:  # noqa: BLE001
        pass
    print("  epoch 数  : %d" % len(secs))
    print("  各 epoch  : %s s" % [round(s, 1) for s in secs])
    print("  稳态 s/ep : %.1f s (排除首个预热 epoch)" % sep)
    print("  峰值显存  : %.3f GB / 31.36 GB" % peak)

    h150 = sep * 150 / 3600
    print("\n" + "=" * 66)
    print("B 组 150 epoch 估算")
    print("=" * 66)
    print("  单 run    : %.2f h" % h150)
    print("  3 run 合计: %.2f h" % (h150 * 3))
    for rate in (1.8, 2.5, 3.0):
        print("    按 %.1f 元/h : %.0f 元" % (rate, h150 * 3 * rate))
    print("\n  参考: 本机 3090 不带判别器实测 115.2 s/ep;")
    print("        本 smoke 是带判别器口径, 两者不可直接比倍率。")

    res = {"sec_per_epoch_list": secs, "sec_per_epoch_steady_mean": sep,
           "gpu_peak_gb": peak, "n_epochs": len(secs),
           "est_hours_per_run_150ep": h150, "est_hours_3runs": h150 * 3}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_5090.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[存档] %s" % (OUT / "smoke_5090.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
