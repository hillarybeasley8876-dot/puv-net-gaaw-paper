# -*- coding: utf-8 -*-
"""在 5090 上跑 3 epoch smoke, 实测 s/ep 与显存, 用于费用估算和倍率核对。

为什么必须先跑 smoke:
    排产表里的"26 h / 65 元"是估算值。150 epoch x 3 run 一旦起了就烧钱,
    先花 ~6 分钟拿到真实 s/ep, 才能给出可信的时长与费用, 也能提前发现
    带判别器时的显存/算子问题 (5090 是 sm_120, 与 3090 不同架构)。

用法:
    python scripts/smoke_5090.py
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import posixpath
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402
import yaml  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT, SSH_USER = 28870, "root"
RWORK = "/root/puv-net"
RVENV = "/root/puvnet-venv/bin/python"
OUT = ROOT / "runs" / "probe_cpod"

SMOKE_EPOCHS = 3


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


def sh(cli, cmd, timeout=2400):
    _i, o, e = cli.exec_command(cmd, timeout=timeout)
    so = o.read().decode("utf-8", errors="replace").rstrip()
    se = e.read().decode("utf-8", errors="replace").rstrip()
    rc = o.channel.recv_exit_status()
    return so, se, rc


def main() -> int:
    # 用 B1 配置 (带判别器, 最重的一类) 做 smoke, 只改 epochs 与 out_dir
    cfg = yaml.safe_load((ROOT / "configs" / "abl_B1_adv_fixed.yaml").read_text(
        encoding="utf-8"))
    cfg["epochs"] = SMOKE_EPOCHS
    cfg["out_dir"] = "runs/SMOKE_5090_B1"
    cfg["select_warmup"] = 1
    cfg["dump_cloud_every"] = 999
    txt = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)

    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                password=pwd, timeout=40, banner_timeout=40, auth_timeout=40,
                look_for_keys=False, allow_agent=False)
    del pwd
    print("已连接, 写 smoke config ...")
    sftp = cli.open_sftp()
    with sftp.open(posixpath.join(RWORK, "configs", "smoke_5090_b1.yaml"), "w") as fh:
        fh.write(txt)
    sftp.close()

    env = ("PUVNET_ROOT=%s PYTHONPATH=%s PYTHONIOENCODING=utf-8 PYTHONUTF8=1"
           % (RWORK, RWORK))
    print("跑 %d epoch smoke (带判别器 B1 口径) ...\n" % SMOKE_EPOCHS)
    t0 = time.time()
    so, se, rc = sh(cli, "cd %s && %s %s scripts/train_pu.py --config "
                         "configs/smoke_5090_b1.yaml 2>&1 | tail -30"
                    % (RWORK, env, RVENV))
    wall = time.time() - t0
    print(so[-3000:])
    if se:
        print("[stderr] %s" % se[-800:])
    print("\nwall=%.1f s rc=%s" % (wall, rc))

    # 取 history 里的真实 s/ep
    so2, _, _ = sh(cli, "cat %s/runs/SMOKE_5090_B1/history.json 2>/dev/null"
                   % RWORK)
    res = {"wall_seconds": wall, "rc": rc}
    if so2.strip():
        hist = json.loads(so2)
        secs = [h["sec"] for h in hist]
        peak = max(h.get("gpu_peak_gb", 0) for h in hist)
        # 首 epoch 含预热, 与本机口径一致地排除
        steady = secs[1:] if len(secs) > 1 else secs
        sep = sum(steady) / len(steady)
        res.update({"sec_per_epoch_list": secs,
                    "sec_per_epoch_steady_mean": sep,
                    "gpu_peak_gb": peak, "n_epochs": len(secs)})
        print("\n" + "=" * 60)
        print("实测结果")
        print("=" * 60)
        print("  各 epoch 秒数 : %s" % [round(s, 1) for s in secs])
        print("  稳态 s/ep     : %.1f  (排除首个预热 epoch)" % sep)
        print("  峰值显存      : %.3f GB" % peak)
        # 与本机 3090 对比 (A2 实测 115.2 s/ep, 但那是不带判别器的口径)
        print("\n  参考: 本机 3090 不带判别器 115.2 s/ep")
        print("        本表是带判别器口径, 不可直接比;")
        print("        与 5090 自身的 B 组 150 epoch 估算才是本次目的。")
        h150 = sep * 150 / 3600
        print("\n  单 run 150 epoch 预估 : %.2f h" % h150)
        print("  B 组 3 run 合计       : %.2f h" % (h150 * 3))
        for rate in (2.0, 3.0):
            print("    按 %.1f 元/h 计 : %.0f 元" % (rate, h150 * 3 * rate))
        res["est_hours_per_run"] = h150
        res["est_hours_3runs"] = h150 * 3
    else:
        print("★ 没拿到 history.json, smoke 可能失败")

    cli.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_5090.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[存档] %s" % (OUT / "smoke_5090.json"))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
