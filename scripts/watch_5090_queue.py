#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""巡检 5090 远端队列进度（只读，不改任何远端状态）。

用法:
    python scripts/watch_5090_queue.py --queue SEED
    python scripts/watch_5090_queue.py --queue SEED --json   # 机器可读

设计约束:
- 只读: 只执行 tail / ls / nvidia-smi / ps 查询, 绝不写远端文件、不启停训练。
- 复用 deploy_5090.py 的连接参数与取密码逻辑, 不重复硬编码。
- 时区: 远端容器 UTC, 本机 UTC+8, 输出同时给两个时刻避免误判。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paramiko  # noqa: E402

import deploy_5090 as D  # noqa: E402  复用连接参数与 get_password


def connect():
    pwd = D.get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=D.SSH_HOST, port=D.SSH_PORT, username=D.SSH_USER,
                password=pwd, timeout=40, banner_timeout=40, auth_timeout=40)
    return cli


def run(cli, cmd, timeout=120):
    _in, out, err = cli.exec_command(cmd, timeout=timeout)
    so = out.read().decode("utf-8", "replace")
    se = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, so, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="SEED", choices=sorted(D.QUEUES.keys()))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    qid = a.queue
    qlog = "/root/puv-net/queue_%s.log" % qid
    # QUEUES 的元素是 (run_name, config_stem) 二元组 —— 键名当场用真实结构验过，
    # 不可直接当字符串用（否则 epoch 查询会全部落成"未开始"的假阴性）。
    expect = [item[0] if isinstance(item, (tuple, list)) else item
              for item in D.QUEUES[qid]]
    assert all(isinstance(x, str) for x in expect), "expect 必须是 run 名字符串列表"

    rep = {"queue": qid, "expect_runs": expect,
           "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")}

    cli = connect()
    try:
        # 1) 队列日志
        rc, so, _ = run(cli, "tail -n 40 %s 2>/dev/null" % qlog)
        rep["queue_log_tail"] = so.strip().splitlines()
        rep["all_done"] = "ALL_DONE" in so

        # 2) 每个 run 的 epoch 数（读 history.json 长度，权威口径）
        py = ("import json,os;"
              "base='/root/puv-net/runs';"
              "out={};"
              "\nfor r in %r:\n"
              "    p=os.path.join(base,r,'history.json')\n"
              "    if not os.path.exists(p):\n"
              "        out[r]=None; continue\n"
              "    try:\n"
              "        j=json.load(open(p))\n"
              "        recs=j.get('records',j) if isinstance(j,dict) else j\n"
              "        out[r]=len(recs)\n"
              "    except Exception as e:\n"
              "        out[r]='ERR:'+str(e)\n"
              "print(json.dumps(out))" % expect)
        rc, so, se = run(cli, "/root/puvnet-venv/bin/python -c \"%s\"" % py.replace('"', '\\"'))
        try:
            rep["epochs"] = json.loads(so.strip().splitlines()[-1])
        except Exception:
            rep["epochs"] = {"_raw": so.strip(), "_err": se.strip()}

        # 3) 训练进程（不用 pgrep -af，避免自匹配假阳性）
        rc, so, _ = run(cli, "ps -eo pid=,args=")
        alive = []
        for line in so.splitlines():
            if "train_pu.py" not in line:
                continue
            if " -c " in line or "pgrep" in line or "grep " in line:
                continue
            alive.append(line.strip()[:200])
        rep["train_procs"] = alive
        rep["n_train_procs"] = len(alive)

        # 4) GPU
        rc, so, _ = run(cli, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                             "--format=csv,noheader 2>/dev/null")
        rep["gpu"] = so.strip()

        # 5) 远端 UTC 时刻
        rc, so, _ = run(cli, "date -u '+%Y-%m-%d %H:%M:%S UTC'")
        rep["remote_time_utc"] = so.strip()
    finally:
        cli.close()

    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print("=" * 66)
    print("5090 队列巡检  queue=%s" % qid)
    print("=" * 66)
    print("本机时间 : %s" % rep["local_time"])
    print("远端时间 : %s" % rep.get("remote_time_utc"))
    print("GPU      : %s" % rep.get("gpu"))
    print("训练进程 : %d 个" % rep["n_train_procs"])
    for p in rep["train_procs"][:3]:
        print("           %s" % p)
    print("-" * 66)
    eps = rep.get("epochs") or {}
    for r in expect:
        v = eps.get(r)
        tag = "未开始" if v is None else ("%s/150" % v)
        print("  %-30s %s" % (r, tag))
    print("-" * 66)
    print("ALL_DONE : %s" % rep["all_done"])
    print("队列日志尾部:")
    for line in rep["queue_log_tail"][-12:]:
        print("  | %s" % line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
