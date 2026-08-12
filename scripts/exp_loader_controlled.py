# -*- coding: utf-8 -*-
"""受控实验: 同机 (5090) 同 config (B1, 带判别器), 只改 num_workers, 比 s/epoch。

为什么必须做这个
----------------
B002(workers=0, 无判别器) 57.4 s/ep  vs  B1(workers=8, 有判别器) 95.2 s/ep
这两个 run **同时差了两个变量** (判别器 + workers), 无法归因。
若不做受控对照就下结论, 等于把一个来源不明的配置留在生产队列里。

做法: 用 B1 的 config 派生两份 smoke (epochs=3, out_dir 加后缀),
      唯一差别是 loader.num_workers = 0 / 8, 串行跑完比中位数。
      epochs=3 且取后 2 个 epoch (跳过含预热的 ep0)。

注意: 这会占用 GPU。B1 正在跑, 所以本脚本**只在 B1/B2 之间的空档或用户
      明确要求时执行**; 默认 --dry-run 只打印将要做什么。
"""
from __future__ import annotations

import argparse
import base64
import io
import posixpath
import sys
import time

sys.path.insert(0, r"E:\AE-CC托管\puv-net")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402
import yaml  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

RWORK = "/root/puv-net"
RVENV = "/root/puvnet-venv/bin/python"
CASES = [("W0", 0), ("W8", 8)]
N_EP = 3


def connect():
    cli = CompShareClient()
    cli.region, cli.zone = "cn-sh2", "cn-sh2-01"
    r = cli.call("DescribeCompShareInstance", Limit=20)
    pwd = None
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == "cpod-1tq6i2ltk5mj":
            raw = h.get("Password") or ""
            try:
                pwd = base64.b64decode(raw).decode("utf-8")
            except Exception:
                pwd = raw
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname="cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn", port=28870,
              username="root", password=pwd, timeout=30,
              look_for_keys=False, allow_agent=False)
    del pwd
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="真的跑 (会占 GPU); 缺省只 dry-run")
    a = ap.parse_args()

    c = connect()
    sftp = c.open_sftp()

    def sh(cmd, timeout=120, quiet=False):
        _si, so, se = c.exec_command(cmd, timeout=timeout)
        o = so.read().decode("utf-8", errors="replace")
        if not quiet and o.strip():
            print(o.rstrip())
        return o

    # 安全检查: B1/B2 在跑就不能开受控实验 (抢 GPU 会污染双方计时)
    busy = sh("pgrep -f 'train_pu.py' | head -1", quiet=True).strip()
    if busy:
        cur = sh("pgrep -af 'train_pu.py' | head -1", quiet=True).strip()
        print("★ 当前有训练在跑, 受控实验会抢 GPU 并污染双方计时:")
        print("  %s" % cur)
        print("  -> 本脚本拒绝执行。请等队列跑完 (queue_B.log 出 ALL_DONE) 再跑。")
        sftp.close()
        c.close()
        return 2

    src = posixpath.join(RWORK, "configs", "remote_ABL_B1_adv_fixed.yaml")
    base_txt = sh("cat %s" % src, quiet=True)
    base = yaml.safe_load(base_txt)

    print("=" * 72)
    print("受控实验: 同机同 config(B1), 只改 num_workers, epochs=%d" % N_EP)
    print("=" * 72)
    for tag, nw in CASES:
        cfg = yaml.safe_load(base_txt)
        cfg["epochs"] = N_EP
        cfg["out_dir"] = "runs/LOADERTEST_%s" % tag
        cfg["dump_cloud_every"] = 999      # 不落点云, 省时间
        cfg.setdefault("loader", {})
        cfg["loader"] = {"num_workers": nw, "pin_memory": nw > 0,
                         "prefetch_factor": 2, "persistent_workers": nw > 0}
        name = "loadertest_%s.yaml" % tag
        print("  %s: num_workers=%d  -> configs/%s" % (tag, nw, name))
        if a.execute:
            with sftp.open(posixpath.join(RWORK, "configs", name), "w") as fh:
                fh.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    if not a.execute:
        print("\n  (dry-run; 加 --execute 才真跑)")
        sftp.close()
        c.close()
        return 0

    for tag, nw in CASES:
        print("\n---- 跑 %s (workers=%d) ----" % (tag, nw))
        sh("cd %s && rm -rf runs/LOADERTEST_%s && %s scripts/train_pu.py "
           "--config configs/loadertest_%s.yaml > /tmp/lt_%s.log 2>&1; echo rc=$?"
           % (RWORK, tag, RVENV, tag, tag), timeout=900)
        sh("grep -E '\\[loader\\]' /tmp/lt_%s.log" % tag)
        sh("python3 -c \"import json;"
           "d=json.load(open('%s/runs/LOADERTEST_%s/history.json'));"
           "s=[r['sec'] for r in d];"
           "print('  epochs sec =', s);"
           "print('  跳过 ep0 后均值 = %%.2f s' %% (sum(s[1:])/max(len(s[1:]),1)))\""
           % (RWORK, tag))
        time.sleep(2)

    print("\n" + "=" * 72)
    print("结论: 取两组'跳过 ep0 后均值'相比, 差异 <5%% 视为无收益。")
    print("=" * 72)
    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
