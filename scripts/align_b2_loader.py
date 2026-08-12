# -*- coding: utf-8 -*-
"""把远端 B2 的 loader.num_workers 改回 0, 与 B002 baseline 对齐。

为什么要改回去
--------------
B 组三个 run 要进同一张表做同口径对比。B002 实际跑的是 workers=0 (它在
优化上线前就启动了, 内存里是旧版代码)。若 B1/B2 用 workers=8, 则三者的
**基础设施配置不一致**。

虽然 dataloader 配置理论上不影响数值结果 (augment 由 seed+i 派生, 与 worker
数无关), 但:
  1. "不影响" 是我的推断, 没有实测证据 (受控实验被 B1 占用 GPU 挡住了)
  2. 论文附录要写训练配置, 三个 run 写不同的 num_workers 需要额外解释
  3. 已实测 workers=8 并未带来加速 (B1 95 s/ep vs B002 57 s/ep, 虽混入
     判别器变量无法归因, 但至少不存在"必须保留它"的理由)

在"无证据的收益"与"额外的口径差异"之间, 选择消除差异。

B1 已经在跑, 改不了 (进程内存里是 workers=8)。这一点必须在论文里如实说明:
  B002_5090 / B2 = workers 0, B1 = workers 8。
若受控实验后续证明 num_workers 对数值无影响, 则可注明"仅影响吞吐, 不影响结果"。
"""
from __future__ import annotations

import base64
import io
import posixpath
import sys

sys.path.insert(0, r"E:\AE-CC托管\puv-net")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402
import yaml  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

RWORK = "/root/puv-net"
TARGET = "remote_ABL_B2_adv_adaptive.yaml"


def main() -> int:
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
    sftp = c.open_sftp()

    def sh(cmd, quiet=False):
        _si, so, _se = c.exec_command(cmd, timeout=60)
        o = so.read().decode("utf-8", errors="replace")
        if not quiet and o.strip():
            print(o.rstrip())
        return o

    # 安全: B2 若已启动就不能改 (改了也没用, 且会误导下次读取)
    #
    # 坑: 直接 `pgrep -af 'remote_ABL_B2_adv_adaptive'` 会把 pgrep 自己所在的
    # `bash -c` 命令行匹配上 (命令行里含有该字符串), 造成 100% 假阳性。
    # 解法: 走 `ps` 全量输出, 在 Python 侧过滤, 只认真正的 train_pu.py 进程,
    # 并排除 `bash -c` / `pgrep` / `grep` 这类自引用行。
    ps = sh("ps -eo pid=,args=", quiet=True)
    cur = ""
    for ln in ps.splitlines():
        s = ln.strip()
        if "remote_ABL_B2_adv_adaptive" not in s:
            continue
        if "train_pu.py" not in s:
            continue
        if " -c " in s or "pgrep" in s or "grep " in s:
            continue
        cur = s
        break
    if cur:
        print("★ B2 已在跑, config 改动不会生效, 放弃:")
        print("  %s" % cur)
        sftp.close()
        c.close()
        return 2

    p = posixpath.join(RWORK, "configs", TARGET)
    txt = sh("cat %s" % p, quiet=True)
    cfg = yaml.safe_load(txt)
    old = (cfg.get("loader") or {}).get("num_workers")
    cfg["loader"] = {"num_workers": 0, "pin_memory": False}
    with sftp.open(p, "w") as fh:
        fh.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print("已改 %s: loader.num_workers %s -> 0" % (TARGET, old))
    print()
    print("--- 改后的 loader 段 ---")
    sh("grep -A3 '^loader:' %s" % p)
    print()
    print("--- 三个 run 的最终 workers 口径 ---")
    for n in ("B002_baseline150_5090", "ABL_B1_adv_fixed", "ABL_B2_adv_adaptive"):
        o = sh("grep -A1 '^loader:' %s/configs/remote_%s.yaml 2>/dev/null "
               "| tail -1" % (RWORK, n), quiet=True).strip()
        # B002 虽然 config 写着 8, 但它是在优化上线前启动的, 进程内实为 0
        note = ""
        if n == "B002_baseline150_5090":
            note = "  (config 显示 8, 但进程启动于优化前, 实际运行为 0)"
        elif n == "ABL_B1_adv_fixed":
            note = "  (已在跑, 实际运行为 8)"
        print("  %-26s %s%s" % (n, o or "无", note))

    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
