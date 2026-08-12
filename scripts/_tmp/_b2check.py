# -*- coding: utf-8 -*-
"""确认 B2 实际运行的 loader.num_workers 是否为 0 (对齐是否生效)。"""
import base64
import io
import sys

sys.path.insert(0, r"E:\AE-CC托管\puv-net")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

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


def sh(cmd):
    _i, o, _e = c.exec_command(cmd, timeout=60)
    return o.read().decode("utf-8", errors="replace")


print("=== 所有 *.log ===")
print(sh("ls -la /root/puv-net/*.log"))

print("=== 正在跑的 train_pu.py 进程 (排除 grep 自引用) ===")
ps = sh("ps -eo pid=,args=")
n = 0
for ln in ps.splitlines():
    s = ln.strip()
    if "train_pu.py" not in s:
        continue
    if " -c " in s or "pgrep" in s or "grep " in s:
        continue
    n += 1
    if n <= 5:
        print(" ", s)
print("  总进程数 =", n, "(workers=0 应为 1; workers=8 应为 17)")

print("=== 各 train log 的 [loader] 行 ===")
print(sh("grep -H -m1 loader /root/puv-net/train_*.log"))

print("=== B2 run 目录 ===")
print(sh("ls -la /root/puv-net/runs/ABL_B2_adv_adaptive/ 2>&1 | head -20"))

print("=== queue_B.log ===")
print(sh("cat /root/puv-net/queue_B.log"))

c.close()
