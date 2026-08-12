# -*- coding: utf-8 -*-
"""B2 是否卡死: log/history 8h 未更新但进程在、GPU 满载。查真相。"""
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


print("=== 远端当前时间 ===")
print(sh("date"))

print("=== B2 log 尾部 12 行 ===")
print(sh("tail -12 /root/puv-net/train_ABL_B2_adv_adaptive.log"))

print("=== B2 history 最后 epoch ===")
print(sh("/root/puvnet-venv/bin/python -c \"import json;h=json.load(open('/root/puv-net/runs/ABL_B2_adv_adaptive/history.json'));h=h if isinstance(h,list) else h.get('epochs',[]);print('n=',len(h));print('last_epoch=',h[-1].get('epoch'));print('last_sec=',h[-1].get('sec'))\""))

print("=== 文件 mtime (精确) ===")
print(sh("stat -c '%y %n' /root/puv-net/train_ABL_B2_adv_adaptive.log /root/puv-net/runs/ABL_B2_adv_adaptive/history.json"))

print("=== 进程 CPU ticks (两次采样间隔 10s, 看是否推进) ===")
print(sh("cat /proc/1721/stat | awk '{print \"utime=\"$14\" stime=\"$15}'; sleep 10; cat /proc/1721/stat | awk '{print \"utime=\"$14\" stime=\"$15}'"))

print("=== 进程状态 / 线程数 ===")
print(sh("cat /proc/1721/status | grep -E 'State|Threads|VmRSS'"))

print("=== 进程 wchan (在等什么) ===")
print(sh("cat /proc/1721/wchan; echo"))

print("=== GPU 计算进程 ===")
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))

print("=== 队列脚本还活着吗 ===")
ps = sh("ps -eo pid=,args=")
for ln in ps.splitlines():
    s = ln.strip()
    if "run_B_queue" in s and " -c " not in s and "grep " not in s:
        print(" ", s)

c.close()
