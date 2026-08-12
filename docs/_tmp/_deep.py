# -*- coding: utf-8 -*-
import base64, io, sys
sys.path.insert(0, r"E:\AE-CC托管\puv-net")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import paramiko
from puvnet.cloud.compshare import CompShareClient

cli = CompShareClient(); cli.region, cli.zone = "cn-sh2", "cn-sh2-01"
r = cli.call("DescribeCompShareInstance", Limit=20)
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

for cmd in [
    "echo '=== ps etimes + lstart ==='",
    "ps -p 619 -o pid,etimes,lstart,cmd 2>/dev/null",
    "echo '=== history.json stat ==='",
    "stat -c 'size=%s mtime=%y' /root/puv-net/runs/B002_baseline150_5090/history.json",
    "echo '=== last.pt mtime (B002 写过就更新) ==='",
    "stat -c 'mtime=%y' /root/puv-net/runs/B002_baseline150_5090/ckpt/last.pt 2>/dev/null",
    "echo '=== monitor_cd 末 5 段 ==='",
    "python3 -c 'import json;d=json.load(open(\"/root/puv-net/runs/B002_baseline150_5090/history.json\"));print(\"len=\",len(d));[print(i,r[\"epoch\"],r.get(\"sec\"),r.get(\"monitor_cd\")) for i,r in enumerate(d[-5:])]'",
    "echo '=== 所有 python 进程 ==='",
    "ps -ef | grep python | grep -v grep",
]:
    si, so, se = c.exec_command(cmd, timeout=30)
    out = so.read().decode("utf-8", errors="replace").rstrip()
    if out:
        print(out)
        print("---")
c.close()
