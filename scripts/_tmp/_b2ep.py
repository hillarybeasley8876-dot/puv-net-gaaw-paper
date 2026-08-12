# -*- coding: utf-8 -*-
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
        try:
            pwd = base64.b64decode(h.get("Password", "")).decode("utf-8")
        except Exception:
            pwd = h.get("Password", "")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(hostname="cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn", port=28870,
          username="root", password=pwd, timeout=30,
          look_for_keys=False, allow_agent=False)
del pwd


def sh(cmd):
    _i, o, _e = c.exec_command(cmd, timeout=60)
    return o.read().decode("utf-8", errors="replace")


# 用单引号包外层, 内部双引号不需要转义
cmd = ("/root/puvnet-venv/bin/python -c "
       "'import json;h=json.load(open(\"/root/puv-net/runs/ABL_B2_adv_adaptive/history.json\"));"
       "h=h if isinstance(h,list) else h.get(\"epochs\",[]);"
       "print(\"n=\",len(h),\"last=\",h[-1].get(\"epoch\"),\"sec=\",h[-1].get(\"sec\"))'")
print("B2:", sh(cmd))
print("queue_B.log:")
print(sh("cat /root/puv-net/queue_B.log"))
c.close()
