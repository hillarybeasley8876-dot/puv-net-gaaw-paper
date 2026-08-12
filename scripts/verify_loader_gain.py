# -*- coding: utf-8 -*-
"""验证 5090 dataloader 优化的真实效果 (B002 workers=0 vs B1 workers=8)。

为什么单独写这个: 加速幅度必须用**同一台机器上前后两个 run 的 sec/epoch**比,
不能用 smoke 估算, 也不能跨机器比。
"""
from __future__ import annotations

import base64
import io
import sys

sys.path.insert(0, r"E:\AE-CC托管\puv-net")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402


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
    if not pwd:
        print("★ 拿不到密码")
        return 1

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname="cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn", port=28870,
              username="root", password=pwd, timeout=30,
              look_for_keys=False, allow_agent=False)
    del pwd

    def sh(cmd, timeout=60):
        _si, so, _se = c.exec_command(cmd, timeout=timeout)
        return so.read().decode("utf-8", errors="replace").rstrip()

    print("=" * 72)
    print("B1 启动日志 (确认 [loader] 行 = 优化是否真生效)")
    print("=" * 72)
    print(sh("grep -E '\\[loader\\]|\\[环境\\]|\\[模型\\]|\\[数据\\]' "
             "/root/puv-net/train_ABL_B1_adv_fixed.log 2>/dev/null | head -8"))

    print()
    print("=" * 72)
    print("速度对比: 同一台 5090, B002(workers=0) vs B1(workers=8)")
    print("=" * 72)
    # B002 已完成: 取全部 150 epoch 的 sec 统计
    print(sh(
        "python3 -c \""
        "import json,statistics as st;"
        "d=json.load(open('/root/puv-net/runs/B002_baseline150_5090/history.json'));"
        "s=[r['sec'] for r in d if r.get('sec')];"
        "print('B002 workers=0 : n=%d  median=%.2f s/ep  mean=%.2f  min=%.2f'"
        " % (len(s), st.median(s), st.mean(s), min(s)))\""))
    # B1 在跑: 跳过第 1 个 epoch (含 worker 启动与 cudnn 预热开销)
    print(sh(
        "python3 -c \""
        "import json,statistics as st,os;"
        "p='/root/puv-net/runs/ABL_B1_adv_fixed/history.json';"
        "d=json.load(open(p)) if os.path.exists(p) else [];"
        "s=[r['sec'] for r in d if r.get('sec')];"
        "print('B1   workers=8 : n=%d  median=%.2f s/ep  mean=%.2f  min=%.2f'"
        " % (len(s), st.median(s), st.mean(s), min(s))) if len(s)>=2 else "
        "print('B1   workers=8 : 仅 %d 个 epoch 落盘, 样本不足待下轮' % len(s))\""))

    print()
    print("注意: B1 建了判别器 (+255,426 参数), 单 epoch 计算量本身比 B002 大,")
    print("      故 B1 若与 B002 持平或更快, 说明 dataloader 优化确实抵掉了")
    print("      判别器带来的额外开销; 直接相减不能归因为纯 dataloader 收益。")

    print()
    print("=" * 72)
    print("B002 最终产物 (第一个 5090 论文 run)")
    print("=" * 72)
    print(sh("ls -la /root/puv-net/runs/B002_baseline150_5090/ | head -12"))
    print(sh("echo '--- 平台区 ---'; python3 -c \""
             "import json;"
             "d=json.load(open('/root/puv-net/runs/B002_baseline150_5090/summary_stats.json'));"
             "pl=d.get('plateau',{});"
             "[print('  %-4s mean=%.6f std=%.6f best=%.6f @ep%s' % (k,v['plateau_mean'],"
             "v['plateau_std'],v['best'],v['best_epoch'])) for k,v in pl.items() "
             "if isinstance(v,dict) and v.get('plateau_mean') is not None]\""))

    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
