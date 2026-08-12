# -*- coding: utf-8 -*-
"""诊断 B1 变慢的真实原因 (workers=8 反而比 workers=0 慢 66%)。

候选假设 (必须逐个用数据排除, 不能猜):
  H1 判别器开销 —— 但 +22% 参数解释不了 +66% 时间
  H2 CPU 抢占: 8 train workers + 8 val workers = 16 进程, 容器仅 14 核
     -> 主进程与 worker 争 CPU, 上下文切换开销吃掉收益
  H3 数据集本身是内存 h5 -> __getitem__ 极快, 根本不是 IO 瓶颈,
     多进程只带来序列化/IPC 开销 (点云 1024x3 float32 每样本要 pickle 传输)
  H4 persistent_workers + val loader 每 epoch 重建 -> 反复 fork 开销
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
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname="cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn", port=28870,
              username="root", password=pwd, timeout=30,
              look_for_keys=False, allow_agent=False)
    del pwd

    def sh(cmd, timeout=60):
        _si, so, _se = c.exec_command(cmd, timeout=timeout)
        return so.read().decode("utf-8", errors="replace").rstrip()

    print("=" * 74)
    print("H2 验证: CPU 核数 vs 进程数")
    print("=" * 74)
    print("  核数: %s" % sh("nproc"))
    print("  train_pu.py 进程数: %s" % sh("pgrep -c -f train_pu.py"))
    print("  负载均值 (1/5/15 分钟): %s" % sh("cat /proc/loadavg"))
    print("  说明: loadavg 若远超核数, 说明 CPU 过载 = H2 成立")

    print()
    print("=" * 74)
    print("H2 续: GPU 利用率 (若 GPU 闲着而 CPU 满 = 数据供给成了瓶颈)")
    print("=" * 74)
    print("  " + sh("nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used "
                    "--format=csv,noheader"))

    print()
    print("=" * 74)
    print("H3 验证: 数据集是否常驻内存 (是 -> 多进程只有 IPC 开销, 无 IO 收益)")
    print("=" * 74)
    print(sh("grep -nE 'h5py|File\\(|\\[:\\]|np\\.array|self\\.inputs|self\\.gts' "
             "/root/puv-net/puvnet/data/pu_dataset.py | head -12"))

    print()
    print("=" * 74)
    print("主进程 vs worker 的 CPU 占用 (top 快照)")
    print("=" * 74)
    print(sh("top -bn1 | head -20"))

    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
