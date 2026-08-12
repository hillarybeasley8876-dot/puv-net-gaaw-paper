# -*- coding: utf-8 -*-
"""关机重启后核验 5090 远端环境是否还在（venv / 代码 / 数据 / torch / 显存）。

为什么必须实地验而不看历史存档：
    容器实例关机再开机后, /root 是否持久化取决于平台实现。
    setup_venv_5090.json 是**关机前**的状态, 不能当作现在的事实。

用法:
    python scripts/verify_remote_5090.py
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT = 28870
SSH_USER = "root"
VENV = "/root/puvnet-venv/bin/python"
WORK = "/root/puv-net"

OUT_DIR = ROOT / "runs" / "probe_cpod"

CHECKS: list[tuple[str, str]] = [
    ("venv_exists",
     f"test -x {VENV} && echo VENV_OK || echo VENV_GONE"),
    ("workdir",
     f"ls -la {WORK} 2>/dev/null || echo WORKDIR_GONE"),
    ("code_tree",
     f"ls {WORK}/scripts/train_pu.py {WORK}/puvnet/__init__.py 2>/dev/null "
     f"| wc -l; echo '--- configs ---'; ls {WORK}/configs/ 2>/dev/null | head -20"),
    ("data",
     f"du -sh {WORK}/data 2>/dev/null; find {WORK}/data -maxdepth 2 "
     f"-name '*.h5' -o -maxdepth 2 -name '*.zip' 2>/dev/null | head -10"),
    ("torch",
     f"{VENV} -c \"import torch;print('torch',torch.__version__);"
     "print('cuda',torch.version.cuda);print('avail',torch.cuda.is_available());"
     "print('dev',torch.cuda.get_device_name(0));"
     "print('cap',torch.cuda.get_device_capability(0));"
     "print('mem_GB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))\" "
     "2>&1 | tail -8"),
    ("gpu_now",
     "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
     "--format=csv,noheader"),
    ("gpu_procs",
     "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
     "--format=csv,noheader || echo '(no compute apps)'"),
    ("disk", "df -h / | tail -1"),
    ("existing_runs",
     f"ls -d {WORK}/runs/*/ 2>/dev/null | head -20 || echo '(no runs)'"),
]


def get_password() -> str:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    if r.get("RetCode") != 0:
        raise RuntimeError("DescribeCompShareInstance RetCode=%s" % r.get("RetCode"))
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            if h.get("State") != "Running":
                raise RuntimeError("实例状态为 %s, 不是 Running" % h.get("State"))
            raw = h.get("Password") or ""
            if not raw:
                raise RuntimeError("Password 字段为空")
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return raw
    raise RuntimeError("未找到实例 %s" % TARGET_ID)


def main() -> int:
    pwd = get_password()
    print("凭证已取（不显示）, 连接 %s:%s ..." % (SSH_HOST, SSH_PORT))
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t0 = time.time()
    try:
        cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                    password=pwd, timeout=40, banner_timeout=40,
                    auth_timeout=40, look_for_keys=False, allow_agent=False)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] SSH 失败: %s: %s" % (type(exc).__name__, exc))
        return 1
    finally:
        del pwd
    print("  连接成功 %.1fs\n" % (time.time() - t0))

    res = {}
    for label, cmd in CHECKS:
        print("=" * 72)
        print("[%s]" % label)
        try:
            _i, o, e = cli.exec_command(cmd, timeout=120)
            so = o.read().decode("utf-8", errors="replace").rstrip()
            se = e.read().decode("utf-8", errors="replace").rstrip()
            rc = o.channel.recv_exit_status()
        except Exception as exc:  # noqa: BLE001
            so, se, rc = "", "%s: %s" % (type(exc).__name__, exc), -1
        if so:
            print(so[:2000])
        if se:
            print("  [stderr] %s" % se[:500])
        res[label] = {"cmd": cmd, "stdout": so, "stderr": se, "rc": rc}
    cli.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "verify_remote_5090.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[存档] %s" % p)

    # 结论汇总
    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    venv_ok = "VENV_OK" in res["venv_exists"]["stdout"]
    code_n = res["code_tree"]["stdout"].split("\n")[0].strip()
    torch_ok = "avail True" in res["torch"]["stdout"]
    print("  venv        : %s" % ("在" if venv_ok else "★ 没了, 要重建"))
    print("  代码文件数  : %s (期望 2)" % code_n)
    print("  torch+CUDA  : %s" % ("可用" if torch_ok else "★ 不可用"))
    dat = res["data"]["stdout"]
    print("  数据        : %s" % (dat.split("\n")[0] if dat else "★ 空"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
