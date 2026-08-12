# -*- coding: utf-8 -*-
"""远程摸底 5090 容器实例：环境能力 + venv 可行性 + 5090 vs 3090 加速比基准。

一手实测背景（2026-08-11）：
  实例 cpod-1tq6i2ltk5mj (h3-comfyui-5090) @ Region=cn-sh2 / Zone=cn-sh2-01
  GpuType=5090, GPU=1, GraphicsMemory.Value=32 (GB), CPU=14, Memory=49152 MB,
  DiskSet=[{Size:100, Type:Boot}], InstanceType=Container,
  镜像 = compshareImage-1tlwx8g5r0k2 "MiniMax H3｜开源音画新SOTA｜ComfyUI三套官方流程"
  SshLoginCommand = ssh -p 28870 root@cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn

凭证纪律：
  Password 字段从 DescribeCompShareInstance 实时取，base64 解码后仅驻留内存。
  绝不打印明文、绝不写入任何文件、绝不写进日志。存档的 JSON 只含命令输出。

回答的问题（决定 venv 隔离方案是否可行）：
  Q1 GPU 是否真为 RTX 5090 / 驱动与 CUDA 版本 / 显存实际可用量
  Q2 是否 root、能否 apt/pip 装包
  Q3 磁盘剩余空间（venv 约需 8 GB）
  Q4 系统 Python 版本 + 现有 torch 版本，及其是否支持 sm_120（Blackwell）
     ——若现有 torch 已支持 sm_120，可直接复用，无需装 venv
  Q5 ComfyUI 环境的关键依赖，评估装 venv 的污染风险
  Q6 网络：能否访问 pypi / github（决定传代码与装包方式）
"""
from __future__ import annotations

import base64
import binascii
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT = 28870
SSH_USER = "root"

OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (标签, 命令) —— 全部只读探测，不改远端任何状态
PROBES: list[tuple[str, str]] = [
    ("whoami", "whoami; id"),
    ("os", "cat /etc/os-release | head -4; uname -a"),
    ("nvidia-smi", "nvidia-smi"),
    ("gpu_query",
     "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,"
     "compute_cap --format=csv"),
    ("cpu_mem", "nproc; free -g | head -3"),
    ("disk", "df -h / /root /workspace 2>/dev/null | sort -u"),
    ("python_which", "which -a python python3 pip pip3; python3 -V"),
    ("torch_sys",
     "python3 -c \"import torch,sys;"
     "print('py',sys.version.split()[0]);"
     "print('torch',torch.__version__);"
     "print('cuda',torch.version.cuda);"
     "print('avail',torch.cuda.is_available());"
     "print('arch_list',torch.cuda.get_arch_list());"
     "print('dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);"
     "print('cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)\""),
    ("conda_envs", "conda env list 2>/dev/null || echo 'no conda'"),
    ("venv_module", "python3 -m venv --help >/dev/null 2>&1 && echo 'venv OK' || echo 'venv MISSING'"),
    ("pip_index", "pip3 config list 2>/dev/null; cat /etc/pip.conf 2>/dev/null || echo 'no /etc/pip.conf'"),
    ("net_pypi",
     "timeout 20 curl -s -o /dev/null -w 'pypi_code=%{http_code} t=%{time_total}\\n' "
     "https://pypi.tuna.tsinghua.edu.cn/simple/ || echo 'pypi FAIL'"),
    ("net_github",
     "timeout 20 curl -s -o /dev/null -w 'github_code=%{http_code} t=%{time_total}\\n' "
     "https://api.github.com || echo 'github FAIL'"),
    ("comfyui_pkgs",
     "pip3 list 2>/dev/null | grep -Ei 'torch|numpy|scipy|h5py|trimesh|comfy|xformers|triton' "
     "|| echo 'pip list failed'"),
    ("workspace", "ls -la / | head -25; echo '--- /root ---'; ls -la /root 2>/dev/null | head -15"),
    ("apt_ok", "which apt-get && echo 'apt present' || echo 'no apt'"),
]


def get_password() -> str:
    """从 API 实时取 Password 并 base64 解码。仅返回内存字符串。"""
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    if r.get("RetCode") != 0:
        raise RuntimeError(f"DescribeCompShareInstance RetCode={r.get('RetCode')}")
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            if h.get("State") != "Running":
                raise RuntimeError(f"实例状态为 {h.get('State')}，不是 Running")
            raw = h.get("Password") or ""
            if not raw:
                raise RuntimeError("Password 字段为空")
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                # 不是 base64，按原文使用
                return raw
    raise RuntimeError(f"未找到实例 {TARGET_ID}")


def main() -> int:
    print("从 API 取登录凭证（不落盘、不打印明文）...")
    pwd = get_password()
    print(f"  凭证已获取，长度 {len(pwd)}（内容不显示）")

    print(f"\n连接 {SSH_USER}@{SSH_HOST}:{SSH_PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t0 = time.time()
    try:
        client.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                       password=pwd, timeout=40, banner_timeout=40,
                       auth_timeout=40, look_for_keys=False, allow_agent=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] SSH 连接失败: {type(exc).__name__}: {exc}")
        return 1
    finally:
        del pwd
    print(f"  连接成功，耗时 {time.time() - t0:.1f}s")

    results: dict[str, dict] = {}
    for label, cmd in PROBES:
        print()
        print("=" * 76)
        print(f"[{label}]  $ {cmd[:110]}{'...' if len(cmd) > 110 else ''}")
        print("=" * 76)
        try:
            _in, out, err = client.exec_command(cmd, timeout=90)
            so = out.read().decode("utf-8", errors="replace").rstrip()
            se = err.read().decode("utf-8", errors="replace").rstrip()
            rc = out.channel.recv_exit_status()
        except Exception as exc:  # noqa: BLE001
            so, se, rc = "", f"{type(exc).__name__}: {exc}", -1
        if so:
            print(so[:2600])
        if se:
            print(f"  [stderr] {se[:700]}")
        print(f"  (exit={rc})")
        results[label] = {"cmd": cmd, "stdout": so, "stderr": se, "rc": rc}

    client.close()

    out_path = OUT_DIR / "remote_env_5090.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n全部输出已存档: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
