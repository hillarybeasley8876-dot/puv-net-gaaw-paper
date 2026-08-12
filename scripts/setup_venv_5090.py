# -*- coding: utf-8 -*-
"""在 5090 容器实例上创建隔离 venv 并安装训练依赖（支持 sm_120 / Blackwell）。

摸底结论（scripts/remote_probe_5090.py, 2026-08-11 实测）：
  GPU        RTX 5090, 32607 MiB, compute_cap 12.0 (sm_120), driver 595.80, CUDA 13.2
  权限        root (uid=0)，apt-get 可用
  磁盘        overlay 100G，已用 36G，剩余 65G
  CPU/内存    14 核 / 48 GB（available 47 GB）
  系统 Python Python 3.10.12 @ /usr/bin/python3，**未安装 torch**
  ComfyUI    跑在自带独立 venv /…/x-h3-comfyui/myenv/bin/python
             -> 系统 python3 干净，新建 venv 物理上不会污染 H3 环境
  pip 源     已全局配置清华镜像；pypi 200 (2.6s)、github 200 (0.36s)

关键约束：
  compute_cap 12.0 (sm_120, Blackwell) 需要 CUDA 12.8+ 编译的 torch。
  本机 3090 用的 torch 2.5.1+cu121 **不支持 sm_120**，远端必须装 cu128 及以上。
  => 本机与云端 torch 版本不同，跨机器数值结果不可混入同一张论文表，
     必须在实验记录中标注运行环境（train_pu.py 的 env_fingerprint 已覆盖）。

本脚本只在远端 /root/puvnet-venv 下操作，不触碰 ComfyUI 目录。
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

VENV = "/root/puvnet-venv"
WORK = "/root/puv-net"
PY = f"{VENV}/bin/python"

OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# torch cu128 轮子来源（sm_120 支持）
#
# 排障记录（2026-08-11 实测）：
#   1) https://download.pytorch.org/whl/cu128  -> 官方源国内极慢
#      实测 pip 缓存 40 s 内只增 0.09 MB = 2.2 KB/s，判定 stalled，已放弃。
#   2) https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/  -> HTTP 404
#      清华 pytorch-wheels 镜像已下线（旧笔记过期，实测确认）。
#   3) https://mirrors.aliyun.com/pytorch-wheels/cu128/  -> 可用 ✅
#      但注意：**这是扁平目录列表，不是 PEP 503 index**，
#      用 --index-url 指过去会解析失败；必须用完整 whl URL 直接 pip install。
#      实测 HEAD 200 / Content-Length 900927048 / Content-Type application/zip
#
# 选 2.9.1+cu128：远端系统 Python 为 3.10.12 -> cp310；x86_64 -> manylinux_2_28。
# 该版本与 CompShare 官方镜像 cuda128_torch291_py312 的 torch 主版本一致。
#
# ⚠️ 实际结果（2026-08-11）：远端最终装成的是 **torch 2.11.0+cu128**，
#    因为上一轮官方源的下载虽被判 stalled 并中止了后台任务，pip 实际已完成安装。
#    本脚本这段阿里云下载因此成了冗余步骤。保留代码作为可复现的备用路径。
#    教训：判 stalled 后必须复核「包是否已装成」，不能只看下载速率就认定失败。
# ---------------------------------------------------------------------------
TORCH_WHEEL_NAME = "torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCH_WHEEL_SIZE = 900927048
TORCH_WHEEL_URL = (
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
    "torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
)
# 其余依赖走清华 pypi（远端已全局配置该源）
PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

STEPS: list[tuple[str, str, int]] = [
    ("确认 ComfyUI 目录不受影响（只读）",
     "ls -d /root/*comfyui* /*comfyui* 2>/dev/null; echo '---'; "
     "ls -d /root/puvnet-venv 2>/dev/null || echo 'venv 尚不存在'", 60),
    ("安装 python3-venv（若缺）",
     "python3 -m venv --help >/dev/null 2>&1 && echo 'venv module OK' || "
     "(apt-get update -qq && apt-get install -y -qq python3-venv)", 600),
    ("创建 venv",
     f"test -x {PY} && echo 'venv 已存在，跳过' || python3 -m venv {VENV}", 300),
    ("venv python 版本",
     f"{PY} -V; {VENV}/bin/pip -V", 60),
    ("升级 pip",
     f"{VENV}/bin/pip install -q --upgrade pip setuptools wheel && echo done", 600),
    # ⚠️ 坑：pip 从 **文件名** 解析包名/版本/tag，用 curl -o 改名会直接报
    #    "Invalid wheel filename (wrong number of parts)"。必须保留原始 whl 文件名。
    ("下载 torch cu128 轮子（阿里云直链，断点续传，保留原始文件名）",
     f"mkdir -p /root/wheels && cd /root/wheels && "
     f"curl -L -C - --retry 5 --retry-delay 3 --connect-timeout 30 "
     f"-o '{TORCH_WHEEL_NAME}' '{TORCH_WHEEL_URL}' "
     f"&& ls -l /root/wheels/'{TORCH_WHEEL_NAME}'", 2400),
    ("校验轮子完整性（zip 结构 + 体积）",
     f"cd /root/wheels && python3 -c \"import zipfile,os;"
     f"p='{TORCH_WHEEL_NAME}';sz=os.path.getsize(p);"
     f"assert sz=={TORCH_WHEEL_SIZE}, f'size mismatch {{sz}}';"
     "z=zipfile.ZipFile(p);bad=z.testzip();"
     "assert bad is None, f'corrupt member {bad}';"
     "print('wheel OK', sz, 'bytes,', len(z.namelist()), 'members')\"", 900),
    ("安装 torch cu128（本地轮子，sm_120）",
     f"{VENV}/bin/pip install -q /root/wheels/'{TORCH_WHEEL_NAME}' "
     f"-i {PYPI_INDEX} && echo done", 1800),
    ("安装其余依赖",
     f"{VENV}/bin/pip install -q numpy scipy h5py trimesh pyyaml tqdm "
     f"tensorboard matplotlib rtree manifold3d requests "
     f"-i {PYPI_INDEX} && echo done", 1200),
    ("验证 torch + sm_120",
     f"{PY} -c \"import torch;"
     "print('torch',torch.__version__);"
     "print('cuda',torch.version.cuda);"
     "print('avail',torch.cuda.is_available());"
     "print('arch_list',torch.cuda.get_arch_list());"
     "print('dev',torch.cuda.get_device_name(0));"
     "print('cap',torch.cuda.get_device_capability(0));"
     "print('mem_GB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))\"", 300),
    ("实际算子跑通测试（矩阵乘 + 反向）",
     f"{PY} -c \"import torch,time;"
     "d=torch.device('cuda');"
     "a=torch.randn(4096,4096,device=d,requires_grad=True);"
     "b=torch.randn(4096,4096,device=d);"
     "torch.cuda.synchronize();t=time.time();"
     "c=(a@b).sum();c.backward();torch.cuda.synchronize();"
     "print('matmul+backward OK', round(time.time()-t,3),'s');"
     "print('grad_norm', float(a.grad.norm()))\"", 300),
    ("验证其余依赖导入",
     f"{PY} -c \"import numpy,scipy,h5py,trimesh,yaml,tqdm,matplotlib;"
     "print('numpy',numpy.__version__);print('scipy',scipy.__version__);"
     "print('h5py',h5py.__version__);print('trimesh',trimesh.__version__);"
     "print('matplotlib',matplotlib.__version__)\"", 300),
    ("创建工作目录",
     f"mkdir -p {WORK}/data {WORK}/runs && ls -la {WORK}", 60),
    ("磁盘占用复核",
     "df -h / | tail -1; du -sh /root/puvnet-venv 2>/dev/null", 120),
    ("确认 ComfyUI venv 未被改动（只读比对）",
     "ls -la /root/*comfyui*/myenv/bin/python* 2>/dev/null || "
     "find / -maxdepth 3 -name 'myenv' -type d 2>/dev/null | head -5", 120),
]


def get_password() -> str:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    if r.get("RetCode") != 0:
        raise RuntimeError(f"DescribeCompShareInstance RetCode={r.get('RetCode')}")
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            if h.get("State") != "Running":
                raise RuntimeError(f"实例状态 {h.get('State')} != Running")
            raw = h.get("Password") or ""
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return raw
    raise RuntimeError(f"未找到 {TARGET_ID}")


def main() -> int:
    pwd = get_password()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                   password=pwd, timeout=40, banner_timeout=40,
                   auth_timeout=40, look_for_keys=False, allow_agent=False)
    del pwd
    print(f"已连接 {SSH_USER}@{SSH_HOST}:{SSH_PORT}\n")

    results: dict = {}
    t_all = time.time()
    for label, cmd, timeout in STEPS:
        print("=" * 78)
        print(f"[{label}]")
        print("=" * 78)
        t0 = time.time()
        try:
            _in, out, err = client.exec_command(cmd, timeout=timeout)
            so = out.read().decode("utf-8", errors="replace").rstrip()
            se = err.read().decode("utf-8", errors="replace").rstrip()
            rc = out.channel.recv_exit_status()
        except Exception as exc:  # noqa: BLE001
            so, se, rc = "", f"{type(exc).__name__}: {exc}", -1
        dt = time.time() - t0
        if so:
            print(so[:2600])
        if se:
            print(f"  [stderr] {se[:1200]}")
        print(f"  (exit={rc}, {dt:.1f}s)\n")
        results[label] = {"cmd": cmd, "stdout": so, "stderr": se,
                          "rc": rc, "seconds": round(dt, 1)}
        if rc != 0 and "跳过" not in so:
            print(f"⚠️ 该步 exit={rc}，继续执行后续步骤以便完整诊断\n")

    client.close()
    print(f"总耗时 {time.time() - t_all:.1f}s")

    out_path = OUT_DIR / "setup_venv_5090.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"存档: {out_path}")

    fails = [k for k, v in results.items() if v["rc"] != 0]
    if fails:
        print(f"\n⚠️ 失败步骤: {fails}")
        return 1
    print("\n✅ 全部步骤 exit=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
