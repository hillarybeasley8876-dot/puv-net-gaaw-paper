# -*- coding: utf-8 -*-
"""上传最小必需代码到 5090 并运行真实模型 GPU 基准。

只传 bench_gpu.py 的依赖闭环（puvnet/models/pu_transformer.py + 包 __init__），
不传数据、不传训练脚本 —— 目的只是测纯计算加速比。

可比性纪律见 bench_gpu.py 头部注释。跨机器 torch 版本差异：
  本机   3090  torch 2.5.1+cu121  sm_86
  云端   5090  torch 2.11.0+cu128 sm_120
=> 加速比只用于**排产决策**（决定消融放哪台机器跑），
   绝不把两台机器的精度数字混进同一张论文表。
"""
from __future__ import annotations

import base64
import binascii
import json
import posixpath
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
PY = f"{VENV}/bin/python"
WORK = "/root/puv-net"

# bench_gpu.py 的最小依赖闭环（相对 ROOT 的 posix 路径）
UPLOAD: list[str] = [
    "puvnet/__init__.py",
    "puvnet/models/__init__.py",
    "puvnet/models/pu_transformer.py",
    "scripts/bench_gpu.py",
]

OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def run(client: paramiko.SSHClient, cmd: str, timeout: int) -> tuple[str, str, int]:
    try:
        _in, out, err = client.exec_command(cmd, timeout=timeout)
        so = out.read().decode("utf-8", errors="replace").rstrip()
        se = err.read().decode("utf-8", errors="replace").rstrip()
        rc = out.channel.recv_exit_status()
        return so, se, rc
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}", -1


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

    # ---------- 1. 上传 ----------
    print("=" * 78)
    print("[1] 上传最小依赖闭环")
    print("=" * 78)
    sftp = client.open_sftp()
    uploaded = []
    for rel in UPLOAD:
        local = ROOT / rel
        if not local.is_file():
            print(f"  ✗ 本地缺失: {rel}")
            results.setdefault("missing", []).append(rel)
            continue
        remote = posixpath.join(WORK, rel)
        rdir = posixpath.dirname(remote)
        run(client, f"mkdir -p '{rdir}'", 60)
        sftp.put(str(local), remote)
        size = sftp.stat(remote).st_size
        ok = size == local.stat().st_size
        print(f"  {'✓' if ok else '✗'} {rel:<44} {size:>10,} B")
        uploaded.append({"rel": rel, "bytes": size, "size_match": ok})
    sftp.close()
    results["uploaded"] = uploaded
    if results.get("missing"):
        print(f"\n❌ 本地缺失文件，中止: {results['missing']}")
        client.close()
        return 1
    print()

    # ---------- 2. 导入自检 ----------
    print("=" * 78)
    print("[2] 远端导入自检 + 参数量一致性")
    print("=" * 78)
    so, se, rc = run(client,
                     f"cd {WORK} && PYTHONPATH={WORK} {PY} -c \""
                     "from puvnet.models.pu_transformer import PUTransformer;"
                     "m=PUTransformer(up_ratio=4);"
                     "n=sum(p.numel() for p in m.parameters());"
                     "print('n_params',n);"
                     "print('match_1152803', n==1152803)\"", 300)
    print(so or "(no output)")
    if se:
        print(f"  [stderr] {se[:1500]}")
    results["import_check"] = {"stdout": so, "stderr": se, "rc": rc}
    if rc != 0:
        print(f"\n❌ 导入自检失败 exit={rc}，中止基准")
        client.close()
        out_path = OUT_DIR / "bench_5090_run.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return 1
    print()

    # ---------- 3. GPU 占用现状（ComfyUI 常驻约 500MiB，需记录） ----------
    print("=" * 78)
    print("[3] 基准前 GPU 现状（ComfyUI 常驻显存需如实记录）")
    print("=" * 78)
    so, se, rc = run(client,
                     "nvidia-smi --query-gpu=name,memory.used,memory.total,"
                     "utilization.gpu,temperature.gpu,power.draw "
                     "--format=csv,noheader", 120)
    print(so or "(no output)")
    results["gpu_before"] = so
    print()

    # ---------- 4. 跑基准 ----------
    print("=" * 78)
    print("[4] 运行真实模型基准（batch64 / n_in256 / up4 / warmup10 / 40 步）")
    print("=" * 78)
    t0 = time.time()
    so, se, rc = run(client,
                     f"cd {WORK} && PYTHONPATH={WORK} {PY} scripts/bench_gpu.py "
                     f"--tag cloud_5090 --out {WORK}/runs/bench_cloud_5090.json",
                     1800)
    dt = time.time() - t0
    print(so or "(no output)")
    if se:
        print(f"  [stderr] {se[:2000]}")
    print(f"  (exit={rc}, {dt:.1f}s)")
    results["bench"] = {"stdout": so, "stderr": se, "rc": rc,
                        "wall_seconds": round(dt, 1)}
    print()

    # ---------- 5. 取回 JSON ----------
    if rc == 0:
        print("=" * 78)
        print("[5] 取回结果 JSON")
        print("=" * 78)
        so2, se2, rc2 = run(client, f"cat {WORK}/runs/bench_cloud_5090.json", 300)
        if rc2 == 0 and so2:
            local_json = ROOT / "runs" / "bench_cloud_5090.json"
            local_json.parent.mkdir(parents=True, exist_ok=True)
            local_json.write_text(so2, encoding="utf-8")
            print(f"  已写入 {local_json}")
            try:
                d = json.loads(so2)
                print(f"  中位数单步   = {d['median_s'] * 1000:.2f} ms")
                print(f"  峰值显存     = {d['peak_mem_GB']:.3f} GB")
                print(f"  推算单 epoch = {d['epoch_seconds'] / 60:.2f} min")
                print(f"  推算 100ep   = {d['est_100epoch_hours']:.2f} h")
                results["bench_json"] = d
            except json.JSONDecodeError as exc:
                print(f"  ⚠️ JSON 解析失败: {exc}")
        else:
            print(f"  ⚠️ 取回失败 exit={rc2}: {se2[:400]}")
        print()

    client.close()

    out_path = OUT_DIR / "bench_5090_run.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"存档: {out_path}")
    return 0 if results.get("bench", {}).get("rc") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
