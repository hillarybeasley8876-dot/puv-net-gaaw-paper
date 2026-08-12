# -*- coding: utf-8 -*-
"""在远端 5090 上实测各 torch 轮子镜像源的真实下载速度，选最快的再开跑。

背景：上一轮直接用官方源 download.pytorch.org 闷头跑，实测只有 2.2 KB/s，
白等十几分钟才发现。教训：**长下载前必须先测速，20 秒定生死**。

判据：
  取 20 秒内下载的字节数 / 20 -> KB/s
  >= 2000 KB/s  -> GOOD（859 MB 约 7 分钟内完成）
  >= 500  KB/s  -> OK（约 29 分钟，可接受但慢）
  <  500  KB/s  -> BAD（换源）
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

PROBE_SECONDS = 20

# 候选源（都指向 cp310 / manylinux_2_28 / x86_64 的 torch cu128 轮子）
CANDIDATES: list[tuple[str, str]] = [
    ("aliyun-2.9.1",
     "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
     "torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"),
    ("official-2.9.1",
     "https://download.pytorch.org/whl/cu128/"
     "torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"),
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


def verdict(kbps: float) -> str:
    if kbps >= 2000:
        return "GOOD"
    if kbps >= 500:
        return "OK"
    return "BAD"


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

    print("=" * 78)
    print("[前置] 清理上一轮官方源残留 pip 缓存 + 查看磁盘")
    print("=" * 78)
    so, se, rc = run(client,
                     "du -sh /root/.cache/pip 2>/dev/null; "
                     "rm -rf /root/.cache/pip; "
                     "echo '--- after ---'; "
                     "du -sh /root/.cache/pip 2>/dev/null || echo 'cache cleared'; "
                     "df -h / | tail -1", 300)
    print(so or "(no output)")
    if se:
        print(f"  [stderr] {se[:600]}")
    results["cleanup"] = {"stdout": so, "stderr": se, "rc": rc}
    print()

    for name, url in CANDIDATES:
        print("=" * 78)
        print(f"[测速 {name}] {PROBE_SECONDS}s 采样")
        print("=" * 78)
        # curl 限时下载到 /dev/null，用 -w 输出真实平均速度
        cmd = (
            f"curl -L --max-time {PROBE_SECONDS} --connect-timeout 15 "
            f"-o /dev/null -s -w 'SIZE=%{{size_download}} SPEED=%{{speed_download}} "
            f"HTTP=%{{http_code}} CONNECT=%{{time_connect}}' '{url}'; echo"
        )
        t0 = time.time()
        so, se, rc = run(client, cmd, PROBE_SECONDS + 60)
        dt = time.time() - t0
        print(f"raw: {so}")
        if se:
            print(f"  [stderr] {se[:600]}")

        kbps = None
        size = None
        http = None
        for tok in (so or "").split():
            if tok.startswith("SIZE="):
                size = int(float(tok.split("=", 1)[1]))
            elif tok.startswith("SPEED="):
                kbps = float(tok.split("=", 1)[1]) / 1024.0
            elif tok.startswith("HTTP="):
                http = tok.split("=", 1)[1]

        if kbps is not None and size is not None:
            eta_min = (900927048 / (kbps * 1024)) / 60 if kbps > 0 else float("inf")
            v = verdict(kbps)
            print(f"  http={http}  下载 {size/1e6:.1f} MB  "
                  f"均速 {kbps:.1f} KB/s ({kbps/1024:.2f} MB/s)")
            print(f"  859MB 预计耗时 {eta_min:.1f} 分钟  ->  {v}")
        else:
            v = "PARSE_FAIL"
            eta_min = None
            print(f"  ⚠️ 解析失败  -> {v}")

        results[name] = {
            "url": url, "raw": so, "stderr": se, "rc": rc,
            "wall_seconds": round(dt, 1),
            "size_bytes": size, "kbps": round(kbps, 1) if kbps else None,
            "http": http, "eta_minutes": round(eta_min, 1) if eta_min else None,
            "verdict": v,
        }
        print()

    client.close()

    out_path = OUT_DIR / "mirror_speed.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"存档: {out_path}")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    ranked = sorted(
        [(k, v) for k, v in results.items() if v.get("kbps")],
        key=lambda kv: kv[1]["kbps"], reverse=True)
    for k, v in ranked:
        print(f"  {v['verdict']:>10}  {k:<18} {v['kbps']:>9.1f} KB/s  "
              f"ETA {v['eta_minutes']} min")
    if ranked and ranked[0][1]["verdict"] in ("GOOD", "OK"):
        print(f"\n✅ 选用: {ranked[0][0]}")
        return 0
    print("\n❌ 所有源都太慢，需另寻方案（如镜像内置 torch 的实例）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
