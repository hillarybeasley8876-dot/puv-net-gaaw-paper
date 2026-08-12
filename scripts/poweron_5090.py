# -*- coding: utf-8 -*-
"""开 5090 实例（Action 名已由 2026-08-11 事故实测确认）。

⚠️ 血泪教训（docs/CLOUD.md §0.2.3）：
    `PoweronCompShareInstance` **不存在**（dry-run 会假阳性通过，实调 RetCode 161）。
    真正能开机的是 `StartCompShareInstance`，且它**不接受** WithoutGpu 无卡模式
    —— 传任何 WithoutGpu 值都会带 GPU 真开机并开始计费。
    所以本脚本没有"无卡模式"选项，开机就是计费。

用法：
    python scripts/poweron_5090.py                # dry-run，只看不动
    python scripts/poweron_5090.py --execute      # 真开机（开始计费！）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.cloud.compshare import CompShareClient

INSTANCE_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
OUT = ROOT / "runs" / "probe_cpod"


def redact(d):
    """递归打码敏感字段 —— 存档 JSON 不得含明文口令。"""
    SENS = {"password", "passwd", "secret", "token", "privatekey"}
    if isinstance(d, dict):
        return {k: ("<redacted>" if k.lower() in SENS else redact(v))
                for k, v in d.items()}
    if isinstance(d, list):
        return [redact(x) for x in d]
    return d


def describe(cli) -> dict:
    r = cli.call("DescribeCompShareInstance", confirm=True,
                 Region=REGION, Zone=ZONE)
    for it in (r.get("InstanceSet") or r.get("UHostSet") or []):
        if it.get("UHostId") == INSTANCE_ID or it.get("InstanceId") == INSTANCE_ID:
            return it
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="真开机（开始计费）")
    a = ap.parse_args()

    cli = CompShareClient()
    print("=" * 68)
    print(f"5090 开机  {INSTANCE_ID}  ({'真执行' if a.execute else 'DRY-RUN'})")
    print("=" * 68)

    cur = describe(cli)
    st = cur.get("State")
    print(f"当前状态: {st}")
    if st == "Running":
        print("已在运行，无需开机。")
        ssh = cur.get("SshLoginCommand") or ""
        print(f"SSH: {ssh if ssh else '(空)'}")
        return 0
    if st != "Stopped":
        print(f"★ 状态非 Stopped，不贸然开机。请人工确认。")
        return 1

    if not a.execute:
        print("\n[DRY-RUN] 将调用 StartCompShareInstance("
              f"UHostId={INSTANCE_ID}, Region={REGION}, Zone={ZONE})")
        print("加 --execute 才会真开机。开机即开始计费（约 2~3 元/小时）。")
        return 0

    print("\n[执行] StartCompShareInstance ...")
    r = cli.call("StartCompShareInstance", confirm=True,
                 Region=REGION, Zone=ZONE, UHostId=INSTANCE_ID)
    print(f"  RetCode={r.get('RetCode')} Message={r.get('Message', '')}")
    if r.get("RetCode") != 0:
        print("★ 开机失败")
        return 1

    print("\n[轮询] 等待 Running 并取 SSH 入口 ...")
    ssh = ""
    for i in range(30):
        time.sleep(10)
        cur = describe(cli)
        st = cur.get("State")
        ssh = cur.get("SshLoginCommand") or ""
        print(f"  [{i+1:02d}] State={st}  ssh={'有' if ssh else '空'}")
        if st == "Running" and ssh:
            break
    if not ssh:
        print("★ 已 Running 但 SSH 入口仍为空，稍后重查 inspect_my_5090.py")
    else:
        print(f"\n✅ 就绪。SSH: {ssh}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poweron_result.json").write_text(
        json.dumps(redact({"instance": INSTANCE_ID, "state": st,
                           "ssh_login_command": ssh,
                           "detail": cur}),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[存档] {OUT / 'poweron_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
