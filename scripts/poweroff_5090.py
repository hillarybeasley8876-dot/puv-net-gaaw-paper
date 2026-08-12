# -*- coding: utf-8 -*-
"""关闭 5090 实例停止计费。

⚠️ 纪律（来自 §0.2.3 事故）：
  1. dry-run 拦截 ≠ Action 存在。关机 Action 名必须先用**非法主键**验证存在性，
     确认返回的不是 RetCode 161（Action 不存在）而是参数类错误。
  2. 变更类操作必须显式 confirm=True，且操作后必须复核真实状态。

用法：
    python scripts/poweroff_5090.py            # 只读：查状态 + 验证 Action 名
    python scripts/poweroff_5090.py --execute  # 真关机
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"

# 候选关机 Action（按可能性排序）。用非法主键验证存在性：
#   RetCode 161 -> Action 不存在
#   其他        -> Action 存在，只是参数不对
STOP_CANDIDATES = [
    "StopCompShareInstance",
    "PoweroffCompShareInstance",
    "StopUHostInstance",
]

OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def client() -> CompShareClient:
    c = CompShareClient()
    c.region, c.zone = REGION, ZONE
    return c


def show_state(cli: CompShareClient) -> dict | None:
    r = cli.call("DescribeCompShareInstance", Limit=20)
    if r.get("RetCode") != 0:
        print(f"  ⚠️ Describe RetCode={r.get('RetCode')} {r.get('Message')}")
        return None
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            print(f"  UHostId    = {h.get('UHostId')}")
            print(f"  Name       = {h.get('Name')}")
            print(f"  State      = {h.get('State')}")
            print(f"  GPU        = {h.get('GPU')}  GpuType={h.get('GpuType')}")
            print(f"  ChargeType = {h.get('ChargeType')}")
            print(f"  StartTime  = {h.get('StartTime')}")
            ssh = h.get("SshLoginCommand") or ""
            print(f"  SshLogin   = {'(空 -> 已关机)' if not ssh else ssh}")
            return h
    print(f"  ⚠️ 未找到 {TARGET_ID}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真正执行关机")
    args = ap.parse_args()

    cli = client()
    record: dict = {}

    print("=" * 78)
    print("[1] 关机前状态")
    print("=" * 78)
    before = show_state(cli)
    record["before"] = {k: before.get(k) for k in
                        ("UHostId", "State", "GPU", "ChargeType", "StartTime")
                        } if before else None
    print()

    if before and before.get("State") != "Running":
        print(f"✅ 实例当前 State={before.get('State')}，无需关机（未在计费）")
        return 0

    print("=" * 78)
    print("[2] 验证关机 Action 名存在性（用非法主键，绝不误操作）")
    print("=" * 78)
    valid_action = None
    probes = {}
    for act in STOP_CANDIDATES:
        # 非法主键：确保即使 Action 存在也不可能作用到真实资源
        r = cli.call(act, confirm=True, UHostId="uhost-__nonexistent_probe__")
        rc, msg = r.get("RetCode"), str(r.get("Message"))[:110]
        exists = rc != 161
        print(f"  {act:<30} RetCode={rc:<7} exists={exists}  {msg}")
        probes[act] = {"RetCode": rc, "Message": msg, "exists": exists}
        if exists and valid_action is None:
            valid_action = act
    record["action_probe"] = probes
    print()

    if not valid_action:
        print("❌ 所有候选关机 Action 都不存在（RetCode 161）。")
        print("   请到控制台手动关机，或补充候选名后重试。")
        (OUT_DIR / "poweroff_5090.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print(f"✅ 选用 Action: {valid_action}")
    print()

    if not args.execute:
        print("=" * 78)
        print("[3] DRY-RUN：未执行关机")
        print("=" * 78)
        print(f"  实例 {TARGET_ID} 仍在 Running 并计费。")
        print(f"  真正关机请加 --execute")
        (OUT_DIR / "poweroff_5090.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    print("=" * 78)
    print(f"[3] 执行关机 {valid_action}({TARGET_ID})")
    print("=" * 78)
    r = cli.call(valid_action, confirm=True, UHostId=TARGET_ID)
    print(f"  RetCode={r.get('RetCode')}  Message={r.get('Message')}")
    record["stop_response"] = r
    print()

    print("=" * 78)
    print("[4] 复核真实状态（轮询至 Stopped，最多 180s）")
    print("=" * 78)
    final = None
    for i in range(18):
        time.sleep(10)
        h = show_state(cli)
        print(f"  --- 第 {i + 1} 次轮询 ---")
        if h and h.get("State") != "Running":
            final = h
            break
    record["after"] = {k: final.get(k) for k in
                       ("UHostId", "State", "GPU", "ChargeType")
                       } if final else None

    (OUT_DIR / "poweroff_5090.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    if final and final.get("State") != "Running":
        print(f"\n✅ 已关机，State={final.get('State')}，停止 GPU 计费")
        return 0
    print("\n⚠️ 轮询结束仍未变为非 Running，请到控制台确认！")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
