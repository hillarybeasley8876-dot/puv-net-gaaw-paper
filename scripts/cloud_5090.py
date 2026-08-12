# -*- coding: utf-8 -*-
"""CompShare 5090 实例生命周期管理：查价 / 计划 / 创建 / 查状态 / 关机 / 释放。

⚠️ 会花钱。所有变更类动作默认 dry-run，必须显式 --yes 才真正执行。

规格定案（全部基于 API 实测，见 docs/CLOUD.md §0）：
    Zone             cn-wlcb-01      （实测唯一支持 5090 的可用区，其余 RetCode 230）
    MachineType      G
    GpuType          5090
    CPU / Memory     4C / 16G        = 1.93~2.05 元/时
                     选它而非最省的 2C/8G：2 核有 DataLoader 抢不到核、
                     反而拖慢 GPU 的真实风险，属于省过头。4 核留余量。
                     相对官方示例 16C/64G(2.77) 仍省约 26%。
    CompShareImageId compshareImage-1minbz219ceq
                     = cuda128_torch291_py312
                     = PyTorch 2.9.1 / CUDA 12.8 / Python 3.12 / Ubuntu 22.04
                     CUDA 12.8 是 sm_120(Blackwell) 的最低可用版本
    系统盘           100 GB CLOUD_SSD（镜像自身 30 GB + 数据 1.3 GB + 余量）

Action 名来源（一手，官方 SDK 示例，非猜测）：
    https://github.com/ucloud/compshare-developer-examples
    python-sdk/compshare/main.py
      create_comp_share_instance   -> CreateCompShareInstance
      describe_comp_share_instance -> DescribeCompShareInstance

用法：
    python scripts/cloud_5090.py price            # 查价（只读）
    python scripts/cloud_5090.py plan             # 打印将发送的创建参数（dry-run）
    python scripts/cloud_5090.py create --yes     # 真创建（开始计费！）
    python scripts/cloud_5090.py status           # 查实例状态
    python scripts/cloud_5090.py poweroff --yes   # 关机（停止实例计费）
    python scripts/cloud_5090.py terminate --yes  # 释放（彻底删除）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

ZONE = "cn-wlcb-01"
GPU_TYPE = "5090"
IMAGE_ID = "compshareImage-1minbz219ceq"   # cuda128_torch291_py312
CPU = 4
MEM_GB = 16
DISK_GB = 100

# 本机实测基线，用于成本估算
LOCAL_EPOCH_SEC = 248.7
EPOCHS = 100

SPEC = {
    "Zone": ZONE,
    "MachineType": "G",
    "GpuType": GPU_TYPE,
    "CompShareImageId": IMAGE_ID,
    "GPU": 1,
    "CPU": CPU,
    "Memory": MEM_GB * 1024,
}
DISKS = [{"IsBoot": True, "Size": DISK_GB, "Type": "CLOUD_SSD"}]

STATE_FILE = (Path(__file__).resolve().parents[1] / "runs" /
              "probe_compshare" / "instance_state.json")


def _flatten_disks(params: dict, disks: list) -> dict:
    """UCloud OpenAPI 的数组参数用 Disks.N.Xxx 下标展开形式传递。"""
    out = dict(params)
    for i, d in enumerate(disks):
        for k, v in d.items():
            out[f"Disks.{i}.{k}"] = v
    return out


def build_create_params(name: str) -> dict:
    p = _flatten_disks(SPEC, DISKS)
    p["Name"] = name
    return p


def cmd_price(client: CompShareClient) -> int:
    r = client.call("GetCompShareInstancePrice", Zone=ZONE, MachineType="G",
                    GpuType=GPU_TYPE, GPU=1, CPU=CPU, Memory=MEM_GB * 1024,
                    Count=1, ChargeType="Dynamic")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r.get("RetCode") != 0:
        return 1
    pr = (r.get("PriceDetails") or [{}])[0].get("Instance")
    print(f"\n5090 {CPU}C/{MEM_GB}G 按量单价：{pr} 元/小时")
    print(f"{'加速比':>8}{'时/run':>10}{'元/run':>10}")
    print("-" * 28)
    for sp in (1.0, 1.5, 2.0, 2.4):
        h = LOCAL_EPOCH_SEC / sp * EPOCHS / 3600
        print(f"{sp:>8.1f}{h:>10.2f}{h * pr:>10.1f}")
    print("\n⚠️ 加速比未实测，仅估算区间。")
    return 0


def cmd_plan(client: CompShareClient, name: str) -> int:
    params = build_create_params(name)
    print("=" * 78)
    print("将要发送的创建参数（dry-run，未发出）")
    print("-" * 78)
    for k in sorted(params):
        print(f"  {k:<26} = {params[k]}")
    print("-" * 78)
    print(f"镜像 {IMAGE_ID} = torch 2.9.1 / CUDA 12.8 / Py3.12 / Ubuntu 22.04")
    print("关机后实例不再计费，但云盘可能仍计费 —— 长期不用应 terminate。")
    print()
    r = client.call("CreateCompShareInstance", **params)
    ok = bool(r.get("_dry_run"))
    print("护栏检查：", "PASS（已被 dry-run 拦下）" if ok
          else "FAIL！竟然真发出去了，立刻检查 MUTATING_ACTIONS")
    return 0 if ok else 1


def cmd_create(client: CompShareClient, name: str, yes: bool) -> int:
    params = build_create_params(name)
    r = client.call("CreateCompShareInstance", confirm=yes, **params)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r.get("_dry_run"):
        print("\n[DRY-RUN] 未创建。确认无误后加 --yes")
        return 0
    if r.get("RetCode") == 0:
        ids = r.get("UHostIds") or []
        print(f"\n[OK] 已创建：{ids}")
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(
            {"UHostIds": ids, "params": params},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"实例 ID 已存档：{STATE_FILE}")
        print("⚠️ 开始计费。用完务必 poweroff 或 terminate。")
        return 0
    print(f"\n[FAIL] RetCode={r.get('RetCode')} {r.get('Message')}")
    return 1


def _saved_ids() -> list:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("UHostIds") or []
    return []


def cmd_status(client: CompShareClient) -> int:
    r = client.list_compshare_instances(zone=ZONE, limit=50)
    print(json.dumps(r, ensure_ascii=False, indent=2)[:8000])
    if r.get("RetCode") == 0:
        n = r.get("TotalCount", 0)
        print(f"\n实例数：{n}")
        for h in (r.get("UHostSet") or []):
            print(f"  {h.get('UHostId')}  State={h.get('State')}  "
                  f"GpuType={h.get('GpuType')}")
        if n == 0:
            print("（无实例 = 未产生实例计费）")
    return 0


def cmd_power(client: CompShareClient, action: str, yes: bool) -> int:
    ids = _saved_ids()
    if not ids:
        print(f"未找到已记录的实例 ID（{STATE_FILE}）。先跑 status 确认。")
        return 1
    rc = 0
    for i in ids:
        r = client.call(action, confirm=yes, UHostId=i, Zone=ZONE)
        print(f"{action} {i} ->")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        if not r.get("_dry_run") and r.get("RetCode") != 0:
            rc = 1
    if not yes:
        print("\n[DRY-RUN] 未执行。确认后加 --yes")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(
        description="CompShare 5090 实例管理（变更类动作默认 dry-run）")
    p.add_argument("command", choices=["price", "plan", "create", "status",
                                       "poweroff", "poweron", "terminate"])
    p.add_argument("--yes", action="store_true",
                   help="确认执行会花钱或改状态的动作")
    p.add_argument("--name", default="puvnet-5090")
    a = p.parse_args()

    client = CompShareClient()
    print(f"客户端：{client}\n")

    if a.command == "price":
        return cmd_price(client)
    if a.command == "plan":
        return cmd_plan(client, a.name)
    if a.command == "create":
        return cmd_create(client, a.name, a.yes)
    if a.command == "status":
        return cmd_status(client)
    mapping = {"poweroff": "PoweroffCompShareInstance",
               "poweron": "PoweronCompShareInstance",
               "terminate": "TerminateCompShareInstance"}
    return cmd_power(client, mapping[a.command], a.yes)


if __name__ == "__main__":
    raise SystemExit(main())
