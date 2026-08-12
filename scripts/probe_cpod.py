# -*- coding: utf-8 -*-
"""只读探测：定位 cpod-* 容器实例所属的 API Action 与上海二A 的 Region/Zone 代号。

背景（一手实测）：
  用户控制台存在实例 h3-comfyui-5090，ID = cpod-1tq6i2ltk5mj，地域「上海二A」，
  规格 RTX50系 x1 / 14 核 / 48 GB / 100 GB 系统盘，镜像 MiniMax H3 ComfyUI。
  此前本项目所有 create 尝试都在 cn-wlcb-01（内蒙），5090 恒返回 RetCode 8333
  ("cpu memory ratio not in 2:1 - 1:12")，而 4090 同参数返回 226604（库存不足）。
  上海二A 实际有 5090 在售 -> 8333 的真实成因极可能是「该 Zone 无此 GPU 型号」，
  错误文案是误导的。本脚本要用只读请求验证这个假设。

零变更、零花费：只发 Describe/Get 类请求（MUTATING_ACTIONS 之外）。

预注册判据（跑之前写好，不通过就如实记 FAIL，不改判据）：
  N1  存在至少一个 Action 返回 RetCode 0 且响应体包含 'cpod-1tq6i2ltk5mj'
      -> 该 Action 是 cpod 实例的正确查询入口
  N2  能确认上海二A 对应的 Region/Zone 代号（某响应里出现，或 5090 询价 RetCode 0）
  N3  上海 Zone 上 GpuType=5090 询价 RetCode 0
      注意：询价成功 != 能创建（前几轮已证 cn-wlcb-01 询价 0 但 create 8333），
      故 N3 只作为「地域代号正确」的证据，不作为「有货」的证据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 已实测存在（RetCode 0）的只读 Action
KNOWN_OK = ["DescribeCompShareInstance", "DescribeUHostInstance"]

# 候选 Action：cpod / container / pod 命名空间
CANDIDATES = [
    "DescribeCPod", "DescribeCPods", "DescribeCpod",
    "DescribeCpodInstance", "DescribeCPodInstance", "DescribeCPodInstances",
    "DescribeContainerInstance", "DescribeContainerInstances",
    "DescribeCompSharePod", "DescribeCompSharePods",
    "DescribeCompShareContainer", "DescribeCompShareContainers",
    "DescribeCompShareCPod", "DescribeCompShareCPods",
    "ListCPod", "ListCPods", "GetCPod",
    "DescribeInstance", "DescribeInstances",
    "DescribeCompShareInstances",  # 复数形式（单数已知存在）
]

# 候选 Region / Zone 组合（上海 + 内蒙对照）
REGION_ZONES = [
    ("cn-sh2", "cn-sh2-01"),
    ("cn-sh2", "cn-sh2-02"),
    ("cn-sh2", "cn-sh2-03"),
    ("cn-sh2", "cn-sh2a"),
    ("cn-sh2", ""),
    ("cn-shanghai", "cn-shanghai-2a"),
    ("cn-wlcb", "cn-wlcb-01"),  # 对照组：已知询价成功、create 失败
]


def main() -> int:
    log: dict = {"target_id": TARGET_ID, "regions": [], "known_ok": [],
                 "candidates": [], "prices": []}

    # 用空 zone 的客户端做 Action 名枚举，避免 Zone 参数干扰
    cli = CompShareClient()
    print(f"client = {cli!r}")
    print()

    print("=" * 74)
    print("[A] GetRegion —— 拿服务端权威的地域/可用区清单")
    print("=" * 74)
    r = cli.call("GetRegion")
    log["regions"].append(r)
    if r.get("RetCode") == 0:
        regions = r.get("Regions") or r.get("RegionSet") or []
        print(f"  RetCode 0, 共 {len(regions)} 项")
        sh = []
        for item in regions:
            name = str(item.get("Region", ""))
            zone = str(item.get("Zone", ""))
            bn = str(item.get("BitMaps", "") or item.get("RegionName", ""))
            if "sh" in name.lower() or "sh" in zone.lower():
                sh.append((name, zone, bn))
        print(f"  含 'sh' 的条目 {len(sh)} 个：")
        for name, zone, bn in sh:
            print(f"    Region={name:14s} Zone={zone:14s} {bn}")
        if not sh:
            print("    （无）全部 Region/Zone：")
            for item in regions[:40]:
                print(f"    {item.get('Region')} / {item.get('Zone')}")
    else:
        print(f"  RetCode={r.get('RetCode')} {r.get('Message')}")

    print()
    print("=" * 74)
    print("[B] 已知 Action x Region/Zone —— 找 cpod 实例")
    print("=" * 74)
    for act in KNOWN_OK:
        for region, zone in REGION_ZONES:
            c = CompShareClient()
            c.region, c.zone = region, zone
            resp = c.call(act)
            body = json.dumps(resp, ensure_ascii=False)
            hit = TARGET_ID in body
            print(f"  {act:28s} {region:12s}/{zone or '-':12s} "
                  f"RetCode={str(resp.get('RetCode')):7s} "
                  f"Total={str(resp.get('TotalCount')):5s} hit={hit}")
            log["known_ok"].append({"action": act, "region": region,
                                    "zone": zone, "resp": resp, "hit": hit})

    print()
    print("=" * 74)
    print("[C] 枚举 cpod / container 命名空间（Action 名是否存在）")
    print("=" * 74)
    found = []
    for act in CANDIDATES:
        resp = cli.call(act)
        ret = resp.get("RetCode")
        msg = str(resp.get("Message", ""))[:52]
        exists = ret != 161
        body = json.dumps(resp, ensure_ascii=False)
        hit = TARGET_ID in body
        print(f"{'**' if exists else '  '}{act:30s} RetCode={str(ret):8s} "
              f"hit={str(hit):5s} {msg}")
        log["candidates"].append({"action": act, "resp": resp,
                                  "exists": exists, "hit": hit})
        if exists:
            found.append(act)

    print()
    print("=" * 74)
    print("[D] GpuType=5090 询价 x Region/Zone（只读）")
    print("=" * 74)
    for region, zone in REGION_ZONES:
        c = CompShareClient()
        c.region, c.zone = region, zone
        # 按用户实机规格 14C/48G 询价
        resp = c.call("GetCompShareInstancePrice", MachineType="G",
                      GpuType="5090", GPU=1, CPU=14, Memory=48 * 1024,
                      Count=1, ChargeType="Dynamic")
        ret = resp.get("RetCode")
        msg = str(resp.get("Message", ""))[:46]
        pset = resp.get("PriceSet") or []
        price = pset[0].get("Price") if pset else None
        print(f"  {region:12s}/{zone or '-':12s} RetCode={str(ret):8s} "
              f"price={price} {msg}")
        log["prices"].append({"region": region, "zone": zone, "resp": resp})

    out = OUT_DIR / "probe_cpod.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 74)
    print("判据结论")
    print("=" * 74)
    hits = sorted({e["action"] for e in log["known_ok"] + log["candidates"]
                   if e.get("hit")})
    print(f"  N1 能看见 {TARGET_ID} 的 Action: "
          f"{'PASS -> ' + ', '.join(hits) if hits else 'FAIL（无）'}")
    ok_price = [(e["region"], e["zone"]) for e in log["prices"]
                if e["resp"].get("RetCode") == 0]
    print(f"  N2/N3 5090 询价 RetCode 0 的 Region/Zone: {ok_price or 'FAIL（无）'}")
    print(f"  存在的候选 Action（RetCode != 161）: {found or '无'}")
    print(f"\n原始响应已存档: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
