# -*- coding: utf-8 -*-
"""CompShare 5090 可行性只读探测。

目的：在不开机、不产生任何账单的前提下回答三个问题：
  Q1. 账户能不能创建 GpuType=5090 的实例？（官方 SDK 示例只列了 4090/3080Ti/3090）
  Q2. 有没有自带 CUDA 12.8+ / torch 2.7+ 的官方镜像？（5090 = sm_120，旧 torch 会
      报 no kernel image is available）
  Q3. 5090 单卡按量价格是多少？（官网页面报 3.32 元/小时，需 API 侧一手复核）

Action 名来源（一手，非猜测）：
  https://github.com/ucloud/compshare-developer-examples
    python-sdk/compshare/main.py  -> create_comp_share_instance / describe_comp_share_instance
  UCloud SDK 命名规则：snake_case 方法名 <-> 驼峰 Action 名
    create_comp_share_instance   -> CreateCompShareInstance
    describe_comp_share_instance -> DescribeCompShareInstance

本脚本**只调用只读 Action**。CreateCompShareInstance 已加入客户端的
MUTATING_ACTIONS 白名单，即使误调也会被 dry-run 拦住。

跑法：
    $env:PYTHONPATH='E:\\AE-CC托管\\puv-net'
    python scripts/probe_compshare_5090.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402


def show(title: str, resp: dict, max_chars: int = 3000) -> dict:
    print("=" * 78)
    print(title)
    print("-" * 78)
    ret = resp.get("RetCode")
    print(f"RetCode = {ret}   Message = {resp.get('Message', '')}")
    body = json.dumps(resp, ensure_ascii=False, indent=2)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n... (截断，全长 {len(body)} 字符)"
    print(body)
    print()
    return resp


def main() -> int:
    client = CompShareClient()
    print(f"客户端：{client}\n")

    results = {}

    # ---- Q2: 镜像列表 ----------------------------------------------------
    # 官方示例里 CompShareImageId 形如 compshareImage-165jmhx19ik7。
    # 对应的查询 Action 猜测面收窄到与 Create/Describe 同族的命名：
    #   DescribeCompShareImage
    # 若返回 161，再试 UCloud 主站惯用的 DescribeImage。
    for action in ("DescribeCompShareImage", "DescribeImage"):
        r = client.call(action, Zone="cn-wlcb-01", Limit=100)
        results[action] = r
        show(f"[Q2] 镜像列表 · Action={action}", r, max_chars=6000)
        if r.get("RetCode") == 0:
            break

    # ---- Q1/Q3: GPU 规格与价格 ------------------------------------------
    # 先看有没有专门的 CompShare 规格查询
    for action in ("DescribeCompShareMachineType", "DescribeCompShareInstanceType"):
        r = client.call(action, Zone="cn-wlcb-01")
        results[action] = r
        show(f"[Q1] GPU 规格 · Action={action}", r)
        if r.get("RetCode") == 0:
            break

    # 查价：CompShare 族的查价 Action
    for action in ("GetCompShareInstancePrice", "GetUHostInstancePrice"):
        r = client.call(
            action,
            Zone="cn-wlcb-01",
            MachineType="G",
            GpuType="5090",
            GPU=1,
            CPU=16,
            Memory=64 * 1024,
            Count=1,
            ChargeType="Dynamic",
        )
        results[action + "@5090"] = r
        show(f"[Q3] 5090 单卡查价 · Action={action}", r)
        if r.get("RetCode") == 0:
            break

    # 对照组：同参数换 4090（官方示例明确支持）。
    # 如果 4090 成功而 5090 报参数错误，就能定位是「GpuType 不支持 5090」
    # 而不是「查价 Action 或其它参数写错」。
    for action in ("GetCompShareInstancePrice", "GetUHostInstancePrice"):
        r = client.call(
            action,
            Zone="cn-wlcb-01",
            MachineType="G",
            GpuType="4090",
            GPU=1,
            CPU=16,
            Memory=64 * 1024,
            Count=1,
            ChargeType="Dynamic",
        )
        results[action + "@4090"] = r
        show(f"[对照] 4090 单卡查价 · Action={action}", r)
        if r.get("RetCode") == 0:
            break

    # ---- 存档 -----------------------------------------------------------
    out = Path(__file__).resolve().parents[1] / "runs" / "probe_compshare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"原始响应已存档：{out / 'probe_result.json'}")

    ok = [k for k, v in results.items() if v.get("RetCode") == 0]
    print(f"\n成功的 Action：{ok if ok else '（全部失败）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
