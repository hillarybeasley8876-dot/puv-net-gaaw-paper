# -*- coding: utf-8 -*-
"""定位 CompShare 专属镜像池（compshareImage-* 命名空间）的查询 Action。

背景（上一轮实测）：
  - DescribeImage 在 cn-wlcb-01 返回 73 个镜像，全是 uimage-* 裸操作系统，
    SupportedGPUTypes 全为空 -> 这是通用云主机镜像池，不是 CompShare 的。
  - 官方 SDK 示例里 CompShareImageId 形如 compshareImage-165jmhx19ik7，
    明显属于另一个命名空间。
  - 因此 "J3 FAIL" 只说明"我查错了池子"，不能推出"5090 无可用镜像"。

策略：不再自由猜测。只在与已确认成功的两个 Action 同族的命名空间内枚举：
  已确认成功：CreateCompShareInstance(来自官方SDK) / DescribeCompShareInstance(同)
              GetCompShareInstancePrice(实测 RetCode 0)
  命名规律：<动词>CompShare<名词>
  候选动词：Describe / Get / List
  候选名词：Image / Images / InstanceImage / MachineType / GpuType / Spec / Product

同时直接调 DescribeCompShareInstance（只读，账户 0 实例应返回空集）来确认
"CompShare 族只读 Action 在本账户可用"，把"Action 不存在"和"权限不足"分开。

只读脚本。CreateCompShareInstance 绝不出现在这里。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "probe_compshare"

VERBS = ("Describe", "Get", "List")
NOUNS = ("Image", "Images", "InstanceImage", "MachineType", "MachineTypes",
         "GpuType", "GpuTypes", "Spec", "Product", "ResourcePool", "Stock")


def main() -> int:
    client = CompShareClient()
    log = {}

    # ---- 基线：确认 CompShare 族只读 Action 在本账户可用 -------------------
    print("=" * 78)
    print("[基线] DescribeCompShareInstance（官方 SDK 一手 Action 名）")
    print("-" * 78)
    r = client.call("DescribeCompShareInstance", Zone="cn-wlcb-01", Limit=20)
    log["DescribeCompShareInstance"] = r
    print(json.dumps(r, ensure_ascii=False, indent=2)[:1500])
    baseline_ok = r.get("RetCode") == 0
    print(f"\n基线可用：{baseline_ok}\n")

    # ---- 枚举镜像/规格查询 Action ----------------------------------------
    print("=" * 78)
    print("[枚举] <动词>CompShare<名词> 命名空间")
    print("-" * 78)
    found = []
    for v in VERBS:
        for n in NOUNS:
            action = f"{v}CompShare{n}"
            resp = client.call(action, Zone="cn-wlcb-01")
            ret = resp.get("RetCode")
            msg = resp.get("Message", "")
            log[action] = resp
            # 161 = Action 不存在，这是绝大多数；只打印非 161 的
            if ret != 161:
                print(f"  {action:<40} RetCode={ret}  {msg[:60]}")
                found.append(action)
    if not found:
        print("  （全部 RetCode 161 Action not found）")
    print()

    # ---- 对已找到的 Action 补参数重试 ------------------------------------
    detail = {}
    for action in found:
        # 有的 Action 需要 GpuType 才肯返回；带上再试一次
        r2 = client.call(action, Zone="cn-wlcb-01", GpuType="5090", Limit=200)
        log[action + "@with_gputype"] = r2
        detail[action] = r2
        print("=" * 78)
        print(f"[明细] {action}  (Zone=cn-wlcb-01, GpuType=5090)")
        print("-" * 78)
        body = json.dumps(r2, ensure_ascii=False, indent=2)
        print(body[:5000] + ("\n...(截断)" if len(body) > 5000 else ""))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "image_action_probe.json").write_text(
        json.dumps({"baseline_ok": baseline_ok, "found_actions": found,
                    "detail": detail}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "raw_image_action_probe.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"存在的 Action：{found if found else '（无）'}")
    print(f"存档：{OUT / 'image_action_probe.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
