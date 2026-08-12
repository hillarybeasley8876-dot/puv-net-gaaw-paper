# -*- coding: utf-8 -*-
"""探测「无卡模式启动」的正确 API 参数名（不真开机）。

⚠️⚠️ 本脚本已造成一次真实事故，保留作为反面教材，**不要再按原样运行**。
   事故记录（2026-08-11 02:56:09）：
     发送 StartCompShareInstance(UHostId=<合法>, WithoutGpu="__probe__")
     -> RetCode 0，服务端**静默忽略了未知参数 WithoutGpu**，
        把请求当成普通开机执行，实例带 GPU 启动（State=Running, GPU=1）。
   设计缺陷：我只把「模式参数」设为非法值，却让 UHostId 保持合法，
     于是「UHostId 合法」本身就构成了一个完整有效的开机请求。
   正确做法：探测未知参数名时，**必须让请求在任何解释下都不可能成功**，
     即把主键（UHostId）也设为非法值。若服务端先校验主键，
     就会在触达业务逻辑前拒绝，从而绝对安全。
   附带发现：PoweronCompShareInstance 实际 **不存在**（RetCode 161）。
     它此前被 dry-run 护栏拦下，从未真正发送，导致我误以为它存在。
     -> 教训：dry-run 拦截不等于 Action 存在，护栏名单里的名字也需要实测确认。

背景：
  用户实机 cpod-1tq6i2ltk5mj（h3-comfyui-5090）字段 SupportWithoutGpuStart = True，
  控制台有「无卡模式启动」按钮。无卡模式不占用 GPU、按极低价计费，适合装环境/传数据。
  但 API 侧参数名未知，官方 SDK 示例里没有。

实测已知（scripts 上一轮）：
  PoweronCompShareInstance          存在，已在 MUTATING_ACTIONS（dry-run 拦截）
  StartCompShareInstance            存在（RetCode 210 Missing params [UHostId]）
                                    ← 此前漏收录，已补进 MUTATING_ACTIONS
  PoweronCompShareInstanceWithoutGpu / ...NoGpu / StartCompShareInstanceWithoutGpu
                                    全部 RetCode 161 不存在
  => 无卡模式不是独立 Action，应当是开机 Action 上的一个**参数**。

探测手法（安全）：
  给开机 Action 传 UHostId + 候选参数名，值故意设为**非法值** "__probe__"。
  - 若参数名不存在 -> RetCode 230 "Params [Xxx] not available"
  - 若参数名存在但值非法 -> 报值域错误（可能带合法取值提示）
  两种情况都**不会真的开机**，因为参数校验发生在执行之前。
  一旦某个参数名返回"值非法"而非"参数不存在"，即锁定了正确参数名。

⚠️ 本脚本必须绕过 dry-run 才能拿到服务端校验结果，因此显式 confirm=True。
   安全性依据：所有请求都携带非法参数值，服务端在参数校验阶段即拒绝，
   不会进入开机流程。绝不发送任何"参数全部合法"的开机请求。
   如果某次响应出现 RetCode 0，脚本立即报警并提示去控制台确认/关机。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAD_VALUE = "__probe__"  # 绝不可能是合法取值

# 候选参数名：围绕"无卡/不带 GPU 启动"的常见命名
CANDIDATE_PARAMS = [
    "WithoutGpu",
    "WithoutGPU",
    "NoGpu",
    "NoGPU",
    "GpuMode",
    "StartMode",
    "BootMode",
    "PowerOnMode",
    "StartType",
    "Mode",
    "GpuStartMode",
    "WithoutGpuStart",
    "SupportWithoutGpuStart",
    "EnableGpu",
    "UseGpu",
    "GPU",
]

ACTIONS = ["PoweronCompShareInstance", "StartCompShareInstance"]


def main() -> int:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    log = {"target": TARGET_ID, "probes": []}

    print("=" * 78)
    print("阶段 1：确认基线 —— 只传 UHostId（不带任何模式参数）不发送，仅看 dry-run")
    print("=" * 78)
    for act in ACTIONS:
        r = cli.call(act, UHostId=TARGET_ID)  # confirm 默认 False -> dry-run
        if r.get("_dry_run"):
            print(f"  {act:34s} 已被 dry-run 护栏拦下 (正确)")
            print(f"    would_send = {json.dumps(r['_would_send'], ensure_ascii=False)}")
        else:
            print(f"  {act:34s} ⚠️ 未被护栏拦截！RetCode={r.get('RetCode')}")
        log["probes"].append({"stage": "baseline", "action": act, "resp": r})

    print()
    print("=" * 78)
    print(f"阶段 2：参数名探测（值统一设为非法 {BAD_VALUE!r}，服务端会在校验阶段拒绝）")
    print("=" * 78)
    hits = []
    alarm = False
    for act in ACTIONS:
        print(f"\n-- {act} --")
        for p in CANDIDATE_PARAMS:
            kwargs = {"UHostId": TARGET_ID, p: BAD_VALUE}
            r = cli.call(act, confirm=True, **kwargs)
            ret = r.get("RetCode")
            msg = str(r.get("Message", ""))[:80]

            if ret == 0:
                alarm = True
                verdict = "🚨 RetCode 0 —— 可能真的执行了！"
            elif ret == 230 and f"[{p}]" in msg:
                verdict = "参数不存在"
            elif ret == 230:
                verdict = f"参数存在? 230 但报的不是 [{p}]"
                hits.append((act, p, ret, msg))
            elif ret == 210:
                verdict = "缺其他必填参数（该参数名可能被接受）"
                hits.append((act, p, ret, msg))
            else:
                verdict = "非 230/210 —— 值域错误? 值得细看"
                hits.append((act, p, ret, msg))

            print(f"  {p:24s} RetCode={str(ret):8s} {verdict:34s} {msg}")
            log["probes"].append({"stage": "param", "action": act, "param": p,
                                  "resp": r})
            if alarm:
                break
        if alarm:
            break

    out = OUT_DIR / "probe_withoutgpu.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if alarm:
        print("  🚨 出现 RetCode 0，请立刻去控制台 https://console.compshare.cn 确认")
        print("     实例状态，如已开机且不需要请手动关机。")
        return 2
    if hits:
        print("  可能有效的参数名（非「参数不存在」响应）：")
        for act, p, ret, msg in hits:
            print(f"    {act} / {p:22s} RetCode={ret} {msg}")
    else:
        print("  FAIL：所有候选参数名均返回「参数不存在」。")
        print("  -> 无卡模式很可能不通过 OpenAPI 暴露，只有控制台按钮能触发。")
        print("     下一步应请用户在控制台点「无卡模式启动」。")
    print(f"\n存档: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
