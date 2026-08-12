# -*- coding: utf-8 -*-
"""只读：拉取用户实机 h3-comfyui-5090 的完整字段，为 venv 隔离部署做准备。

一手实测定位（scripts/probe_cpod.py, 2026-08-11）：
    Region = cn-sh2   Zone = cn-sh2-01   Action = DescribeCompShareInstance
    -> TotalCount = 1，命中 cpod-1tq6i2ltk5mj
    cpod / container 命名空间的 20 个候选 Action 全部 RetCode 161（不存在），
    容器实例与普通实例共用同一个 DescribeCompShareInstance 入口。

关注点（决定能不能装 PyTorch venv）：
    - 状态 State：截图为「关机」，需确认 API 侧字段名与取值
    - 连接方式：SSH IP / Port / 或 Jupyter/WebUI 地址
    - 镜像 ID 与 Softwares：MiniMax H3 ComfyUI 自带的 torch/CUDA/Python 版本
      （5090 = Blackwell sm_120，torch 必须是能编 sm_120 的版本）
    - 磁盘：100 GB 已用 35.34%，剩余空间是否够装独立 venv（约 8 GB）
    - 计费：Zone cn-sh2-01 询价时报 Params [ChargeType] not available，
      说明该区计费方式清单与 cn-wlcb-01 不同，需从实例字段读真实 ChargeType
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION = "cn-sh2"
ZONE = "cn-sh2-01"
OUT_DIR = ROOT / "runs" / "probe_cpod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 可能含敏感信息的字段名片段（打印时脱敏，存档时也脱敏）
SENSITIVE_HINTS = ("password", "passwd", "secret", "token", "key")


def redact(obj):
    """递归脱敏：字段名含敏感词的值替换为 <redacted>。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(h in k.lower() for h in SENSITIVE_HINTS):
                out[k] = "<redacted>"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def main() -> int:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE

    resp = cli.call("DescribeCompShareInstance", Limit=20)
    if resp.get("RetCode") != 0:
        print(f"[FAIL] RetCode={resp.get('RetCode')} {resp.get('Message')}")
        return 1

    hosts = resp.get("UHostSet") or []
    print(f"TotalCount = {resp.get('TotalCount')}, UHostSet 长度 = {len(hosts)}")
    print()

    target = None
    for h in hosts:
        hid = h.get("UHostId") or h.get("InstanceId") or h.get("Id") or ""
        if hid == TARGET_ID:
            target = h
        print(f"  实例 {hid}  Name={h.get('Name')}  State={h.get('State')}")

    if target is None:
        print(f"\n[FAIL] 未在 UHostSet 中找到 {TARGET_ID}")
        # 仍存档，便于查字段名
        (OUT_DIR / "instance_raw.json").write_text(
            json.dumps(redact(resp), ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    safe = redact(target)

    print()
    print("=" * 74)
    print(f"实例 {TARGET_ID} 全部字段（已脱敏）")
    print("=" * 74)
    for k in sorted(safe):
        v = safe[k]
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
            if len(s) > 400:
                s = s[:400] + f" ...(共 {len(s)} 字符，完整见存档)"
            print(f"  {k:26s} = {s}")
        else:
            print(f"  {k:26s} = {v}")

    print()
    print("=" * 74)
    print("关键结论提取")
    print("=" * 74)

    def pick(*names):
        for n in names:
            if n in safe and safe[n] not in (None, "", [], {}):
                return n, safe[n]
        return None, None

    for label, names in [
        ("状态", ("State", "Status", "PowerState")),
        ("计费方式", ("ChargeType",)),
        ("镜像 ID", ("CompShareImageId", "ImageId", "BasicImageId")),
        ("镜像名", ("BasicImageName", "ImageName", "ImageDescribe")),
        ("CPU", ("CPU", "Cpu")),
        ("内存(MB)", ("Memory",)),
        ("GPU 型号", ("GpuType", "SourceGpuType")),
        ("GPU 数", ("GPU", "Gpu")),
        ("到期/创建", ("CreateTime", "ExpireTime")),
        ("公网/内网 IP", ("IPSet", "IpSet", "PublicIp", "PrivateIp")),
        ("SSH", ("SshCommand", "SshIp", "SshPort", "LoginPort")),
        ("端口映射", ("PortMappings", "SoftwarePorts", "FirewallPorts")),
        ("磁盘", ("DiskSet", "Disks", "BootDiskSize")),
        ("软件栈", ("Softwares",)),
    ]:
        n, v = pick(*names)
        if n is None:
            print(f"  {label:14s} -> （无对应字段，候选 {names} 均缺失）")
        else:
            s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            print(f"  {label:14s} -> [{n}] {s[:300]}")

    out = OUT_DIR / "instance_h3_comfyui_5090.json"
    out.write_text(json.dumps(redact(resp), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print(f"完整响应（脱敏后）已存档: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
