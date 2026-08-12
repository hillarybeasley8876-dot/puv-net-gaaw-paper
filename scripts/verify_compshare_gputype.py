# -*- coding: utf-8 -*-
"""验证 GpuType 参数是否真被服务端识别，并从镜像列表里筛 5090 可用的 CUDA 镜像。

背景：上一轮探测发现 GetCompShareInstancePrice 对 GpuType=5090 返回 RetCode 0
且价格（2.77）与 4090（2.05）不同。但"价格不同"还不足以证明 5090 被支持 ——
必须排除"服务端把未知 GpuType 当默认值处理"这种静默失败。

判据（预注册，跑之前写死）：
  J1  传一个明显不存在的 GpuType（如 "9090"）：
        若返回非 0 → 说明服务端**校验** GpuType，那么 5090 返回 0 就是真支持。  [支持]
        若返回 0 且价格与 5090 相同 → 说明服务端忽略该参数，5090 结论作废。  [不可信]
  J2  5090 与 4090 与 3090 三档价格互不相同 → 佐证参数参与计价。
  J3  镜像列表里存在 SupportedGPUTypes 含 "5090" 的镜像 → 5090 确实是上架规格。
  J4  这些镜像中存在镜像名/描述含 CUDA 12.8+ 或 torch 2.7+ 的 → 可直接用官方镜像；
        否则需自行装 torch（记录为已知成本，不算阻塞）。

本脚本只调用只读 Action。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "probe_compshare"


def price_of(client: CompShareClient, gpu_type: str) -> tuple:
    r = client.call(
        "GetCompShareInstancePrice",
        Zone="cn-wlcb-01", MachineType="G", GpuType=gpu_type,
        GPU=1, CPU=16, Memory=64 * 1024, Count=1, ChargeType="Dynamic",
    )
    ret = r.get("RetCode")
    price = None
    details = r.get("PriceDetails") or []
    if details:
        price = details[0].get("Instance")
    return ret, price, r.get("Message", ""), r


def main() -> int:
    client = CompShareClient()
    log = {}

    # ---- J1 / J2: GpuType 是否被校验 ------------------------------------
    print("=" * 78)
    print("[J1/J2] GpuType 参数有效性对照")
    print("-" * 78)
    print(f"{'GpuType':<12} {'RetCode':>8} {'价格(元/时)':>12}  Message")
    table = {}
    for g in ("5090", "4090", "3090", "3080Ti", "9090", "H100", ""):
        ret, price, msg, raw = price_of(client, g)
        table[g or "<空>"] = {"RetCode": ret, "price": price, "Message": msg}
        log[f"price@{g or 'empty'}"] = raw
        print(f"{g or '<空>':<12} {ret:>8} {str(price):>12}  {msg[:50]}")
    print()

    p5090 = table["5090"]["price"]
    p4090 = table["4090"]["price"]
    p3090 = table["3090"]["price"]
    bogus = table["9090"]

    j1 = bogus["RetCode"] != 0 or bogus["price"] != p5090
    print(f"[{'PASS' if j1 else 'FAIL'}] J1 服务端校验 GpuType："
          f"伪造值 9090 -> RetCode={bogus['RetCode']}, price={bogus['price']}")
    j2 = len({p5090, p4090, p3090}) == 3
    print(f"[{'PASS' if j2 else 'FAIL'}] J2 三档价格互异："
          f"5090={p5090} / 4090={p4090} / 3090={p3090}")
    print()

    # ---- J3 / J4: 镜像 ---------------------------------------------------
    print("=" * 78)
    print("[J3/J4] 镜像清单中支持 5090 的项")
    print("-" * 78)
    imgs = []
    for zone in ("cn-wlcb-01",):
        r = client.call("DescribeImage", Zone=zone, Limit=1000)
        log[f"images@{zone}"] = r
        imgs.extend(r.get("ImageSet") or [])
    print(f"cn-wlcb-01 镜像总数：{len(imgs)}")

    with_gpu = [im for im in imgs if im.get("SupportedGPUTypes")]
    print(f"带 SupportedGPUTypes 字段（非空）的镜像：{len(with_gpu)}")

    all_types = sorted({t for im in with_gpu for t in im["SupportedGPUTypes"]})
    print(f"镜像声明过的所有 GPU 型号：{all_types}")

    for_5090 = [im for im in with_gpu if "5090" in im["SupportedGPUTypes"]]
    j3 = len(for_5090) > 0
    print(f"\n[{'PASS' if j3 else 'FAIL'}] J3 支持 5090 的镜像数：{len(for_5090)}")

    # 从镜像名 / 描述 / PrimarySoftware / IntegratedSoftware 里抽 CUDA / torch 版本
    def blob(im: dict) -> str:
        return " ".join(str(im.get(k, "")) for k in
                        ("ImageName", "OsName", "ImageDescription",
                         "PrimarySoftware", "IntegratedSoftware", "FuncType"))

    cuda_re = re.compile(r"cuda[ _-]?(\d+)\.(\d+)", re.I)
    torch_re = re.compile(r"(?:torch|pytorch)[ _-]?(\d+)\.(\d+)", re.I)

    cand = []
    for im in for_5090:
        b = blob(im)
        cu = cuda_re.search(b)
        th = torch_re.search(b)
        cu_v = (int(cu.group(1)), int(cu.group(2))) if cu else None
        th_v = (int(th.group(1)), int(th.group(2))) if th else None
        cand.append({
            "ImageId": im.get("ImageId"),
            "ImageName": im.get("ImageName"),
            "cuda": cu_v, "torch": th_v,
            "SupportedGPUTypes": im.get("SupportedGPUTypes"),
        })

    print("\n支持 5090 的镜像明细（按 CUDA 版本降序）：")
    cand.sort(key=lambda c: (c["cuda"] or (0, 0), c["torch"] or (0, 0)), reverse=True)
    for c in cand[:40]:
        print(f"  {c['ImageId']:<24} cuda={c['cuda']} torch={c['torch']}  "
              f"{c['ImageName'][:60]}")

    ready = [c for c in cand
             if (c["cuda"] and c["cuda"] >= (12, 8))
             or (c["torch"] and c["torch"] >= (2, 7))]
    j4 = len(ready) > 0
    print(f"\n[{'PASS' if j4 else 'INFO'}] J4 自带 CUDA>=12.8 或 torch>=2.7 的镜像数："
          f"{len(ready)}")
    for c in ready[:20]:
        print(f"  -> {c['ImageId']}  cuda={c['cuda']} torch={c['torch']}  "
              f"{c['ImageName'][:70]}")
    if not j4:
        print("  （无现成镜像 -> 上机后需自行 pip install torch cu128，"
              "记为已知成本，非阻塞）")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gputype_verify.json").write_text(
        json.dumps({"price_table": table, "images_for_5090": cand,
                    "verdict": {"J1": j1, "J2": j2, "J3": j3, "J4": j4}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "raw_verify.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存档：{OUT / 'gputype_verify.json'}")

    print("\n" + "=" * 78)
    print(f"结论：J1={j1} J2={j2} J3={j3} J4={j4}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
