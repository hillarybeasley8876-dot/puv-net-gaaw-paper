# -*- coding: utf-8 -*-
"""拉取 CompShare 专属镜像池，筛出适配 5090（sm_120）的 CUDA/torch 环境。

Action 来源：实测枚举命中 DescribeCompShareImages（复数，RetCode 0）。
已知约束（实测）：该 Action 不接受 Limit 参数（RetCode 230 Params [Limit] not available）。

判据（预注册）：
  K1  能拿到 compshareImage-* 形态的镜像列表（非空）。
  K2  列表中存在声明支持 5090 的镜像。
  K3  其中存在 CUDA>=12.8 或 torch>=2.7 的镜像 -> 可直接开机就用。
        若 K2 通过而 K3 不通过 -> 结论是"5090 可租但需自行装 torch cu128"，
        这是成本不是阻塞，必须如实这么写。

只读脚本。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "probe_compshare"

CUDA_RE = re.compile(r"cuda[ _-]?v?(\d+)\.(\d+)", re.I)
TORCH_RE = re.compile(r"(?:torch|pytorch)[ _-]?v?(\d+)\.(\d+)", re.I)


def main() -> int:
    client = CompShareClient()
    log = {}

    # 不带 Limit；分别试 带/不带 GpuType 两种过滤
    variants = {
        "plain": {"Zone": "cn-wlcb-01"},
        "gpu5090": {"Zone": "cn-wlcb-01", "GpuType": "5090"},
    }
    images = []
    used = None
    for name, kw in variants.items():
        r = client.call("DescribeCompShareImages", **kw)
        log[name] = r
        ret = r.get("RetCode")
        print(f"[{name}] RetCode={ret} {r.get('Message','')}")
        if ret == 0:
            # 响应里装镜像的键名未知，扫一遍找 list-of-dict
            for k, v in r.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"    -> 镜像列表字段名 = {k}, 数量 = {len(v)}")
                    if len(v) > len(images):
                        images, used = v, (name, k)
    print()

    if not images:
        print("[FAIL] K1 未拿到镜像列表。原始响应：")
        print(json.dumps(log, ensure_ascii=False, indent=2)[:4000])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "compshare_images_raw.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print(f"[PASS] K1 镜像列表来自 variant={used[0]} 字段={used[1]}，共 {len(images)} 项")
    print(f"       单条镜像的字段：{sorted(images[0].keys())}\n")

    def blob(im: dict) -> str:
        return " ".join(str(v) for v in im.values())

    def parse(im: dict) -> dict:
        b = blob(im)
        cu = CUDA_RE.search(b)
        th = TORCH_RE.search(b)
        return {
            "id": im.get("CompShareImageId") or im.get("ImageId") or im.get("Id"),
            "name": (im.get("ImageName") or im.get("Name") or ""),
            "cuda": (int(cu.group(1)), int(cu.group(2))) if cu else None,
            "torch": (int(th.group(1)), int(th.group(2))) if th else None,
            "gpus": im.get("SupportedGPUTypes") or im.get("GpuTypes") or [],
            "raw": im,
        }

    parsed = [parse(im) for im in images]

    # ---- K2 ----
    def supports_5090(p: dict) -> bool:
        if any("5090" in str(g) for g in p["gpus"]):
            return True
        return "5090" in blob(p["raw"])

    for_5090 = [p for p in parsed if supports_5090(p)]
    k2 = len(for_5090) > 0
    print(f"[{'PASS' if k2 else 'FAIL'}] K2 声明支持 5090 的镜像：{len(for_5090)} / {len(parsed)}")

    pool = for_5090 if k2 else parsed
    label = "支持5090" if k2 else "全部（5090 未单独声明，列全部供人工判断）"

    print(f"\n镜像清单（{label}），按 CUDA 降序：")
    pool.sort(key=lambda p: (p["cuda"] or (0, 0), p["torch"] or (0, 0)), reverse=True)
    for p in pool:
        print(f"  {str(p['id']):<30} cuda={str(p['cuda']):<10} torch={str(p['torch']):<10} "
              f"gpus={p['gpus']}  {p['name'][:56]}")

    # ---- K3 ----
    ready = [p for p in pool
             if (p["cuda"] and p["cuda"] >= (12, 8))
             or (p["torch"] and p["torch"] >= (2, 7))]
    k3 = len(ready) > 0
    print(f"\n[{'PASS' if k3 else 'INFO'}] K3 自带 CUDA>=12.8 或 torch>=2.7：{len(ready)} 项")
    for p in ready:
        print(f"  -> {p['id']}  cuda={p['cuda']} torch={p['torch']}  {p['name'][:70]}")
    if not k3:
        print("  （需自行装 torch cu128：约 3~5 分钟 pip，按 2.77 元/时 折合 <0.25 元，"
              "记为成本非阻塞）")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compshare_images.json").write_text(
        json.dumps({"verdict": {"K1": True, "K2": k2, "K3": k3},
                    "source_variant": used,
                    "count": len(parsed),
                    "pool": [{k: v for k, v in p.items() if k != "raw"} for p in pool]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "compshare_images_raw.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存档：{OUT / 'compshare_images.json'}")
    print("=" * 78)
    print(f"结论：K1=True K2={k2} K3={k3}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
