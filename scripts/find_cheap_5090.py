# -*- coding: utf-8 -*-
"""在"卡型锁定 5090"的前提下，找最便宜的 5090 配置。

用户明确：卡就用 5090，要找便宜的。所以不换卡型，只在 5090 这一档里
横扫可变因素找最低价。

之前 2.77 元/时 是抄官方 SDK 示例的 16C/64G 配比查出来的 —— 那是给 4090
写的示例，从未针对本课题优化。真实可变因素：

  1. CPU 核数   —— 本课题 GPU-bound，h5 全量仅 1.01 GB 且实测 0.7 s 全量载入内存，
                  DataLoader 压力极小，16 核大概率过剩
  2. 内存       —— 数据集 1 GB 级，64 GB 明显过剩
  3. 地域 Zone  —— 不同可用区常有价差
  4. 计费方式   —— Dynamic(按量) / Month / Year

本课题实测约束（非猜测）：
  显存峰值 5.918 GB @ batch64 全量  -> 5090 的 32 GB 极度富余
  全量 h5  1,059,842,048 B ≈ 1.01 GB

判据（预注册）：
  M1  同一 5090，扫 CPU/内存组合，找出最低单价配置。
  M2  用最省配比扫全部可用 Zone，确认是否存在地域价差。
  M3  报告"最省配置"相对基准 16C/64G 的降幅；若降幅 < 5%，
      诚实说明"配比对价格影响很小，2.77 基本是 5090 地板价"，不硬吹省钱。

只读脚本，零花费。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "probe_compshare"

GPU_TYPE = "5090"
BASE_CPU, BASE_MEM_GB = 16, 64      # 官方示例配比 = 之前查到 2.77 的那个

CPU_MEM_GRID = [
    (2, 8), (4, 8), (4, 16), (8, 16), (8, 32),
    (12, 32), (12, 48), (16, 32), (16, 48), (16, 64),
]

ZONES = ["cn-wlcb-01", "cn-bj2-02", "cn-sh2-01", "cn-gd-02", "us-den-01"]

LOCAL_EPOCH_SEC = 248.7   # 本机 3090 实测
EPOCHS = 100
N_RUNS = 9


def price(client, cpu, mem_gb, zone, charge="Dynamic"):
    r = client.call("GetCompShareInstancePrice", Zone=zone, MachineType="G",
                    GpuType=GPU_TYPE, GPU=1, CPU=cpu, Memory=mem_gb * 1024,
                    Count=1, ChargeType=charge)
    if r.get("RetCode") != 0:
        return None, r
    d = (r.get("PriceDetails") or [{}])[0]
    o = (r.get("OriginalPriceDetails") or [{}])[0]
    return {"price": d.get("Instance"), "orig": o.get("Instance")}, r


def main() -> int:
    client = CompShareClient()
    raw = {}

    print("=" * 80)
    print("[M1] 5090 @ cn-wlcb-01 —— 扫 CPU/内存配比（按量）")
    print("-" * 80)
    print(f"{'CPU':>4}{'内存GB':>8}{'元/时':>10}{'原价':>10}")
    print("-" * 80)
    grid, base_price = [], None
    for cpu, mem in CPU_MEM_GRID:
        p, r = price(client, cpu, mem, "cn-wlcb-01")
        raw[f"cpu{cpu}_mem{mem}"] = r
        if p is None:
            print(f"{cpu:>4}{mem:>8}      -- RetCode={r.get('RetCode')} "
                  f"{r.get('Message', '')[:40]}")
            continue
        if (cpu, mem) == (BASE_CPU, BASE_MEM_GB):
            base_price = p["price"]
        grid.append({"cpu": cpu, "mem_gb": mem, **p})
        tag = "  <- 基准(官方示例配比)" if (cpu, mem) == (BASE_CPU, BASE_MEM_GB) else ""
        print(f"{cpu:>4}{mem:>8}{p['price']:>10}{p['orig']:>10}{tag}")

    if not grid:
        print("\n没有任何配比查价成功，无法继续。")
        return 1

    cheapest = min(grid, key=lambda g: g["price"])

    print()
    print("按价格升序：")
    print(f"{'CPU':>4}{'内存GB':>8}{'元/时':>10}{'相对基准':>12}")
    print("-" * 40)
    for g in sorted(grid, key=lambda x: x["price"]):
        if base_price:
            g["vs_base_pct"] = (g["price"] - base_price) / base_price * 100
            print(f"{g['cpu']:>4}{g['mem_gb']:>8}{g['price']:>10}"
                  f"{g['vs_base_pct']:>+11.1f}%")
        else:
            print(f"{g['cpu']:>4}{g['mem_gb']:>8}{g['price']:>10}")

    print()
    print("=" * 80)
    print(f"[M2] 扫地域（配比 {cheapest['cpu']}C/{cheapest['mem_gb']}G）")
    print("-" * 80)
    zone_rows = []
    for z in ZONES:
        p, r = price(client, cheapest["cpu"], cheapest["mem_gb"], z)
        raw[f"zone_{z}"] = r
        if p is None:
            print(f"  {z:<14} 不可用  RetCode={r.get('RetCode')} "
                  f"{r.get('Message', '')[:44]}")
            continue
        zone_rows.append({"zone": z, **p})
        print(f"  {z:<14} {p['price']:>8} 元/时   (原价 {p['orig']})")

    print()
    print("=" * 80)
    print("[M3] 计费方式")
    print("-" * 80)
    charge_rows = []
    for ct in ("Dynamic", "Month", "Year"):
        p, r = price(client, cheapest["cpu"], cheapest["mem_gb"],
                     "cn-wlcb-01", charge=ct)
        raw[f"charge_{ct}"] = r
        if p is None:
            print(f"  {ct:<10} 不可用 RetCode={r.get('RetCode')} "
                  f"{r.get('Message', '')[:44]}")
            continue
        charge_rows.append({"charge": ct, **p})
        print(f"  {ct:<10} {p['price']:>10}  (原价 {p['orig']})")

    best_zone = min(zone_rows, key=lambda z: z["price"]) if zone_rows else None

    print()
    print("=" * 80)
    print("结论")
    print("-" * 80)
    print(f"最省配比：{cheapest['cpu']}C / {cheapest['mem_gb']}G = "
          f"{cheapest['price']} 元/时")
    if base_price:
        save = (base_price - cheapest["price"]) / base_price * 100
        print(f"基准配比：{BASE_CPU}C / {BASE_MEM_GB}G = {base_price} 元/时")
        print(f"降幅：{save:.1f}%")
        print("  -> 配比对价格影响很小，基本是 5090 地板价。" if save < 5
              else f"  -> 值得改配比，省 {save:.1f}%。")
    if best_zone:
        print(f"最便宜地域：{best_zone['zone']} = {best_zone['price']} 元/时")

    pr = (best_zone or cheapest)["price"]
    print()
    print(f"按最低价 {pr} 元/时估算（加速比未实测，给区间）：")
    print(f"{'加速比':>8}{'时/run':>10}{'元/run':>10}{'9run总价':>12}")
    print("-" * 42)
    for sp in (1.0, 1.5, 2.0, 2.4):
        h = LOCAL_EPOCH_SEC / sp * EPOCHS / 3600
        print(f"{sp:>8.1f}{h:>10.2f}{h * pr:>10.1f}{h * pr * N_RUNS:>12.0f}")
    print("\n⚠️ 加速比 1.0~2.4 为估算区间，必须云端实跑校准后才能定论。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "price_5090_cheapest.json").write_text(json.dumps(
        {"grid": grid, "zones": zone_rows, "charges": charge_rows,
         "base_price_16c64g": base_price, "cheapest": cheapest,
         "best_zone": best_zone}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "price_5090_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存档：{OUT / 'price_5090_cheapest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
