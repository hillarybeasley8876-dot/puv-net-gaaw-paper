"""探查 B 系 run 的 history.json 是否含 adv_w_adaptive 逐 epoch 记录。

若含 train_adv_w_adaptive，则 rho = target_ratio / w_auto 可逐 epoch 反解，
无需补跑短 run。target_ratio 从该 run 的 config.yaml 读，不硬编码。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ["ABL_B1_adv_fixed", "ABL_B2_adv_adaptive", "B002_baseline150_5090"]

for r in RUNS:
    d = ROOT / "runs" / r
    hp = d / "history.json"
    print("=" * 70)
    print(r, "exists" if hp.exists() else "MISSING history.json")
    if not hp.exists():
        continue
    h = json.loads(hp.read_text(encoding="utf-8"))
    print("  epochs:", len(h))
    keys = sorted(h[0].keys())
    print("  fields:", len(keys))
    for k in keys:
        print("    ", k)
    hits = [k for k in keys if "adapt" in k or "adv" in k]
    print("  adv/adapt hits:", hits)
    for k in hits:
        vals = [rec.get(k) for rec in h]
        num = [v for v in vals if isinstance(v, (int, float))]
        if num:
            print(f"    {k}: n={len(num)} first={num[0]!r} last={num[-1]!r} "
                  f"min={min(num)!r} max={max(num)!r}")
        else:
            print(f"    {k}: 非数值样本 {vals[:2]!r}")
    # notes 留痕（adv_adaptive_error 会落在这里）
    notes = [rec.get("notes") for rec in h if rec.get("notes")]
    print("  notes 记录数:", len(notes), notes[:2] if notes else "")
    # target_ratio 溯源
    cp = d / "config.yaml"
    if cp.exists():
        txt = cp.read_text(encoding="utf-8")
        for line in txt.splitlines():
            ls = line.strip()
            if ("target_ratio" in ls or "adaptive_adv" in ls
                    or ls.startswith("w_adv")):
                print("  cfg:", ls)
    else:
        print("  cfg: MISSING config.yaml")
print("=" * 70)
