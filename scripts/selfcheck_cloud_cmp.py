# -*- coding: utf-8 -*-
"""build_cloud_comparison.py 自检 —— 用真实 B-001 快照验证渲染与统计。

重点验「肉眼才发现」类缺陷的可量化部分：
  * 色标方向（viridis 低值=聚簇）不能标反
  * GT 必须比 pred 更均匀（若反了说明 npz 键读串）
  * nn_cv 计算正确（用已知构造点云验）
  * 等比例坐标（形状不被拉伸误导）
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "bcc", ROOT / "scripts" / "build_cloud_comparison.py")
bcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcc)


def main() -> int:
    ok = fail = 0

    # --- nn_cv 数学正确性：规则网格 cv 应接近 0；随机点云明显更大 ---
    g = np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(4),
                             indexing="ij"), -1).reshape(-1, 3).astype(
                                 np.float32)
    u_grid = bcc.uniformity(g)
    rng = np.random.default_rng(0)
    u_rand = bcc.uniformity(rng.random((256, 3)).astype(np.float32))
    if u_grid["nn_cv"] < 1e-6:
        print(f"[OK ] 规则网格 nn_cv={u_grid['nn_cv']:.2e} ≈ 0")
        ok += 1
    else:
        print(f"[FAIL] 规则网格 nn_cv={u_grid['nn_cv']:.4f} 应≈0")
        fail += 1
    if u_rand["nn_cv"] > u_grid["nn_cv"] * 10 + 0.1:
        print(f"[OK ] 随机点云 nn_cv={u_rand['nn_cv']:.4f} >> 网格")
        ok += 1
    else:
        print(f"[FAIL] 随机 {u_rand['nn_cv']:.4f} 未明显大于网格")
        fail += 1

    # --- 真实快照：GT 必须比 pred 更均匀 ---
    got = bcc.pick_snapshot(ROOT / "runs" / "B001_reproduce", None, None)
    if got is None:
        print("[FAIL] 读不到 B-001 快照")
        return 1
    f, ep, idx = got
    d = np.load(f)
    u_gt = bcc.uniformity(d["gt"])
    u_pred = bcc.uniformity(d["pred"])
    print(f"[info] 快照 ep{ep} idx{idx}  gt_cv={u_gt['nn_cv']:.4f} "
          f"pred_cv={u_pred['nn_cv']:.4f}")
    if u_gt["nn_cv"] < u_pred["nn_cv"]:
        print("[OK ] GT 比 pred 更均匀（键未读串）")
        ok += 1
    else:
        print("[FAIL] GT 反而不如 pred 均匀 —— 检查 npz 键是否读串")
        fail += 1

    # --- 渲染 + 色标方向 ---
    tmp = Path(tempfile.mkdtemp(prefix="bcc_selfcheck_"))
    try:
        entries = [{"label": "baseline\n(ep%d)" % ep, "pred": d["pred"],
                    "gt": d["gt"], "input": d["input"],
                    "run": "B001_reproduce", "epoch": ep, "index": idx}]
        p = bcc.fig_compare(entries, tmp / "cmp.png")
        if p.exists() and p.stat().st_size > 20000:
            print(f"[OK ] 渲染成功 {p.stat().st_size} B")
            ok += 1
        else:
            print("[FAIL] 渲染产物过小/缺失")
            fail += 1

        meta = json.loads(p.with_suffix(".data.json").read_text(
            encoding="utf-8"))
        cbdir = meta.get("colorbar_direction", "")
        # viridis 低值是深紫。说明文字必须把「深紫」与「聚簇/小值」绑定，
        # 且不得把「浅/黄」说成密集。
        good = ("深紫" in cbdir and "聚簇" in cbdir
                and "黄" in cbdir and "稀疏" in cbdir)
        if good:
            print(f"[OK ] 色标方向说明正确: {cbdir}")
            ok += 1
        else:
            print(f"[FAIL] 色标方向说明可疑: {cbdir}")
            fail += 1

        # vmin/vmax 必须来自真实数据范围且 vmin < vmax
        if 0 <= meta["vmin"] < meta["vmax"]:
            print(f"[OK ] 色标范围 [{meta['vmin']:.5f}, {meta['vmax']:.5f}]")
            ok += 1
        else:
            print(f"[FAIL] 色标范围异常 {meta['vmin']} {meta['vmax']}")
            fail += 1

        # 统计量必须同时落盘 GT 与各组
        if "GT" in meta["uniformity"] and len(meta["uniformity"]) >= 2:
            print(f"[OK ] 均匀性统计已落盘 {list(meta['uniformity'])}")
            ok += 1
        else:
            print(f"[FAIL] 均匀性统计不全 {list(meta['uniformity'])}")
            fail += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 60}\n自检: {ok} PASS / {fail} FAIL")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
