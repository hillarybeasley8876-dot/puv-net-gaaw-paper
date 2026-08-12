# -*- coding: utf-8 -*-
"""B-002 进度巡检 —— 供 cron 复用。

键名以真实产物核实为准（2026-08-11）：
  metrics.json   : dict, 取 ['records'] -> list[dict] 每 epoch 一条
  selection.json : dict, ['summary']['best_epoch'] / ['shadow_cd_only']['best_cd_epoch']
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "B002_baseline150"
TARGET_EPOCHS = 150


def main() -> int:
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))["records"]
    s = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    last = m[-1]
    n = len(m)
    summ = s.get("summary", {})
    shadow = s.get("shadow_cd_only", {})
    best_ep = summ.get("best_epoch")
    best_cd_ep = shadow.get("best_cd_epoch")
    diverge = (best_ep is not None and best_cd_ep is not None
               and best_ep != best_cd_ep)

    err = RUN.parent / "b002_stderr.log"
    err_bytes = os.path.getsize(err) if err.exists() else -1

    print("epochs        : %d / %d" % (n, TARGET_EPOCHS))
    print("latest ep%-3d  : cd %.6f  hd %.6f  nuc %.6f"
          % (last["epoch"], last["monitor_cd"], last["monitor_hd"],
             last["monitor_nuc"]))
    print("select best   : ep%s (score %.6f)"
          % (best_ep, summ.get("best_score", float("nan"))))
    print("shadow cd-only: ep%s (cd %.6f)"
          % (best_cd_ep, shadow.get("best_cd", float("nan"))))
    print("diverge       : %s%s" % (diverge, "  <-- 两准则分歧 ★" if diverge else ""))
    print("stderr bytes  : %d%s" % (err_bytes,
                                    "  <-- 非空，需查！" if err_bytes > 0 else ""))
    # sec/epoch 只在 history.json 里，metrics.json 的 records 没有该键
    hist_f = RUN / "history.json"
    sec = None
    if hist_f.exists():
        h = json.loads(hist_f.read_text(encoding="utf-8"))
        if h and isinstance(h, list):
            sec = h[-1].get("sec")
    if sec:
        print("sec/epoch     : %.1f  (from history.json)" % sec)
        if n < TARGET_EPOCHS:
            print("ETA           : %.2f h" % ((TARGET_EPOCHS - n) * sec / 3600.0))
    else:
        print("sec/epoch     : n/a")
    print("DONE          : %s" % (n >= TARGET_EPOCHS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
