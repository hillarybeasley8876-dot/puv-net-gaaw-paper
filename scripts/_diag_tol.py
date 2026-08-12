# [一次性诊断脚本] 用于校准 selfcheck_ch3.py 的溯源容差与池子密度; 结论已写入
# selfcheck_ch3.py 的 REL_TOL / DETAIL_KEYS 注释, 日志见 docs/_diag_*.log。保留以便复核。
# -*- coding: utf-8 -*-
"""诊断: 存档数值池对任意数字的"误匹配率"，用于校准溯源检查的容差与候选变换。"""
import io
import json
import random
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def flat(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            flat(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flat(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))


arch = []
for a in ["docs/_ch3_stats.json", "docs/_ch3_shapes.json", "docs/_ch3_diag.json"]:
    flat(json.load(open(a, encoding="utf-8")), arch)
print("pool = %d" % len(arch))


def hits(t, tol, scales):
    r = []
    for c in [t * s for s in scales]:
        for x in arch:
            if x != 0 and abs(c - x) <= tol * abs(x):
                r.append(x)
    return r


for tol in (0.011, 0.002, 0.0005):
    for scales in ([1.0, 0.01, 100.0], [1.0]):
        # 随机 3 位小数, 模拟"被篡改的正文数字"
        random.seed(7)
        probe = [round(random.uniform(0.001, 5.0), 3) for _ in range(2000)]
        fp = sum(1 for p in probe if hits(p, tol, scales))
        print("tol=%-7g scales=%-18s 随机数误匹配率 %5.1f%%  (0.777 -> %d 命中)"
              % (tol, scales, 100.0 * fp / len(probe), len(hits(0.777, tol, scales))))
