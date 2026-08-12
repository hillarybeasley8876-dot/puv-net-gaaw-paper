# [一次性诊断脚本] 用于校准 selfcheck_ch3.py 的溯源容差与池子密度; 结论已写入
# selfcheck_ch3.py 的 REL_TOL / DETAIL_KEYS 注释, 日志见 docs/_diag_*.log。保留以便复核。
# -*- coding: utf-8 -*-
"""查 0.777 在池中的匹配来源, 判断是"池子该收窄"还是"注入用例该换值"。"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def walk(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "per_sample":
                continue
            walk(v, path + "/" + str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, "%s[%d]" % (path, i), out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append((float(obj), path))


rows = []
for a in ["docs/_ch3_stats.json", "docs/_ch3_shapes.json", "docs/_ch3_diag.json",
          "runs/E000_metric_calibration/result.json", "runs/D1v2_cd_blindspot/result.json"]:
    walk(json.load(open(a, encoding="utf-8")), a, rows)

for target, tol in ((0.777, 0.005), (0.916, 0.005)):
    print("== target %s tol %.3f ==" % (target, tol))
    for v, p in rows:
        if v != 0 and abs(abs(target) - abs(v)) <= tol * abs(v):
            print("   %-22r  %s" % (v, p))
