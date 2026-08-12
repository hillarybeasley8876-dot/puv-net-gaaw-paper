# -*- coding: utf-8 -*-
"""[一次性诊断脚本] 逐键 diff 两份统计产物, 确认 ch3_stats.py 改造未污染既有数字。"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, p + "/" + str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, p + "/[%d]" % i))
    else:
        out[p] = o
    return out

a = flat(json.load(open(sys.argv[1], encoding="utf-8")))
b = flat(json.load(open(sys.argv[2], encoding="utf-8")))
ka, kb = set(a), set(b)
for k in sorted(ka - kb):
    print("ONLY_OLD %s = %r" % (k, a[k]))
for k in sorted(kb - ka):
    print("ONLY_NEW %s = %r" % (k, b[k]))
n = 0
for k in sorted(ka & kb):
    if a[k] != b[k]:
        print("CHANGED  %s: %r -> %r" % (k, a[k], b[k]))
        n += 1
print("共有键 %d, 数值变化 %d" % (len(ka & kb), n))
