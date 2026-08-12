# -*- coding: utf-8 -*-
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "docs", "REFERENCES.json"), encoding="utf-8"))
for n in [4, 8, 12, 14, 18, 24, 36, 37, 38, 43, 48, 51, 56, 57, 59, 60, 66, 67, 68, 69, 70, 88, 90, 91]:
    r = next((x for x in d["references"] if x["number"] == n), None)
    if r:
        print("[%d] %s (%s) - %s" % (n, r["title"][:60], r.get("year"), r.get("key")))
    else:
        print("[%d] NOT IN DB" % n)
