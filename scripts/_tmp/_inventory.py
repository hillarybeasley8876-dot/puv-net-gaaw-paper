"""只读盘点：现有 run 的 epoch 数 + cv_nn 表里已有哪些 run。"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs")

print("=" * 70)
print("runs/ 目录下 history.json 长度")
print("=" * 70)
for d in sorted(os.listdir(RUNS)):
    h = os.path.join(RUNS, d, "history.json")
    if not os.path.exists(h):
        continue
    try:
        n = len(json.load(open(h, encoding="utf-8")))
    except Exception as e:
        n = "ERR:" + type(e).__name__
    print("  {:36s} {}".format(d, n))

cvp = os.path.join(ROOT, "docs", "_cv_nn_measure.json")
print()
print("=" * 70)
print("docs/_cv_nn_measure.json 已测 run")
print("=" * 70)
if os.path.exists(cvp):
    d = json.load(open(cvp, encoding="utf-8"))
    print("  meta: seed={} n_sample={} val_ratio={}".format(
        d.get("seed"), d.get("n_sample"), d.get("val_ratio")))
    for k, v in d.get("runs", {}).items():
        print("  {:36s} gpu={:6s} cv_nn={:.6f}".format(
            k, str(v.get("gpu")), v["cv_nn"]["mean"]))
else:
    print("  (不存在)")
