import io, json, os, sys, urllib.request
base = "https://raw.githubusercontent.com/guochengqian/PU-GCN/master/"
files = ["evaluate.py", "Common/metrics.py",
         "tf_ops/nn_distance/tf_nndistance_cpu.py",
         "evaluation_code/evaluation.cpp", "Common/ops.py"]
out = os.path.join("refs", "pu_gcn", "snapshot")
os.makedirs(out, exist_ok=True)
manifest = []
for f in files:
    try:
        with urllib.request.urlopen(base + f, timeout=60) as r:
            b = r.read()
    except Exception as e:
        print("FAIL", f, e); continue
    dst = os.path.join(out, f.replace("/", "__"))
    with open(dst, "wb") as fh:
        fh.write(b)
    import hashlib
    manifest.append({"path": f, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    print("OK", f, len(b))
with io.open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as fh:
    json.dump({"repo": "guochengqian/PU-GCN", "branch": "master",
               "tree_sha": "0a8daac57dbda037689d797c679d3540ee253317",
               "fetched": "2026-08-11", "files": manifest}, fh, ensure_ascii=False, indent=2)
print("manifest written")
