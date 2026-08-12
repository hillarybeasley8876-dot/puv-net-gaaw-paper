import json, pathlib
p = pathlib.Path(r"E:\AE-CC托管\puv-net\docs\REFERENCES.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("top type:", type(d).__name__)
if isinstance(d, dict):
    print("top keys:", list(d.keys())[:15])
    # 找容器
    for k, v in d.items():
        print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v,'__len__') else '-'}")
