import json, pathlib
p = pathlib.Path(r"E:\AE-CC托管\puv-net\docs\REFERENCES.json")
refs = json.loads(p.read_text(encoding="utf-8"))["references"]
print("n =", len(refs))
print("field keys of [0]:", list(refs[0].keys()))
print("=" * 70)
NEED = ["pu-gan", "pu gan", "pu-gcn", "pu gcn", "dis-pu", "pu-transformer", "gradnorm", "uncertainty weighting"]
for i, r in enumerate(refs):
    blob = json.dumps(r, ensure_ascii=False).lower()
    if any(n in blob for n in NEED):
        print(f"[{i}] key={r.get('key')!r}")
        print(f"     number={r.get('number')}  title={str(r.get('title'))[:90]}")
        print(f"     venue={r.get('venue')} year={r.get('year')}")
