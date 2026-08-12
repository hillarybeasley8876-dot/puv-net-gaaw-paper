import json, pathlib, sys
R = pathlib.Path(r"E:\AE-CC托管\puv-net\runs")
for d in sorted(R.iterdir()):
    if not d.is_dir():
        continue
    h = d / "history.json"
    if not h.exists():
        continue
    try:
        j = json.loads(h.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"{d.name:34s} ERR {e}")
        continue
    recs = j.get("records", j) if isinstance(j, dict) else j
    n = len(recs) if isinstance(recs, list) else -1
    last = recs[-1] if isinstance(recs, list) and recs else {}
    cd = last.get("monitor_cd", last.get("cd", "?"))
    print(f"{d.name:34s} ep={n:4d}  last_cd={cd}")
