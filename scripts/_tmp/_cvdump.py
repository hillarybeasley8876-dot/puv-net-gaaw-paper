import json, pathlib
p = pathlib.Path(r"E:\AE-CC托管\puv-net\docs\_cv_nn_measure.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("keys:", list(d.keys())[:20])
runs = d.get("runs", d)
if isinstance(runs, dict):
    for k, v in runs.items():
        if isinstance(v, dict):
            print(f"{k:34s} cv={v.get('cv_nn_mean', v.get('mean'))} se={v.get('cv_nn_se', v.get('se'))} q={v.get('q4_q1')} gpu={v.get('gpu')}")
elif isinstance(runs, list):
    for v in runs:
        print(f"{v.get('run',''):34s} cv={v.get('cv_nn_mean')} se={v.get('cv_nn_se')} q={v.get('q4_q1')} gpu={v.get('gpu')}")
