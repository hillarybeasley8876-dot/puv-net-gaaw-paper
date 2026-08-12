import json, os
D='docs'
for fn in ['REFERENCES.json','REF_REJECTED_B1.json','REF_REJECTED_B2.json']:
    p=os.path.join(D,fn)
    d=json.load(open(p,encoding='utf-8'))
    print('==',fn,type(d).__name__)
    if isinstance(d,dict):
        for k,v in d.items():
            print('   ',k,'->',type(v).__name__, (len(v) if isinstance(v,(list,dict)) else repr(v)[:80]))
        # 找容器
        for k,v in d.items():
            if isinstance(v,(list,dict)) and len(v)>3:
                first = v[0] if isinstance(v,list) else list(v.values())[0]
                print('    容器',k,'首元素键:',list(first.keys()) if isinstance(first,dict) else type(first).__name__)
                if isinstance(first,dict):
                    print('    首元素:',json.dumps(first,ensure_ascii=False)[:600])
                break
