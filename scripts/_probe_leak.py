import json
d=json.load(open('docs/REFERENCES.json',encoding='utf-8'))
refs={r['key']:r for r in d['references']}
rej=[]
for fn in ['docs/REF_REJECTED_B1.json','docs/REF_REJECTED_B2.json']:
    j=json.load(open(fn,encoding='utf-8'))
    for r in j['rejected']:
        rej.append((fn.split('/')[-1], r['key'], r['reason'], r['input'].get('title',''), r['input'].get('arxiv',''), r['input'].get('year','')))
print('=== 全部被拒记录 ===')
for f,k,rs,t,a,y in rej:
    print('%-22s %-18s %-22s arxiv=%-12s y=%s' % (f,k,rs,a,y))
    print('   input.title:',t[:90])
print()
print('=== 同名 key 在最终库里的真实记录 ===')
for f,k,rs,t,a,y in rej:
    if k in refs:
        r=refs[k]
        v=r.get('verified',{})
        print('[%d] %-18s reason_was=%s' % (r['number'],k,rs))
        print('    title :',r.get('title','')[:100])
        print('    year  :',r.get('year'),' arxiv:',r.get('arxiv') or r.get('arxiv_id'),' doi:',r.get('doi'))
        print('    chan  :',v.get('channels'))
        print('    urls  :',(v.get('verify_urls') or v.get('urls') or [])[:2])
        print()
