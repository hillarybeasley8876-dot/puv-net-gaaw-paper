import json, collections
d=json.load(open('docs/REFERENCES.json',encoding='utf-8'))
refs=d['references']
print('总数', len(refs))
print()
# 1) 按 topic 聚类
byt=collections.Counter(r.get('topic','(none)') for r in refs)
print('=== topic 分布 ===')
for t,c in byt.most_common():
    print('  %-28s %d' % (t,c))
print()
# 2) 按声明的引用章节
ch=collections.Counter()
nocite=[]
for r in refs:
    cs=r.get('cite_in_chapter') or []
    if not cs: nocite.append(r['key'])
    for c in cs: ch[c]+=1
print('=== 声明引用章节分布 ===')
for c,n in sorted(ch.items()):
    print('  %-8s %d' % (c,n))
print('未声明章节的条目:', len(nocite), nocite[:20])
print()
# 3) 打印全部, 便于人工判相关性
print('=== 全表 (number | key | year | topic | title) ===')
for r in sorted(refs,key=lambda x:x['number']):
    print('[%3d] %-20s %s %-24s %s' % (r['number'], r['key'], r.get('year'), (r.get('topic') or '')[:24], (r.get('title') or '')[:70]))
