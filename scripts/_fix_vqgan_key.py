# -*- coding: utf-8 -*-
"""
统一 cite key：{{cite:VQGAN}} -> {{cite:TamingTransformers}}

依据：docs/REFERENCES.json 中该文献的 key 为 TamingTransformers（number 116），
      title = "Taming Transformers for High-Resolution Image Synthesis"
      authors = Esser, Rombach, Ommer；venue = CVPR 2021
      note 字段已明确其为本文 GAAW 定位必须披露的直接先例。
      库中不存在 key = VQGAN 的条目，故正文的 VQGAN 写法无法替换编号。

处理方式：正文改 key，不改库。理由是库的 key 命名与 verified 记录已固化，
          且 cite_in_chapter 字段已标注 5.1 / 5.5，与本文第 5 章一致。

安全设计：替换后校验 (a) 正文不再有 VQGAN key；(b) TamingTransformers 的
          出现次数等于替换前 VQGAN 次数加原有次数。
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(ROOT, "docs", "chapters")
FILES = ["ch1_introduction.md", "ch2_related_work.md", "ch3_baseline.md",
         "ch4_design.md", "ch5_mechanism.md", "ch6_experiments.md", "ch7_conclusion.md"]

OLD, NEW = "{{cite:VQGAN}}", "{{cite:TamingTransformers}}"

print("=" * 72)
print("统一 cite key: VQGAN -> TamingTransformers")
print("=" * 72)

before_old = before_new = 0
changed = []
for f in FILES:
    p = os.path.join(CH, f)
    t = io.open(p, encoding="utf-8").read()
    n_old = t.count(OLD)
    n_new = t.count(NEW)
    before_old += n_old
    before_new += n_new
    if n_old:
        t = t.replace(OLD, NEW)
        io.open(p, "w", encoding="utf-8").write(t)
        changed.append((f, n_old))
        print(f"  FIX  {f:26s} 替换 {n_old} 处")
    else:
        print(f"  --   {f:26s} 无 VQGAN key")

print()
print(f"  替换前: VQGAN {before_old} 处, TamingTransformers {before_new} 处")

# 校验
after_old = after_new = 0
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    after_old += t.count(OLD)
    after_new += t.count(NEW)

print(f"  替换后: VQGAN {after_old} 处, TamingTransformers {after_new} 处")
print()
ok = (after_old == 0) and (after_new == before_old + before_new)
if ok:
    print("  PASS  VQGAN 已清零，且总数守恒")
else:
    print("  FAIL  校验不通过")
    sys.exit(1)

# 顺带检查正文里是否还有"VQGAN"作为普通词出现（正文提及该模型名是允许的，
# 只要不是 cite key）——报告数量供人工确认，不作 FAIL。
print()
print("  正文中作为普通词出现的 'VQGAN'（非 cite key，允许）：")
for f in FILES:
    t = io.open(os.path.join(CH, f), encoding="utf-8").read()
    n = len(re.findall(r"VQGAN", t))
    if n:
        print(f"     {f:26s} {n} 处")
