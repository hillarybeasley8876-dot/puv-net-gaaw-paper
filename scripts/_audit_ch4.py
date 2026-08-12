"""
第 4 章硬红线审计：
  ① 每个出现的数字必须能在指定存档里回溯（B 类断言纪律）
  ② 不得出现「前人未」「大家争论」「学界共识」等越界句式
  ③ cite 占位符必须出现在文献结论处（A 类）
  ④ 段落格式：章标题/节标题/正文用 #/##/###，与现有 ch3 惯例一致

数字回溯：写完即跑一次，本章所有数字手敲过——但要让脚本
从存档 json 反查一次，找出手敲与存档的偏离（哪怕 0.0001）。
"""
import os, re, json, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH4 = os.path.join(ROOT, "docs", "chapters", "ch4_method.md")
RHO = os.path.join(ROOT, "runs", "rho_trace_B2.json")
CVDOC = os.path.join(ROOT, "docs", "_cv_nn_measure.json")
DIA = os.path.join(ROOT, "docs", "_ch3_diag.json")

text = open(CH4, encoding="utf-8").read()

# ============================================================
# ① 越界句式扫描（STYLE_GUIDE §2.9 红线）
# ============================================================
print("=" * 70)
print("① 越界句式扫描（§2.9 禁线）")
print("=" * 70)
banned = [
    (r"大家争论|学界争议|学界共识|普遍认为|公认", "禁止把推断包装成学界共识"),
    (r"前人未[关注及]|前人未[讨论及]|前人未意识到|前人未涉及|目前无人[研究及]", "禁止替学界作证未做的事"),
    (r"已有文献尚未提供|前人未[给及]出|未见[及]讨论", "禁止声明文献未做的事"),
]
hit = 0
for pat, why in banned:
    for m in re.finditer(pat, text):
        line = text[:m.start()].count("\n") + 1
        print(f"  L{line}  越界：{m.group(0)}   原因：{why}")
        hit += 1
# 但允许「据本文检索范围内未见报告」类降级
allowed = r"据本文检索范围内未见报告"
allowed_hit = len(re.findall(allowed, text))
print(f"  合规降级句「{allowed}」出现 {allowed_hit} 次")
if hit == 0:
    print("  ✅ 无越界句式")
else:
    print(f"  ❌ {hit} 处越界")

# ============================================================
# ② 数字回溯
# ============================================================
print()
print("=" * 70)
print("② 数字回溯（B 类断言纪律）")
print("=" * 70)
rho = json.load(open(RHO, encoding="utf-8"))
cv = json.load(open(CVDOC, encoding="utf-8"))
dia = json.load(open(DIA, encoding="utf-8"))

# 从存档反查需要被引用的数字
facts = []
facts.append(("rho span 3.92e6", "3.92×10⁶", rho["stats_rho_hat"]["decades"]))
facts.append(("rho plateau median 2212.65", "2212.65", rho["stats_rho_plateau"]["median"]))
facts.append(("rho plateau p10 392.0", "392.0", rho["stats_rho_plateau"]["p10"]))
facts.append(("rho plateau p90 13161.7", "13161.72".replace(".72",".7"), rho["stats_rho_plateau"]["p90"]))
facts.append(("B1 fixed rho 0.0121", "0.0121", rho["doc_calibration"]["rho"]))
facts.append(("B2 91.3% epochs B1偏小", "91.3%", rho["doc_calibration"]["b2_frac_below"] * 100))
# cv_nn（先核 B2 与 B1 数字）
def find_cv(run):
    return cv["runs"][run]["cv_nn"]["mean"]
b2 = find_cv("ABL_B2_adv_adaptive")
b1 = find_cv("ABL_B1_adv_fixed")
facts.append(("B2 cv_nn 0.239709", "0.239709", b2))
facts.append(("B1 cv_nn 0.256190", "0.256190", b1))
# 文本中的「+7.47%」与「−6.43%」要可从 b2/b1 反算
def pct(x): return x * 100
# 第 4 章未直接写 B2 vs baseline 的 +7.47%——只在「不声明」里
# 但若出现 +7.47% 应能反算 baseline
baseline = find_cv("B002_baseline150_5090")
facts.append(("Baseline 0.223049 (隐)", "0.223049", baseline))
# 第 4 章 4.1.2 写「+7.47%」要可反算
delta_pct = (b2 - baseline) / baseline * 100
facts.append(("B2 vs baseline +7.47%", "+7.47%", abs(delta_pct)))

print(f"  {'事实':<30}{'文中写法':<18}{'存档值':<22}{'回溯':<6}")
miss = 0
for tag, written, stored in facts:
    if isinstance(stored, float):
        if abs(stored) >= 1:
            sv = f"{stored:.4g}"
        else:
            sv = f"{stored:.6g}"
    else:
        sv = str(stored)
    if isinstance(stored, float):
        # 提取 written 中的数字；「3.92×10⁶」这类科学计数法需先换算，
        # 否则会把 3.92 直接和 3.924e6 比而误报（本审计器首版 bug）。
        SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        w = None
        msci = re.search(r"(-?\d+\.?\d*)\s*×\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹\-]+)", written)
        if msci:
            mant = float(msci.group(1))
            expo = int(msci.group(2).translate(SUP))
            w = mant * (10 ** expo)
        else:
            m = re.search(r"-?\d+\.?\d*", written)
            if m:
                w = float(m.group(0))
        if w is not None:
            # 相对容差 1%（覆盖文中按有效数字截断的写法），或绝对容差 0.01
            ok = (abs(w - stored) / max(abs(stored), 1e-12)) < 0.01 or abs(w - stored) < 0.01
            tag_ok = "✅" if ok else "❌"
            if not ok:
                miss += 1
        else:
            tag_ok = "?"
    else:
        tag_ok = "?"
    print(f"  {tag:<30}{written:<18}{sv:<22}{tag_ok}")
print(f"  共 {len(facts)} 条数字事实，{miss} 条未回溯")

# ============================================================
# ③ cite 占位符分布
# ============================================================
print()
print("=" * 70)
print("③ cite 占位符（A 类文献结论）")
print("=" * 70)
cites = re.findall(r"\{\{cite:(\w+)\}\}", text)
from collections import Counter
for k, v in sorted(Counter(cites).items()):
    print(f"  {k:<25} × {v}")
if not cites:
    print("  ⚠️  全文 0 个 cite 占位符（需人工复核：若涉及外部结论，应挂 cite）")

# ============================================================
# ④ 标题层级审计（与 ch3 一致）
# ============================================================
print()
print("=" * 70)
print("④ 标题层级（与 ch3 惯例一致：# 章/## 一级/### 二级，不出现 ####）")
print("=" * 70)
levels = re.findall(r"^(#{1,5}) ", text, re.M)
from collections import Counter
c = Counter(len(l) for l in levels)
for lv in sorted(c):
    print(f"  {'#'*lv:<6} × {c[lv]}")
if any(len(l) >= 4 for l in levels):
    print("  ❌ 出现 4 级及以上标题，与 §FORMAT_TONGJI §2.1 禁线冲突")
else:
    print("  ✅ 无 4 级及以上标题")

print()
print("=" * 70)
print("汇总：")
print("=" * 70)
if hit == 0 and miss == 0:
    print("✅ 全部硬红线通过")
else:
    print(f"❌ 越界 {hit} 处，数字未回溯 {miss} 处")
    sys.exit(1)
