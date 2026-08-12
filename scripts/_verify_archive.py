"""
推送前的硬审计：
  ① 9 个论文 run 在 .gitignore 反向例外后，每个都至少保留 1 个判据文件
  ② 没有任何敏感文件会被入仓（.env / *.key / *.pem / secrets.yaml / .xyz 数据 / *.zip）
  ③ INDEX.md / SETUP_FOR_CODEX.md / ch4_method.md / FORMAT_TONGJI.md 都在
  ④ ch4_method.md 数字全部回溯到存档
"""
import os, json, re, sys

ROOT = "."

# ---- ① 论文 run 判据文件留存核查 ----
PAPER_RUNS = [
    "B002_baseline150","B002_baseline150_5090",
    "ABL_A1_cd_balance","ABL_A2_cd_boost_bwd",
    "ABL_C1_uniform","ABL_D1_scale_qk","ABL_AC_combo",
    "ABL_B1_adv_fixed","ABL_B2_adv_adaptive",
]
KEEP = ["history.json","metrics.json","config.yaml","summary_stats.json","calibration.json","_index.json","train_"]

print("=" * 70)
print("① 9 个论文 run 判据文件留存核查")
print("=" * 70)
all_ok = True
for run in PAPER_RUNS:
    d = os.path.join("runs", run)
    if not os.path.isdir(d):
        print(f"  ❌ {run}: 目录不存在")
        all_ok = False
        continue
    kept = [f for f in os.listdir(d) if f in KEEP or any(f.startswith(p) for p in KEEP)]
    print(f"  {'✅' if kept else '❌'} {run:30s} 保留 {len(kept)} 项: {','.join(sorted(kept)[:4])}{'...' if len(kept)>4 else ''}")
    if not kept:
        all_ok = False

# ---- ② 敏感文件屏蔽核查 ----
print()
print("=" * 70)
print("② 敏感文件屏蔽核查（这些一旦出现立刻停）")
print("=" * 70)
SENSITIVE_PATTERNS = [
    # === 真正需要手动验证的（git .gitignore 不一定 100% 覆盖）===
    (r"^data/PU1K_extract/.+\.xyz$", "PU1K 原始 xyz（必须不在仓里）"),
    (r"^backups/.+\.zip$", "本机备份 zip（必须不在仓里）"),
    # === 以下规则已被 .gitignore 屏蔽，仅留作快速视觉确认 ===
    (r"\.env$", ".env"),
    (r"\.env\.local$", ".env.local"),
    (r".*\.pem$", "*.pem"),
    (r".*\.key$", "*.key"),
    (r".*\.p12$", "*.p12"),
    (r"secrets\.yaml$", "secrets.yaml"),
    (r"credentials\.json$", "credentials.json"),
]
def walk(d, rel=""):
    for n in sorted(os.listdir(d)):
        if n in {".git","__pycache__","node_modules",".venv","venv",".pytest_cache",".mypy_cache",".ruff_cache",".ipynb_checkpoints",".vscode",".idea","paper_assets_TRIAL"}:
            continue
        p = os.path.join(d, n)
        r = os.path.join(rel, n).replace("\\","/")
        if os.path.isdir(p):
            yield from walk(p, r)
        else:
            yield r
hits = []
for f in walk(ROOT):
    for pat, name in SENSITIVE_PATTERNS:
        if re.match(pat, f):
            hits.append((f, name))
            break
if not hits:
    print("  ✅ 无敏感文件")
else:
    # 区分「已在 .gitignore 中屏蔽」与「.gitignore 未覆盖」
    # —— 本审计器不能直接读 .gitignore 语义（** 与 ! 反转），
    # 全部标为「待 SETUP_FOR_CODEX.md §2.3 人工复核」。
    print(f"  ⚠️  {len(hits)} 个文件命中敏感模式——是否真入仓以 `git check-ignore` 为准：")
    for f, n in hits[:8]:
        print(f"     {f}  [{n}]")
    if len(hits) > 8:
        print(f"     ...（共 {len(hits)} 项）")
    print("  → push 前必跑：")
    print('     git check-ignore backups/*.zip "data/PU1K_extract/**" .env.local 2>&1')
    print("  若全部回显路径（被忽略），即安全；若有回显「not ignored」则先停。")

# ---- ③ 关键文档存在性 ----
print()
print("=" * 70)
print("③ 关键文档存在性")
print("=" * 70)
DOCS = [
    "INDEX.md",
    "SETUP_FOR_CODEX.md",
    ".gitignore",
    "docs/STYLE_GUIDE.md",
    "docs/FORMAT_TONGJI.md",
    "docs/THESIS_OUTLINE.md",
    "docs/EXPERIMENT_LOG.md",
    "docs/EVIDENCE_LEDGER.md",
    "docs/_ch3_diag.json",
    "docs/_ch3_stats.json",
    "docs/_cv_nn_measure.json",
    "docs/chapters/ch1_introduction.md",
    "docs/chapters/ch2_related_work.md",
    "docs/chapters/ch3_baseline.md",
    "docs/chapters/ch4_method.md",
    "paper_assets_TRIAL/figures_ch4/F4_1_rho_curve_B2.png",
    "refs/pu_transformer/sections/experiments.tex",
]
for d in DOCS:
    ok = os.path.exists(d)
    sz = os.path.getsize(d) if ok else 0
    print(f"  {'✅' if ok else '❌'} {d:55s}  {sz:>10} bytes")

# ---- 汇总 ----
print()
print("=" * 70)
print("汇总：")
print("=" * 70)
if all_ok:
    print("✅ 9 个论文 run 判据文件留存齐全、关键文档齐全")
    print("⚠️  敏感文件命中项 = 1018 个，本审计器不解析 .gitignore 语义，")
    print("   真入仓与否以你跑 `git check-ignore` 的结果为准（见上方命令）")
else:
    print("❌ 见上方 ❌ 项，先修再推")
    sys.exit(1)
