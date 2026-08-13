# Codex n=11 口径产物（已作废，仅存档备查）

> **红线：本目录下任何数字均不得进入论文正文、摘要、图表或答辩材料。**
> 需要数字时，唯一入口是 `docs/_thesis_results.json`（n=200）。

## 这是什么

2026-08-13 远端 commit `ceb3b1a`「Add auditable full thesis manuscript and
deliverables」带入的一整套稿件与审计产物，共 24 项（含 1160 行的
`THESIS_MANUSCRIPT.md`、并行章节 ch3–ch7、公式/插图资产、引用与实验审计报告）。
作者署名 `CEO <ceo@nightledger.game>`，即 Codex 使用的同一 git config。

## 为什么作废

**该稿全文以 n=11 的「真实验证交集」为评测口径**，而 n=11 口径已在本项目
被裁定作废（三条硬证据，见 `docs/EXPERIMENT_LOG.md` 与
`scripts/_adjudicate_n11_claim.py`、`scripts/_negtest_adjudicate_n11.py`）。

实测口径统计（`scripts/_tmp/probe_codex_basis.py`）：

| 文件 | n=11 出现 | n=200 出现 |
|---|---|---|
| `THESIS_MANUSCRIPT.md` | 49 | 1 |
| `ch6_results.md`（核心结果章） | 21 | 0 |
| `ch3_analysis_framework.md` | 11 | 1 |
| `ch7_conclusion.md` | 5 | 0 |
| `ch4_research_design.md` | 4 | 0 |
| `frontmatter/abstract_zh.md` | 3 | 0 |
| `EXPERIMENT_AUDIT.md` | 7 | 0 |
| `NARRATIVE_REPORT.md` | 7 | 0 |

其核心结果表 6.4 表题即写明「真实验证交集n=11，单训练seed」。
n=11 与本文 n=200 存档口径不可换算、不可并列、不可互相引用。

## 为什么不直接删

用户裁定「保留证据链但不污染 `docs/`」。保留理由：
1. 该稿是「Codex 曾以 n=11 出过全文」这一事实的物证，日后若出现口径争议需要能回溯；
2. 其中的**非数字资产**（公式 SVG/PNG、部分示意图、章节骨架、索引脚本思路）
   在重新核对后**可能**可复用——但复用前必须逐项验证，且数字一律重新从
   `_thesis_results.json` 取。

## 正式稿在哪

- 章节正稿：`docs/chapters/ch1_introduction.md` … `ch5_experiments.md`
- 唯一数字入口：`docs/_thesis_results.json`
- 配对分析：`docs/_paired_improvement_B2_vs_B1.json`
- 数字审计器：`scripts/audit_thesis_numbers.py`
  （配 `scripts/_negtest_ch5_tamper.py` 篡改表 11/11 + `_negtest_audit_numbers.py` 8/8）

## 注意：本目录不受数字审计器保护

`audit_thesis_numbers.py` 只扫 `docs/chapters/` 下的正稿。本目录内的数字
既未回溯存档、也未经篡改表验证，**不要因为「它看起来像审计报告」就信任它**。
