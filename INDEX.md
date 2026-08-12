# puv-net INDEX

> 论文存档入口。本文件给 Codex / 任何接手者用：先看这个文件定位要找什么，然后看对应的源文件。
>
> **纪律（不许破）**：所有数字必须回溯到存档 json（`runs/<name>/history.json` 等），不凭记忆敲。详见 `docs/STYLE_GUIDE.md §2.9`「断言纪律」。

---

## 0. 一句话定位

**GAAW = Gradient-Adaptive Adversarial Weighting**。
在 PU-Transformer + PU-GAN 联合训练框架下，**用对抗分支与重建分支的梯度范数比动态决定对抗权重 `w_adv`**，替代 PU-GAN 经验惯例「`w_adv=8.27` 跨 epoch 固定」。

- 论文题目（暂定）：基于梯度自适应对抗约束的点云上采样方法研究
- 论文 run 集合（9 个，**全部 150ep，全部 cv_nn 补测**）：
  - 同机 3090：基线 `B002_baseline150` + A1/A2/D1/C1/AC_combo（共 6 个）
  - 同机 5090：基线 `B002_baseline150_5090` + B1/B2（共 3 个）
- 第 5 章主对比：**B2 (GAAW) vs B1（照搬 PU-GAN 固定权重）**，同机同 seed
- 主指标 `cv_nn`，同报 CD/HD/NUC，分层 Q4/Q1；门槛 2SE；REJECT_NULL 纪律

---

## 1. 目录速查

| 路径 | 是什么 | 谁该读 |
|---|---|---|
| `docs/THESIS_OUTLINE.md` | 整篇论文大纲（章节结构与关系） | 全局 |
| `docs/STYLE_GUIDE.md` | 写作纪律（**§2.9 断言纪律**，**§2.9.1 研究缺口五段式**） | 所有写章节的人 |
| `docs/FORMAT_TONGJI.md` | 同济模板排版规范（字体/字号/行距/缩进/编号） | 写章节 + 转 Word |
| `docs/EXPERIMENT_LOG.md` | 实验大事记（按时间序，每件事带存档指针） | 写实验节 + 查历史 |
| `docs/EVIDENCE_LEDGER.md` | 证据台账（每条证据对应 run + 数字 + 解释） | 写实验节 |
| `docs/SOTA_SURVEY.md` | 现有方法横向综述 | 写第 2 章相关工作 |
| `docs/REFERENCES_TABLE.md` | 参考文献编号表 | 替换 `{{cite:KEY}}` |
| `docs/_ch3_diag.json` `docs/_ch3_stats.json` `docs/_ch3_shapes.json` `docs/_cv_nn_measure.json` `docs/_rho_*` | 第 3 章基线诊断与第 5 章 cv_nn 测量的所有数字存档 | 任何引用数字的章节 |
| `docs/chapters/ch1_introduction.md` | 第 1 章正稿 | 第 1 章 |
| `docs/chapters/ch2_related_work.md` | 第 2 章正稿 | 第 2 章 |
| `docs/chapters/ch3_baseline.md` | 第 3 章正稿（基线复现+瓶颈诊断） | 第 3 章 / 第 4 章伏笔 |
| `docs/chapters/ch4_method.md` | **第 4 章正稿（GAAW 形式化）** | 第 4 章 |
| `docs/chapters/ch1_introduction.pre_outline_v2.md` | 第 1 章 v2 草稿（**非正稿**） | 查变更历史 |
| `paper_assets_TRIAL/figures_ch4/F4_1_rho_curve_B2.png` | 第 4 章 ρ 曲线图 | 写第 4 章 §4.3.4 |
| `refs/pu_transformer/sections/*.tex` | PU-Transformer LaTeX 原文（已扒） | A 类引用核原文 |
| `refs/tongji_template/` | 同济写作示例（`.docx` 转换版 + 逐段格式 dump） | 排版规范回溯 |
| `puvnet/` | 模型代码（`models/pu_gan.py` `losses/upsampling.py` 等） | 代码贡献者 |
| `scripts/` | 训练 + 审计 + 备份脚本集 | 跑实验 / 写脚本 |
| `configs/*.yaml` | 各 run 的训练配置 | 复现实验 |
| `runs/<run_name>/` | 各 run 的判据存档（详见 §3） | 所有数字回溯目标 |
| `tests/` | 判据脚本的负例测试 | 改脚本前必跑 |

---

## 2. 「写一个数字要回溯到哪」速查

任何章节里出现的数字，必须能在这张表里找到它的存档来源。

| 数字类型 | 存档路径 | 谁写 |
|---|---|---|
| 单 epoch loss / 指标曲线 | `runs/<name>/history.json`（数组，每元素一个 epoch） | `train_pu.py` |
| best epoch / 选点 | `runs/<name>/selection.json` | `train_pu.py` |
| 推理期逐样本数据 | `runs/<name>/metrics.json` | `train_pu.py` |
| 配置 + 训练环境 | `runs/<name>/config.yaml` + `runs/<name>/env.json` | `train_pu.py` |
| 平台区统计 | `docs/_ch3_stats.json`（第 3 章基线） / `compare_runs.py` 输出（第 5 章消融） | `ch3_diagnose.py` / `compare_runs.py` |
| cv_nn 主指标（含 Q4/Q1 分层） | `docs/_cv_nn_measure.json` | `measure_cv_nn.py` |
| ρ 演化曲线 | `runs/rho_trace_B2.json`（含 stats + per-epoch rows） | `analyze_rho_trace.py` |
| 张量形状 / 参数量 | `docs/_ch3_shapes.json` | `ch3_diagnose.py` |
| 极端对照模型标定 | `runs/E000_metric_calibration/result.json` | `metric_calibration.py` |

**核数字脚本**：`scripts/_audit_ch4.py`（第 4 章专用，可复制后改名给其他章节用）。
**统一审计**：`scripts/audit_archive.py`（论文 run 判据文件齐全性 + 跨机器分组）。

---

## 3. 9 个论文 run 的存档位置

| run 名 | 机器 | 判据文件落盘路径 |
|---|---|---|
| B002_baseline150 | 3090 | `runs/B002_baseline150/{history,metrics,config,env,summary_stats,selection,calibration,_index}.json` |
| ABL_A1_cd_balance | 3090 | `runs/ABL_A1_cd_balance/...`（同上） |
| ABL_A2_cd_boost_bwd | 3090 | `runs/ABL_A2_cd_boost_bwd/...` |
| ABL_D1_scale_qk | 3090 | `runs/ABL_D1_scale_qk/...` |
| ABL_C1_uniform | 3090 | `runs/ABL_C1_uniform/...` |
| ABL_AC_combo | 3090 | `runs/ABL_AC_combo/...` |
| B002_baseline150_5090 | 5090 | `runs/B002_baseline150_5090/...` |
| ABL_B1_adv_fixed | 5090 | `runs/ABL_B1_adv_fixed/...` |
| ABL_B2_adv_adaptive | 5090 | `runs/ABL_B2_adv_adaptive/...` |

**未含的 run**：
- B001_reproduce（100ep 试跑，**仅用于取 B1 固定权重 8.27**，不进第 5 章主表）
- SEED_*（s20260812/s20260813，用户停机后未完成，**已弃**，论文不报 seed 稳健性）
- SMOKE_* / R001_local_smoke / _pipeline_check（冒烟辅助，**不入论文**）

---

## 4. 第 4 章主线一句话回顾（写章节时不要搞错）

- **命题**：与 B1（照搬 PU-GAN 固定权重）相比，**B2（GAAW 动态权重）在同机同 seed 条件下，CD 与 cv_nn 同时更优**
- **机制证据**：图 4-1 给出 ρ 跨 150 epoch 演化跨越 3.92×10⁶ 倍、B1 固定 ρ 在 91.3% epoch 偏大
- **对照修订**：第 3.5.5 预注册主指标对照是「无基线」，本章改为「B1」，作为显式修订执行
- **不声明**：不声明 GAAW 在 cv_nn 上一定优于无基线（实测 B2 vs baseline 为 +7.47% 反向显著，按预注册据实报告）
- **H1–H5**（4.2.2 节）的接收条件逐条独立，任一失败不蕴涵其他失败

---

## 5. 引用规范（`{{cite:KEY}}` 替换流程）

1. 草稿期所有外部结论写 `{{cite:KEY}}`（KEY 是 `docs/REFERENCES_TABLE.md` 里的 key，不是数字）
2. 写完一章用 `python scripts/_key2num.py > docs/_key2num.txt` 取最新 `key→number` 映射
3. `python scripts/_dump_cites.py` 打印「[N] 所在句子 | key」表，**人工逐条核对语义一致**
4. 三步通过才跑 `selfcheck_inline_cites.py` 形式校验

**警告**：自动校验器只验「编号在库 + 上下文够字数」，**抓不到「编号有效但语义错位」**，必须人工核语义。

---

## 6. 给接手者的常见坑

- **跨机器**不可比：5090 的数字只能和 `B002_baseline150_5090` 比，不能和 3090 的 `B002_baseline150` 并列。
- **跨 epoch σ ≠ 跨 seed SE**：单 epoch 内 n=75 平台区统计是「同 run 内波动」，跨 seed 才是「run 间波动」，二者口径不可互换。
- **禁止把代码注释 / 内部对话 / agent 分析当外部依据**（见 `STYLE_GUIDE.md §2.9`）。
- **禁止「前人未」「学界共识」「大家争论」**，全降级为「据本文检索范围内未见报告」。
- **「vs 谁的 cv_nn 多少」必须先看是不是同机同 seed，再写结论**。
- **ρ 的 caveat**（写论文必须复述）：`w_auto` 是 epoch 内 batch 的算术平均，`ρ_hat` 是调和口径代表值，**不是 mean(ρ)**，仅可用于数量级演化趋势。
