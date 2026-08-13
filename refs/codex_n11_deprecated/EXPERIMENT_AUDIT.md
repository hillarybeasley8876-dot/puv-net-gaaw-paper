# Experiment Audit Report

**Date**: 2026-08-13  
**Auditor**: fresh GPT-5.6-Sol ultra reviewer（同模型族、只读、provisional）  
**Project**: PUV-Net / GAAW thesis  
**Overall Verdict**: **FAIL**

本判定针对原始实验归档与评价链完整性，不等同于“数据造假”。审计确认真值来自PU1K数据而非模型输出；失败主要来自训练/验证划分不一致、评价口径冲突、D1模型配置重建错误、正式评价链未产出当前论文结果，以及缺少多seed和完整运行包。论文正文已撤回受影响主张，但不能把正文降级解释成原始实验归档已通过审计。

## A. Ground Truth Provenance：WARN

- PU1K训练输入和真值分别读取，见`puvnet/data/pu_dataset.py:142-150`；样本返回时使用真值中心与尺度进行共享坐标变换，见`puvnet/data/pu_dataset.py:187-195`。
- 正式测试路径从独立的输入与真值文件读取，见`puvnet/data/pu_dataset.py:210-235`。
- `measure_cv_nn.py`将模型预测与数据集真值比较，见`scripts/measure_cv_nn.py:123-150`。没有发现由模型输出伪造真值的路径。
- 但训练脚本使用固定置换划分验证集，见`scripts/train_pu.py:203-221`；旧补测脚本却抽取数组尾部，见`scripts/measure_cv_nn.py:203-210`。拆分重建表明旧200条记录中仅11条属于真实验证划分，189条属于训练划分。
- P2F实现使用`trimesh`替代官方CGAL路径，NUC为简化实现，分别见`puvnet/metrics/pointcloud.py:218-254`和`:261-337`，不能冒充官方指标。

## B. Score Normalization：FAIL

- 正式风格的归一化函数把预测与真值分别除以各自最大半径，见`puvnet/metrics/pointcloud.py:72-78,164-210`。预测自身统计量进入分母，会消除全局尺度误差；若采用该口径，必须同时保留并报告`scale_ratio`。
- 当前论文逐样本补测走另一条路径：预测与真值共同处于真值中心和真值尺度定义的坐标框中，并计算平方距离，见`scripts/measure_cv_nn.py:124-150`与`puvnet/data/pu_dataset.py:191-195`。
- `evaluate.py`的元数据把`cd_squared`写为`false`，同时实际计算平方距离，见`scripts/evaluate.py:184-203,288-290`。元数据与实现不一致。
- 当前正文已把补测CD/HD统一改称“共享真值坐标框诊断”，不再称正式独立归一化测试指标；但正式评价代码仍需修复。

## C. Result Existence and Claim Matching：FAIL

- 所列JSON文件均存在并可解析；对`docs/_cv_nn_measure.json`的1,800条运行—样本记录重算后，存档均值可复现。
- `docs/_split_audit_and_holdout_subset.json`保存11/200的拆分归属，`docs/_thesis_results_summary.json`绑定源JSON哈希并只汇总11条真实验证交集。
- D1补测无效：`scripts/measure_cv_nn.py:97-115`始终构造默认模型；D1要求`scale_qk=true`，见`configs/abl_D1_scale_qk.yaml:55-71`，而模型默认值为`false`，见`puvnet/models/pu_transformer.py:371-384`。该开关不改变权重结构，错误加载不会自动报错。
- 当前第6章、摘要和第7章已使用纠正后的探索性口径：B2相对B1的`cv_nn`改善4.84%，但CD、HD分别劣化2.80%和13.43%；B2相对B0四项均未占优；D1旧数值全部排除。
- 结果文件尚未同时绑定数据集、checkpoint、配置、拆分清单、评价代码和commit哈希；当前仓库中的原始训练包也不完整，因而不能从论文包内端到端重放全部运行。

## D. Dead Code and Pipeline Path：WARN

- `evaluate.py`调用正式风格CD/HD、P2F和NUC，见`scripts/evaluate.py:184-224,243-254`；当前论文结果却来自`measure_cv_nn.py`的手工诊断路径，见`:73-81,118-177`。
- 训练期监控只检查首个验证batch中的少量样本，见`scripts/train_pu.py:117-154`，不能替代完整验证或测试评价。
- 训练保存包装后的`PUTransGAN`状态，而`evaluate.py`按原始`PUTransformer`和不一致配置键重建模型，见`scripts/train_pu.py:98-106,368-374`及`scripts/evaluate.py:419-424`。正式评价路径需要先修复checkpoint兼容性。

## E. Scope Assessment：FAIL

- 当前范围为单一PU1K数据集、单一4倍patch设置、每配置一个训练seed。
- 正式测试加载器包含127个模型，见`puvnet/data/pu_dataset.py:210-235`，但当前没有对应的正式测试结果文件。
- 11条真实验证交集是对旧补测档案的事后小样本复核，不能代表完整验证集；样本SE也不能替代seed级不确定性。
- 因此，统计显著、稳健性、泛化、正式测试性能、SOTA或“全面优于”等主张均不受当前证据支持。

## F. Evaluation Type Classification：FAIL

- `scripts/evaluate.py`：`real_gt`正式测试风格评价，但当前无结果证据。
- `scripts/measure_cv_nn.py`、`scripts/ch3_diagnose.py`及`_cv_nn_measure.json`：`real_gt post-hoc patch diagnostic`；原200条切片受训练重叠污染。
- `docs/_split_audit_and_holdout_subset.json`：拆分与来源审计。
- `docs/_thesis_results_summary.json`：11条交集上的事后探索性派生结果。
- 未发现human evaluation或模型生成真值。

## Claim Impact

| 主张 | 裁定 |
|---|---|
| 真值不是由模型输出生成 | 支持 |
| 基线局部间距离散 | 仅对11条真实验证交集作探索性支持 |
| B2相对B1同时改善`cv_nn`、CD和HD | 不支持；旧结论已撤回 |
| B2相对B1缓解`cv_nn` | 探索性方向支持：改善4.84%，8/11条改善 |
| B2优于无对抗基线 | 不支持；四项指标均未占优 |
| C1为竞争基线 | 条件支持；`cv_nn`与HD改善，CD与Q4/Q1恶化 |
| D1效果 | 无效测量，不能解释 |
| 统计显著、稳健、泛化、正式测试或SOTA | 不支持 |

## Required Actions

1. 在真实验证清单或完整官方PU1K测试集上重测所有checkpoint。
2. 从checkpoint保存的模型配置严格重建网络，并重新补测D1。
3. 修复`evaluate.py`对包装checkpoint的加载方式及`cd_squared`元数据。
4. 为数据、checkpoint、配置、拆分、评价代码、commit和结果同时记录SHA-256。
5. 独立归一化指标必须与共享坐标框或原始尺度指标并列，并保留`scale_ratio`。
6. 核心配置至少完成3个独立seed，以seed作为推断单位。
7. 按正式协议补齐NUC、P2F和定性点云结果。

## Current Paper Status after Correction

论文源文件已公开拆分错误、将CD/HD降级为共享真值坐标框诊断、排除D1，并把全部结果限定为11条交集上的探索性描述。这些修正防止了错误主张继续进入摘要与结论，但原始实验归档的总体审计仍为**FAIL / provisional**，不能改写为PASS。
