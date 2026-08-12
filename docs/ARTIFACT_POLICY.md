# 实验产物留档规范（过程图 / 数据支撑）

> 目的：论文里每一张图、每一个数字，都能在半年后被追溯到具体的 run 与原始数据。
> 铁律：**图必须由落盘数据重绘，不得手填数字作图。**

---

## 1. 为什么必须提前定这个规矩

过程图有个特点：**训练跑完再想补就没了**。中间 epoch 的权重、损失分项、
验证点云、显存曲线，如果当时没存，只能重跑（云端就是真金白银）。
所以留档策略必须在第一个正式 run 之前定死。

---

## 2. 目录约定

```
runs/{exp_id}/                     ← exp_id 见 EXPERIMENT_LOG 的 ID 规则
├── config.yaml                    ← 本 run 的完整配置快照（含随机种子）
├── env.json                       ← 环境指纹：torch/cuda/驱动/GPU 型号/commit
├── history.json                   ← 逐 epoch 标量：loss 各分项 / lr / 时长 / 显存峰值
├── metrics.json                   ← 逐次验证的 CD/HD/P2F/NUC
├── ckpt/
│   ├── best.pt                    ← 按验证 CD 最优
│   └── last.pt                    ← 断点续跑用（云端必须有）
├── figures/                       ← 过程图，每张图配同名 .data.json
│   ├── F_loss.png / .data.json
│   ├── F_metric.png / .data.json
│   ├── F_cloud_{model}.png / .data.json
│   ├── F_hist_{model}.png / .data.json
│   └── ...
└── clouds/                        ← 关键中间产物点云（npz，可复现所有定性图）
    ├── epoch{NNN}_{model}.npz
    └── final_{model}.npz
```

---

## 3. 必存清单（每个训练 run）

| 项目 | 频率 | 理由 |
|------|------|------|
| 逐 epoch 标量 (loss/lr/时长/显存) | 每 epoch | 训练曲线图；显存数据用于论文效率表 |
| 验证四指标 | 每 N epoch | 指标曲线；早停依据 |
| **验证集固定样本的输出点云** | 每 N epoch | **定性演进图的唯一来源，事后无法重建** |
| checkpoint (best/last) | 每 N epoch | 续跑 + 复现 + 后续消融复用 |
| 环境指纹 | 一次 | 复现性；跨机器对比时排查差异 |
| 配置快照 | 一次 | 防止「代码改了但记录没改」 |

**固定样本**：验证集里预先选定 3–5 个模型（写死索引），
所有 run 都用同一批，这样跨方法/跨消融的定性图才可比。

---

## 4. 图与论文的对应

| 图 | 生成函数 | 论文位置 | 数据来源 |
|----|----------|----------|----------|
| F_loss | `plot_training_curves` | 实验章·训练细节 | history.json |
| F_metric | `plot_metric_curves` | 实验章·收敛性 | metrics.json |
| F_cloud | `plot_point_clouds` | 主结果·定性对比 | clouds/*.npz |
| F_hist | `plot_nn_histogram` | 均匀性论证 | clouds/*.npz |
| F_ablation | `plot_ablation_bars` | 消融章 | 各 run 的 metrics.json 汇总 |
| F_noise | `plot_noise_robustness` | 鲁棒性章 | beta 扫描各 run |

---

## 5. 可视化模块的已知注意点（实测踩过）

1. **3D 散点必须开深度着色 + 背面剔除**。
   正投影会把前后表面重叠成「实心圆盘」，读者看不出曲面结构与上采样质量。
   `plot_point_clouds(depth_shade=True, cull_backface=True)` 为默认。
2. **点数标注要区分「总点数」与「可见点数」**，剔除背面后二者不同，
   不区分会让读者误判点数。
3. **局部放大框不能太小**。半径 0.18（单位球）下可见点仅 18 个，图失去意义；
   自检已加 `n_drawn >= 30` 断言拦截。
4. **自检只能验「文件存在且非零」，验不出「图是否有效表达信息」。**
   新增图类型后必须肉眼看一次。这条是实测教训：
   第一版点云图自检 PASS，但肉眼一看是三个实心圆盘，完全不可用。
5. 中文字体缺失时自动退回英文标签，不静默出乱码方框（`_CJK_OK` 开关）。

---

## 6. 论文投稿时的溯源包

需要向审稿人/答辩委员提供数据支撑时，打包：

```
figures/*.png          论文用图
figures/*.data.json    每张图的原始数值
history.json           训练全过程
metrics.json           全部指标
config.yaml + env.json 复现所需的配置与环境
EXPERIMENT_LOG.md      对应 run 的记录（含「本 run 不能得出的结论」）
```

`clouds/*.npz` 体积较大，按需提供。

---

## 7. 存档完整性审计（2026-08-12 加）

> 上面的 §3 是「应该存什么」，这一节是「怎么证明真的存到了」。

`scripts/audit_archive.py` 按 §3 逐条查全部 run，输出三类结果：

| 标记 | 含义 | 是否阻塞论文 |
|---|---|---|
| `✗` issues | 硬缺口（真丢了数据） | **阻塞** |
| `·` warns | 可解释缺口（历史豁免 / run 仍在跑 / 非 git 仓库） | 不阻塞 |
| `OK` | 无硬缺口 | — |

除单 run 检查外，它还做两项**只有横向比对才能暴露**的检查：

1. **跨机器混排**：论文 run 按 `env.json` 的 `gpu_name` 分组。
   同一张表的数字**不得跨机并列**（3090 的 baseline 不能和 5090 的消融同表）。
2. **消融可比性**：所有论文 run 的 `batch_size` / `seed` 必须一致，
   否则「单一变量」的前提失效，消融结论无法归因。

**为什么必须配负例测试**：判据类脚本最危险的失效是**假绿**——逻辑写错导致
什么都检不出来，而人看到一片 OK 就以为没问题。
`scripts/negtest_audit_archive.py` 故意造 7 类缺陷（删主表出口 / 删必备图 /
删图的 data.json / 删点云 / 截断 history / 平台区置空 / 删 ckpt），
断言审计器必须全部拦截。**审计器改动后必须重跑负例测试**，否则其 OK 结论不可信。

已知豁免（写在 `LEGACY_EXEMPT`，改代码即改承诺，必须同步改本节）：

- `B001_reproduce` 缺 `summary_stats.json` / `selection.json`：
  它跑在这两个机制上线之前。替代证据 = `convergence_sensitivity.json`
  （多窗口事后重算）+ `selection_replay.json`（同一 selector 事后重放）。

**`git_commit` 为 null 的说明**：本项目当前不是 git 仓库，故 `env_fingerprint()`
采集不到 commit。复现性依赖 `config.yaml` 快照 + `EXPERIMENT_LOG.md` 记录。
若日后 git 化，审计器会自动把这条从 warn 升级为 issue（检测 `.git` 存在）。
