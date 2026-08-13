# Citation Audit Report

**审计日期**：2026-08-13  
**总体结论**：`BLOCKED / provisional`  
**正文实际引用**：81 篇  
**参考库**：116 篇  

## 结论先行

正文中的 81 个引用键全部能在 `docs/REFERENCES.json` 中找到，标题、作者、年份和来源字段完整；每条记录都保留了此前通过 arXiv 或 OpenAlex 等官方接口核验的通道、URL和时间戳。参考库自检为 29 PASS / 0 FAIL，引用键缺失为 0，重复归一化标题为 0。

这些结果支持“文献记录可查询且没有把未核到的候选条目混入正文引用库”。它们不等于“每个引用句都已逐篇阅读全文确认”。本轮未按引用条目完成独立全文语境审查，故审计不得标为全面 PASS，最终状态保持 `BLOCKED / provisional`。

## 三层审计状态

| 层级 | 当前状态 | 可据此声称 | 不可据此声称 |
|---|---|---|---|
| 存在性 | 81/81 有既存官方API核验记录 | 引用记录可追溯、可查询 | 当前网络再次访问一定成功 |
| 元数据 | 81/81 标题、作者、年份、来源字段完整 | 本地库结构和引用键一致 | 所有版本差异均已人工裁定 |
| 语境适切性 | 未完成逐篇独立全文复核 | 已抽取全部引用位置供复核 | 每个句子均得到原文直接支持 |

## 自动核验摘要

- 正文去重引用：81 篇，超过70篇要求。
- 引用总出现次数（不含参考文献表）：224 处。
- 官方核验通道记录：{'arxiv': 67, 'crossref': 1, 'openalex': 14}；同一篇可有多个通道。
- 有效引用键缺失：0。
- 无引用上下文的正文引用键：0。
- 参考库中的35篇未被正文引用，不计入本次81篇正文引用审计，也不冒充引用数量。
- `docs/REF_REJECTED_B1.json` 与 `docs/REF_REJECTED_B2.json` 保留了11条未通过或错误匹配的候选记录，它们没有进入最终引用库。

## 需要完成的提交前工作

1. 逐条打开论文原文，对 `.aris/citation-audit/contexts.txt` 中每个引用句给出 `SUPPORTS / WEAK / WRONG`。
2. 对同一文献的所有出现位置分别检查，不能只凭标题判断。
3. 若发现版本、年份或来源漂移，优先采用正式发表版本并同步正文表述。
4. 在上述复核完成前，不得把本报告状态改为 PASS。

## 逐条状态

表中“已记录核验”仅指存在性/元数据记录；“待全文语境复核”是本轮阻断项。

| 引用键 | 题名 | 年份 | 记录通道 | 存在/元数据 | 语境 |
|---|---|---:|---|---|---|
| `PointNet` | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointNet2` | PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `VoxelNet` | VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointRCNN` | PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `LOAM` | LOAM: Lidar Odometry and Mapping in Real-time | 2014 | openalex | 已记录核验 | 待全文语境复核 |
| `PoissonRecon` | Screened poisson surface reconstruction | 2013 | openalex | 已记录核验 | 待全文语境复核 |
| `PointCloudSurvey` | Deep Learning for 3D Point Clouds: A Survey | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `BallPivoting` | The ball-pivoting algorithm for surface reconstruction | 1999 | openalex | 已记录核验 | 待全文语境复核 |
| `MLS` | Computing and rendering point set surfaces | 2003 | openalex | 已记录核验 | 待全文语境复核 |
| `LOP` | Parameterization-free projection for geometry reconstruction | 2007 | openalex | 已记录核验 | 待全文语境复核 |
| `WLOP` | Consolidation of unorganized point clouds for surface reconstruction | 2009 | openalex | 已记录核验 | 待全文语境复核 |
| `EAR` | Edge-aware point set resampling | 2013 | openalex | 已记录核验 | 待全文语境复核 |
| `PUNet` | PU-Net: Point Cloud Upsampling Network | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `MPU` | Patch-based Progressive 3D Point Set Upsampling | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUGAN` | PU-GAN: a Point Cloud Upsampling Adversarial Network | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUGCN` | PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUTransformer` | PU-Transformer: Point Cloud Upsampling Transformer | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `MAFU` | Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `Grad-PU` | Grad-PU: Arbitrary-Scale Point Cloud Upsampling via Gradient Descent with Learned Distance Functions | 2023 | arxiv | 已记录核验 | 待全文语境复核 |
| `SAPCU` | Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUFlow` | PU-Flow: A Point Cloud Upsampling Network With Normalizing Flows | 2022 | openalex | 已记录核验 | 待全文语境复核 |
| `PUDM` | A Conditional Denoising Diffusion Probabilistic Model for Point Cloud Upsampling | 2023 | arxiv | 已记录核验 | 待全文语境复核 |
| `RepKPU` | RepKPU: Point Cloud Upsampling with Kernel Point Representation and Deformation | 2024 | openalex | 已记录核验 | 待全文语境复核 |
| `GradNorm` | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `UncertaintyWeighting` | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `DWA` | End-to-End Multi-Task Learning with Attention | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `PCGrad` | Gradient Surgery for Multi-Task Learning | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `CAGrad` | Conflict-Averse Gradient Descent for Multi-task Learning | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `NashMTL` | Multi-Task Learning as a Bargaining Game | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `TamingTransformers` | Taming Transformers for High-Resolution Image Synthesis | 2021 | arxiv, crossref | 已记录核验 | 待全文语境复核 |
| `PointSetGen` | A Point Set Generation Network for 3D Object Reconstruction from a Single Image | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `DensityAwareCD` | Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `SRCNN` | Image Super-Resolution Using Deep Convolutional Networks | 2014 | arxiv | 已记录核验 | 待全文语境复核 |
| `EMDLoss` | The Earth Mover's Distance as a Metric for Image Retrieval | 2000 | openalex | 已记录核验 | 待全文语境复核 |
| `HausdorffMetric` | Comparing images using the Hausdorff distance | 1993 | openalex | 已记录核验 | 待全文语境复核 |
| `EC-Net` | EC-Net: an Edge-aware Point set Consolidation Network | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `iPUNet` | iPUNet:Iterative Cross Field Guided Point Cloud Upsampling | 2023 | arxiv | 已记录核验 | 待全文语境复核 |
| `FPS` | The farthest point strategy for progressive image sampling | 1997 | openalex | 已记录核验 | 待全文语境复核 |
| `PointCNN` | PointCNN: Convolution On $\mathcal{X}$-Transformed Points | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `KPConv` | KPConv: Flexible and Deformable Convolution for Point Clouds | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `DGCNN` | Dynamic Graph CNN for Learning on Point Clouds | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `Transformer` | Attention Is All You Need | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `ViT` | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `PCT` | PCT: Point cloud transformer | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointTransformer` | Point Transformer | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointTransformerV2` | Point Transformer V2: Grouped Vector Attention and Partition-based Pooling | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointTransformerV3` | Point Transformer V3: Simpler, Faster, Stronger | 2023 | arxiv | 已记录核验 | 待全文语境复核 |
| `StratifiedTransformer` | Stratified Transformer for 3D Point Cloud Segmentation | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `FastPointTransformer` | Fast Point Transformer | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `TransformerSurvey3D` | Transformers in 3D Point Clouds: A Survey | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `DeepPointsConsolidation` | Deep points consolidation | 2015 | openalex | 已记录核验 | 待全文语境复核 |
| `Dis-PU` | Point Cloud Upsampling via Disentangled Refinement | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUCRN` | Point Cloud Upsampling via Cascaded Refinement Network | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUEVA` | PU-EVA: An Edge Vector based Approximation Solution for Flexible-scale Point Cloud Upsampling | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUDense` | Density-Imbalance-Eased LiDAR Point Cloud Upsampling via Feature Consistency Learning | 2022 | openalex | 已记录核验 | 待全文语境复核 |
| `NePs` | Neural Points: Point Cloud Representation with Neural Fields for Arbitrary Upsampling | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `PUSDF` | Parametric Surface Constrained Upsampler Network for Point Cloud | 2023 | arxiv | 已记录核验 | 待全文语境复核 |
| `GAN` | Generative Adversarial Networks | 2014 | arxiv | 已记录核验 | 待全文语境复核 |
| `LSGAN` | Least Squares Generative Adversarial Networks | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `WGAN` | Wasserstein GAN | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `WGANGP` | Improved Training of Wasserstein GANs | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `SNGAN` | Spectral Normalization for Generative Adversarial Networks | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `GANTrainTricks` | Improved Techniques for Training GANs | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `TTUR` | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `R1Reg` | Analyzing and Improving the Image Quality of StyleGAN | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `GANStability` | Which Training Methods for GANs do actually Converge? | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `Pix2Pix` | Image-to-Image Translation with Conditional Adversarial Networks | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `SRGAN` | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network | 2016 | arxiv | 已记录核验 | 待全文语境复核 |
| `ESRGAN` | ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `LGAN` | Learning Representations and Generative Models for 3D Point Clouds | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `FoldingNet` | FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation | 2017 | arxiv | 已记录核验 | 待全文语境复核 |
| `TreeGAN` | 3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions | 2019 | arxiv | 已记录核验 | 待全文语境复核 |
| `P2PNet` | P2P-NET: Bidirectional Point Displacement Net for Shape Transform | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `ShapeGF` | Learning Gradient Fields for Shape Generation | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointDiffusion` | Diffusion Probabilistic Models for 3D Point Cloud Generation | 2021 | arxiv | 已记录核验 | 待全文语境复核 |
| `LION` | LION: Latent Point Diffusion Models for 3D Shape Generation | 2022 | arxiv | 已记录核验 | 待全文语境复核 |
| `PointCloudGenSurvey` | A Survey on Deep Geometry Learning: From a Representation Perspective | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `SAGAN` | Self-Attention Generative Adversarial Networks | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `GradVac` | Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models | 2020 | arxiv | 已记录核验 | 待全文语境复核 |
| `MGDA` | Multi-Task Learning as Multi-Objective Optimization | 2018 | arxiv | 已记录核验 | 待全文语境复核 |
| `MTLSurvey` | Multi-Task Learning for Dense Prediction Tasks: A Survey | 2020 | arxiv | 已记录核验 | 待全文语境复核 |

## 可复核产物

- `CITATION_AUDIT.json`：机器可读状态与输入哈希。
- `.aris/citation-audit/contexts.txt`：按引用键、文件和行号抽取的正文语境。
- `.aris/traces/citation-audit/2026-08-13_run01/reviewer.md`：说明独立全文复核未运行，防止误报。
