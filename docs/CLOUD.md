# CompShare 云端环境说明

> 状态：**API 全线打通、规格与镜像已实测确认、仍未开机、累计花费 0 元。**
> 最近更新：2026-08-11，补入 5090 规格/价格/镜像的一手 API 实测结果。
> 开机前请先读 §4 的前置条件。

---

## 0. 一手实测速查表（2026-08-11）

全部经 `api.compshare.cn` 只读接口实测，原始响应存档于
`runs/probe_compshare/`（`probe_result.json` / `gputype_verify.json` /
`compshare_images.json` / `compshare_images_raw.json`）。

### 0.1 真实 Action 名（来源：官方 SDK 仓库 + 实测枚举，非猜测）

一手来源：`https://github.com/ucloud/compshare-developer-examples`
→ `python-sdk/compshare/main.py`（`create_comp_share_instance` /
`describe_comp_share_instance`），UCloud SDK 的 snake_case 方法名对应驼峰 Action 名。

| Action | 实测 RetCode | 用途 | 坑 |
|---|---|---|---|
| `GetRegion` | 0 | 地域列表 | — |
| `DescribeImage` | 0 | **通用云主机**镜像池（73 项 `uimage-*`） | ⚠️ 不是 CompShare 的池子，`SupportedGPUTypes` 全空 |
| `DescribeCompShareImages` | 0 | **CompShare 专属**镜像池（41 项 `compshareImage-*`） | ⚠️ 复数形式；**不能传 `Limit`**，否则 RetCode 230 |
| `DescribeCompShareInstance` | 0 | 实例列表（当前 `TotalCount=0`） | — |
| `GetCompShareInstancePrice` | 0 | 按量查价 | — |
| `CreateCompShareInstance` | 未调用 | 创建实例 | 🔒 已列入 `MUTATING_ACTIONS`，默认 dry-run |

已实测**不存在**（RetCode 161）的形式，勿再尝试：
`DescribeCompShareImage`（单数）、`DescribeCompShareMachineType`、
`DescribeCompShareInstanceType`、`DescribeMachineType`、
`DescribeUHostInstanceType`、`DescribeResourceType`、`DescribeGPUType`。

### 0.2 GPU 规格与价格（cn-wlcb-01, GPU=1, CPU=16, Mem=64G, ChargeType=Dynamic）

| GpuType | 现价（元/时） | 原价 | 查价 RetCode | **创建 RetCode** |
|---|---|---|---|---|
| **5090** | **2.77** | 2.92 | 0 | ❌ **8333** |
| 4090 | 2.05 | 2.15 | 0 | ⚠️ 226604 库存不足 |
| 3090 | 1.13 | — | 0 | 未测 |
| 3080Ti | 0.93 | — | 0 | 未测 |
| `9090` / `H100` / 空串（伪造对照） | — | — | **230** | — |

### 0.2.1 ⚠️⚠️ 最大的坑：8333 的真实成因是**地域**，不是配比（已破案）

**这是本轮花最多时间踩到的坑，务必先读。**

`GetCompShareInstancePrice` 对 5090 返回 `RetCode 0` 且价格合理（2.05~2.77
元/时，随配比变化），**但在 `cn-wlcb-01` 上 `CreateCompShareInstance` 一律返回**：

```
RetCode 8333  "cpu memory ratio not in 2:1 - 1:12"
```

**该错误文案是彻底误导的 —— 与 CPU/内存配比毫无关系。** 实测证据链：

1. 扫了 24 种 CPU/内存组合（8~32 核 × 2/4/8 倍内存），**全部 8333**，
   其中包含官方 SDK 示例的 16C/64G 与落在 "2:1 - 1:12" 区间内的多种配比。
2. **完全不传 `CPU` / `Memory`**（让服务端用默认值）→ 仍然 8333。
   → 证明它并非在校验我传的配比。
3. **关键对照**：同一套参数只把 `GpuType` 换成 `4090` →
   `RetCode 226604 "This GPU type is currently out of resources"`。
   → 证明请求参数结构、签名、镜像、磁盘参数**全部正确**，
     能走到库存检查这一步；4090 只是缺货。
4. `Memory` 单位确认是 **MB**（传 16 得到 `230 Params [Memory] not available`，
   传 16384 得到 8333）。

#### 真相（2026-08-11 破案）

用户在控制台已有一台 **5090 实例位于「上海二A」**，据此反查得到：

| Region | Zone | 5090 情况 |
|---|---|---|
| `cn-wlcb` | `cn-wlcb-01` | 查价 RetCode 0，**创建恒 8333** |
| **`cn-sh2`** | **`cn-sh2-01`** | ✅ **实际有 5090 在售并可运行** |
| `cn-sh2` | `cn-sh2-02` / `-03` | 查价 RetCode 0 |
| `cn-shanghai`、`cn-sh2a`、`cn-shanghai-2a` | — | `230`，**均为非法代号** |

**结论：`8333` 的真实语义是「该 Zone 不提供此 GPU 型号」**，服务端错误映射到了
一个配比校验的文案上。之前"5090 需要配额/白名单"的猜测**是错的，已作废**。

**行动建议**：遇到 8333 先换 Zone，不要在配比上穷举。
上海二A 的正确代号是 `Region=cn-sh2` + `Zone=cn-sh2-01`。

> 另一个 Zone 差异：`cn-sh2-01` 询价时传 `ChargeType=Dynamic` 会得到
> `230 Params [ChargeType] not available`，而 `cn-sh2-02/-03`、`cn-wlcb-01` 正常。
> 该区实例实际字段为 `ChargeType=Postpay`。**不同 Zone 的合法计费方式清单不同。**

**为什么价格表可信**：预注册判据 J1 —— 先验证服务端**确实校验** `GpuType`。
伪造值 `9090`、`H100`、空串全部被打回 230，而 5090 返回 0 且价格与其它档位互异
（判据 J2），排除了「服务端静默忽略未知参数、返回默认档价格」这一失败模式。

> 注：官网页面标价 3.32 元/时，API 实测 2.77 元/时（原价 2.92）。以 API 为准，
> 差异可能来自页面规格配比不同或活动折扣。**论文成本核算用 2.77。**
> 另：官方 SDK 示例注释里 `GpuType` 只写了 "4090, 3080Ti or 3090"，
> **该注释已过期**，实测 5090 可用。

### 0.2.3 🚨 事故记录：探测请求把机器真开起来了（2026-08-11 02:56:09）

**这是一次真实的误操作，必须记住教训。**

目标是找出「无卡模式启动」的 API 参数名。手法是给开机 Action 传候选参数名、
值设为非法字符串 `__probe__`，预期服务端在参数校验阶段拒绝，从而安全地
区分「参数不存在」与「参数存在但值非法」。

实际发生：

```
StartCompShareInstance(UHostId=<合法>, WithoutGpu="__probe__")  ->  RetCode 0
实例带 GPU 启动：State=Running, GPU=1, StartTime=1786388169
```

**根因（设计缺陷，不是运气问题）**：服务端对**未知参数静默忽略**，于是
「`UHostId` 合法」本身就构成了一个完整有效的开机请求。我只把模式参数设为非法值，
却让主键保持合法 —— 请求在"忽略未知参数"这一解释下完全成立。

**正确做法**：探测未知参数名时，必须让请求**在任何解释下都不可能成功**，
即**把主键（`UHostId`）也设为非法值**。服务端会先校验主键，在触达业务逻辑前拒绝。

**附带发现两条**：

1. **`PoweronCompShareInstance` 实际不存在**（`RetCode 161`）。
   它此前一直被 dry-run 护栏拦下、从未真正发送，导致我误以为它存在。
   → **dry-run 拦截 ≠ Action 存在**；护栏名单里的名字同样需要实测确认。
2. **`StartCompShareInstance` / `StopCompShareInstance` 才是真实的开关机 Action**，
   而它们此前**不在 `MUTATING_ACTIONS` 名单里**，能直接绕过 dry-run 护栏。
   已补入名单（连同 `Restart*` / `Modify*` / `CreateCompShareImage`）。

**损失**：约 1 分钟 GPU 计费，不到 0.05 元。
**意外收获**：拿到 `SshLoginCommand`（关机状态下该字段为空），
且确认登录用户是 **root** —— 这是 venv 隔离方案成立的前提。

**护栏纪律更新**：
- 凡开关机/删除/创建类 Action，一律先进 `MUTATING_ACTIONS` 再谈调用。
- 参数名探测必须让主键也非法。
- 任何探测脚本收到 `RetCode 0` 都要立即停止并核查真实资源状态。

### 0.2.2 签名 bug：bool 参数导致 RetCode 171（已修）

`CreateCompShareInstance` 需要 `Disks.0.IsBoot=true`，这是第一个带 bool 的请求，
一发就 `RetCode 171 Signature VerifyAC Error`。

根因：签名侧 `_encode_value(True)` 按文档得到 `"true"`，
而发送侧 `urllib.parse.urlencode(dict)` 调用 `str(True)` 得到 `"True"`。
服务端按收到的 `True` 重算签名 → 必然不匹配。

**因为此前所有只读调用都不含 bool 参数，这个 bug 潜伏了整轮调研**，
且症状（171）与"密钥错误"完全一样，极易误判。

修复：发送前统一 `{k: _encode_value(v) for k, v in req.items()}`，
让签名与发送共用同一套编码规则。已加防回归自检项，
`python -m puvnet.cloud.compshare selfcheck` 现为 **7/7 PASS**。

> 教训：`RetCode 171` 的排查顺序应为
> ① 签名自检过官方向量 → ② **检查签名侧与发送侧编码是否一致** →
> ③ 换 endpoint → ④ 最后才怀疑密钥。

### 0.3 适配 5090 的镜像（41 项中筛出）


`SupportedGpuTypes` 含 `5090` 的镜像共 **36 / 41**。版本信息的**权威来源是
`Softwares` 子字典**（`Framework` / `FrameworkVersion` / `CUDAVersion` /
`PythonVersion` / `OsVersion`）——不要解析 `Name` 字符串，命名不规范
（`cuda128` 无小数点，正则会漏判）。

| CompShareImageId | Name | Framework | CUDA | Python | 盘 | 选用 |
|---|---|---|---|---|---|---|
| `compshareImage-1minbz219ceq` | cuda128_torch291_py312 | **PyTorch 2.9.1** | **12.8** | 3.12 | 30 GB | ✅ **首选** |
| `compshareImage-1mindjx61soz` | cuda130_torch291_py312 | PyTorch 2.9.1 | 13.0 | 3.12 | 50 GB | 备选 |
| `compshareImage-1t1vuerl80np` | cuda132_torch2130_py312 | PyTorch 2.13.0 | 13.2 | 3.12 | 50 GB | 备选（版本跨度大） |
| `compshareImage-1minbz...`/`1t1w06ukq68v` | cuda13x_python312 | 纯 CUDA | 13.0/13.2 | 3.12 | 30 GB | 需自装 torch |

全部 `Status=Available`、`Price=0`（镜像本身不额外收费）、`Container=True`、
`OsVersion=Ubuntu 22.04`。

**首选 `compshareImage-1minbz219ceq` 的理由**：
1. CUDA 12.8 是 sm_120（Blackwell）的**最低可用版本**，官方 torch 从 cu128 起提供 sm_120 kernel；
2. torch 2.9.1 与本机 2.5.1 跨度可控，API 破坏性变更风险低于 2.13；
3. 盘 30 GB（比 50 GB 的省钱且够用：数据 1.3 GB + 镜像自身）；
4. Python 3.12 vs 本机 3.10 —— 需注意，见 §4 前置条件。

---

## 0.4 ✅ 实机：用户已有的 5090（本项目实际使用的机器）

**2026-08-11 起，本项目云端训练用这台，不再新建实例。**

| 字段 | 值 |
|---|---|
| `UHostId` | `cpod-1tq6i2ltk5mj` |
| `Name` | `h3-comfyui-5090` |
| `Region` / `Zone` | **`cn-sh2` / `cn-sh2-01`**（上海二A） |
| `InstanceType` | **`Container`** |
| `GpuType` / `GPU` | **5090** / 1 |
| 显存实测 | **32607 MiB**（`GraphicsMemory.Value=32`） |
| `compute_cap` | **12.0（sm_120, Blackwell）** |
| 驱动 / CUDA | **595.80 / CUDA 13.2** |
| `CPU` / `Memory` | 14 核 / 49152 MB（`free` 可用 47 GB） |
| 磁盘 | 100 GB（`overlay`，已用 36 G，**剩余 65 G**） |
| `ChargeType` | **`Postpay`**（按量，关机不计算力费） |
| `SupportWithoutGpuStart` | `True`（控制台有「无卡模式启动」按钮） |
| 镜像 | `compshareImage-1tlwx8g5r0k2` MiniMax H3｜ComfyUI 三套官方流程 |
| `IngressHost` | `cpod-1tq6i2ltk5mj-s1.pod.compshare.cn` |
| `Ports` | HTTP `[8888, 8889, 8188]` / TCP `[23]` |
| SSH | `ssh -p 28870 root@cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn` |

**查询入口（重要）**：容器实例**没有**独立的 cpod API。
枚举过 `DescribeCPod` / `DescribeCpodInstance` / `DescribeCompSharePod` /
`DescribeContainerInstances` 等 **20 个候选，19 个 `RetCode 161` 不存在**。
正确入口就是普通的 `DescribeCompShareInstance`，只要 Region/Zone 填对：

```python
cli = CompShareClient(); cli.region, cli.zone = "cn-sh2", "cn-sh2-01"
cli.call("DescribeCompShareInstance", Limit=20)   # TotalCount=1
```

**`SshLoginCommand` 只在 `State=Running` 时才有值**，关机时为空字符串。
`Password` 字段是 **base64 编码**的，需 `base64.b64decode()` 后使用。

### 0.4.1 环境隔离结论：venv 方案零污染风险（实测）

摸底发现（`scripts/remote_probe_5090.py`）：

| 检查 | 结果 |
|---|---|
| 登录用户 | **root（uid=0）**，`apt-get` 可用 |
| 系统 Python | 3.10.12 @ `/usr/bin/python3` |
| **系统 python3 的 torch** | **未安装**（`ModuleNotFoundError: No module named 'torch'`） |
| ComfyUI 运行环境 | 自带独立 venv `/…/x-h3-comfyui/myenv/bin/python`（见 nvidia-smi 进程列表） |
| `python3 -m venv` | 可用 |
| pip 源 | 全局已配清华镜像 |
| 网络 | pypi 200（2.6 s）、github 200（0.36 s） |

**关键结论：ComfyUI 跑在它自己的 venv 里，系统 python3 是干净的。**
因此在 `/root/puvnet-venv` 建独立 venv **物理上不可能污染 H3 那条线** ——
风险等级从"需要小心"降为"不存在"。

### 0.4.2 ⚠️ 跨机器 torch 版本差异必须在论文里标注

`compute_cap 12.0`（sm_120）要求 **CUDA 12.8+ 编译的 torch**。
本机 3090 用的 `torch 2.5.1+cu121` **不支持 sm_120**，远端必须装 cu128 及以上。

这带来一个**真实的实验设计约束**：

- 本机 3090：`torch 2.5.1+cu121`，sm_86
- 云端 5090：`torch 2.x+cu128`，sm_120

**两者的数值结果严格来说不可直接混进同一张论文表**（不同 CUDA/cuDNN 版本的
归约顺序与 TF32 策略可能不同）。处理方式：

1. `scripts/train_pu.py` 的 `env_fingerprint()` 已记录 torch/CUDA/设备/capability，
   每个 run 的 `env.json` 都有；
2. 论文主表里**同一张表的所有行必须来自同一台机器**；
3. 若必须跨机器比较，须补一组"同配置双机复跑"对照，量化环境带来的差异。

**2026-08-11 实测落定的版本组合：**

| | 本机 | 云端 5090 |
|---|---|---|
| GPU | RTX 3090 | RTX 5090 |
| capability | 8.6 (sm_86) | **12.0 (sm_120)** |
| SM 数 | 82 | **170** |
| 显存 | 24.0 GB | 31.36 GB |
| torch | 2.5.1+cu121 | **2.11.0+cu128** |
| CUDA | 12.1 | 12.8 |
| Python | 3.10.9 | 3.10.12 |
| OS | Windows-10-10.0.26200 | Linux-6.12.35-generic / glibc2.35 |
| `arch_list` | sm_50…sm_90 | sm_75,80,86,90,**100,120** |

远端 `arch_list` 含 `sm_120`，`torch.cuda.is_available()=True`，
4096³ matmul + backward 实跑通（0.436 s，grad_norm 265046.28）。

### 0.4.3 ✅ venv 部署实录（含两个必须记住的坑）

远端环境：`/root/puvnet-venv`（7.0 GB），工作目录 `/root/puv-net`。
ComfyUI 的 `/workspace/minimax-h3-comfyui/myenv` 全程未被触碰 —— **零污染已验证**。

依赖实测版本：`numpy 2.2.6` / `scipy 1.15.3` / `h5py 3.16.0` /
`trimesh 5.0.0` / `matplotlib 3.10.9`。

#### 坑 1：torch 轮子源国内速度差 268 倍（必须先测速）

上一轮直接用官方源闷头下，实测 **2.2 KB/s**，白等十几分钟。
教训写成了 `scripts/test_mirror_speed.py`（20 秒采样定生死），实测：

| 源 | 均速 | 859 MB 预计耗时 | 判定 |
|---|---|---|---|
| `mirrors.aliyun.com/pytorch-wheels/cu128/` | **16403.7 KB/s (16.0 MB/s)** | **0.9 min** | GOOD |
| `download.pytorch.org/whl/cu128` | 61.3 KB/s | 239.2 min | BAD |

- **清华 `pytorch-wheels` 镜像已下线**（`HTTP 404`，旧笔记过期，实测确认）。
- 阿里云该目录是**扁平文件列表，不是 PEP 503 index** →
  用 `--index-url` 指过去会解析失败，必须用完整 whl URL 直接 `pip install`。
- 可用直链（已验证 `HEAD 200` / `Content-Length 900927048` / `application/zip`）：
  `https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl`
- 远端系统 Python 3.10.12 → 必须选 `cp310`；x86_64 → `manylinux_2_28`（**不是** `linux_x86_64`）。

#### 坑 2：`curl -o` 改名会让 pip 拒装

`pip` 从**文件名**解析包名/版本/tag。用 `curl -o torch_cu128.whl` 改名后：

```
ERROR: Invalid wheel filename (wrong number of parts): 'torch_cu128'
```

→ 下载本地轮子时**必须保留原始 whl 文件名**（已修入 `setup_venv_5090.py`）。

#### 坑 3：判 "stalled" 后必须复核"包是否已装成"

本轮实际装成的是 **torch 2.11.0+cu128**，不是我下载的 2.9.1 ——
上一轮被判 stalled 并 `task_stop` 的官方源下载，**pip 其实已完成安装**。
速度判断没错，但对**最终结果**的判断错了，导致这轮 859 MB 下载纯属冗余。

> **纪律**：中止长任务后，先查产物状态（`pip show` / `import`），再决定是否重做。

### 0.4.4 ⚡ GPU 加速比基准（真实模型真实训练步）

**为什么不用合成 matmul**：本模型含大量 kNN gather / index_select，
可能偏访存瓶颈，matmul 基准会严重高估 5090 收益。
故用真实 `PUTransformer` 的完整训练步（forward + 双向 CD + backward + `optimizer.step`），
脚本 `scripts/bench_gpu.py`，SEED=20260811，batch64 / n_in256 / up4 / warmup10 / 40 步。

#### ✅ 最终结论（干净测量，2026-08-11）

| | 3090（干净） | 5090 | 比值 |
|---|---|---|---|
| 中位数单步 | 104.25 ms | **46.66 ms** | **2.234×** |
| 均值 ± 标准差 | 104.33 ± 0.70 ms | 46.76 ± 0.54 ms | — |
| 最快 / 最慢 | 103.16 / 106.48 ms | 46.57 / 49.94 ms | — |
| **峰值显存** | **5.761 GB** | **5.762 GB** | **1.000** |
| 参数量 | 1,152,803 | 1,152,803 | ✅ 一致 |
| SM 数 | 82 | 170 | 2.073× |
| 单 epoch (1078 步) | 1.87 min | **0.84 min** | — |
| 100 epoch | 3.12 h | **1.40 h** | — |

**真实加速比 = 2.23×**

**关键洞察：每 SM 效率 = 2.234 / 2.073 = 1.078。**
加速比几乎完全由 SM 数量解释，Blackwell 架构代际红利仅 **7.8%**。
这证实了「本模型偏访存瓶颈而非算力瓶颈」的判断 ——
**若当初用合成 matmul 测，会得到远高于 2.23× 的结论并据此做出错误排产决策。**

#### 🚨 污染测量的教训：虚高 3.30 倍

第一次测 3090 时 B-001 全量训练正占用 GPU（util 98% / 9221 MiB），得到：

| | 污染值 | 干净值 | 虚高 |
|---|---|---|---|
| 3090 中位数单步 | 343.92 ms | 104.25 ms | **3.30×** |
| 推算加速比 | 7.370× | 2.234× | **3.30×** |

当时未采信 7.37×，判据是 **SM 比只有 2.07×，7.37 远超硬件差距能解释的范围**，
并预估真值在 2~3× 区间 —— 干净测量得 2.234×，落在区间内。

> **纪律**：GPU 基准必须在**独占** GPU 时测。
> 若无法独占，先用「硬件规格比」（SM 数 / 带宽）做上界 sanity check，
> 超出上界的加速比一律不可采信。
> 峰值显存跨机器一致（5.761 vs 5.762 GB，差 1 MB）是可比性的独立证据，
> 三次测量都一致 —— 不靠口头声明。

#### 排产测算（8 个消融 × 100 epoch）

| 方案 | 墙钟 | 成本 |
|---|---|---|
| 全本机 3090 串行 | **25.0 h** | **0 元** |
| 全 5090 串行 | 11.2 h | 23~31 元 |
| 2 台 5090 并行 | 5.6 h | 23~31 元 |
| 4 台 5090 并行 | 2.8 h | 23~31 元 |
| 8 台 5090 并行 | **1.4 h** | 23~31 元 |

注：并行不增加总卡时，故总成本不变，只压缩墙钟。

另注：5090 基准前 GPU 现状 `508 MiB / 32607 MiB, 0%, 29°C, 9.18 W`
（ComfyUI 常驻约 500 MiB，已如实记录；基准峰值 5.762 GB 远未触顶）。

---

## 1. 现在到底要不要开机

用户既定原则：**先零成本本地跑通，再上云花钱。** 当前状态：

| 项目 | 状态 |
|------|------|
| 最小闭环（数据→模型→loss→反传→ckpt） | ✅ R-001 PASS |
| `scripts/evaluate.py`（numpy 精确指标） | ✅ 自检 6/6 PASS |
| `configs/b001_reproduce.yaml`（正式规模） | ✅ 已定 |
| 本地显存峰值实测 | ✅ batch16 = 1.469 GB → batch64 推算 ≈ 5.9 GB |
| 本地单 epoch 时长实测 | ✅ 6.4 s @1900 样本 → 全量 69000 ≈ 232 s ≈ 3.9 min |

**推算结论：B-001 全量 100 epoch 本机 3090 约 6.5 小时，显存 5.9/24 GB 富余。**
即本机**跑得动**，云端的价值不在「跑得动」而在「并行」与「时间墙」，见 §1.1。

### 1.1 云端的真实价值：并行消融，不是单跑主模型

本课题的算力瓶颈不是单次训练，而是**消融组数量**。按 `docs/PAPER_REMAKE_PLAN.md`
的 B 组设计，至少需要：B-001 基线 + 改进 A/B/C/D 各一组 + 组合若干
+ PU-GAN baseline，合计 **7~10 个 run × 6.5 h**。

| 方案 | 墙钟时长 | 花费 |
|---|---|---|
| 全部本机串行（3090） | 45~65 h | 0 元 |
| 5090 单卡串行 | 约 30~45 h（按 1.5× 加速估） | 83~125 元 |
| 5090 开 3~4 台并行 | **8~12 h** | 83~125 元（总卡时不变） |

**关键洞察：按量计费下，并行 4 台的总花费与串行 1 台相同**（总 GPU·小时不变），
但墙钟时间压缩到 1/4。所以云端的正确用法是**同时开多台各跑一个消融组**，
而不是把单个 B-001 搬上去。

> ⚠️ 1.5× 加速比是**估算，未实测**。5090 相对 3090 的实际加速取决于本模型是
> 访存瓶颈还是算力瓶颈（本模型含大量 kNN gather，可能偏访存）。
> **开机后第一件事是跑 R-001 同配置测真实加速比**，若加速比 < 1.2，
> 云端只剩「并行」价值，须重新核算是否值得。

---

## 2. ⚠️ 关键技术坑：5090 与 torch 版本不兼容（已定解法）

**这是必须在开机前解决的问题，不是可选项。**

| | 本机 | CompShare 目标 |
|---|------|----------------|
| GPU | RTX 3090 24G | RTX 5090 32G |
| 架构 | Ampere, **sm_86** | Blackwell, **sm_120** |
| 当前 torch | 2.5.1+cu121 | ❌ **不支持 sm_120** |

torch 2.5.1 的 cu121 构建不含 sm_120 的编译产物。直接把本地环境搬到 5090 上，
典型症状是 `no kernel image is available for execution on the device`，
或退化到极慢的 PTX JIT 路径。

**解法（已实测确认可用）**：直接用官方镜像 `compshareImage-1minbz219ceq`
（PyTorch 2.9.1 + CUDA 12.8 + Python 3.12），不自己装 torch。

- 本地（sm_86）：`torch==2.5.1+cu121`
- 云端（sm_120）：`torch==2.9.1` / CUDA 12.8（镜像自带）

  不要凭本文档的推测直接锁版本**）

因此代码必须做到「不依赖特定 torch 小版本」，避免踩到 API 变更。

---

## 3. 已就绪的组件

### 3.0 ⚠️ endpoint 与地域：踩过的两个坑（实测结论）

**坑 1：CompShare 有独立 API 网关，与 UCloud 主站密钥不通用。**

| endpoint | 用 CompShare 密钥的结果 |
|----------|------------------------|
| `https://api.compshare.cn` | ✅ `RetCode 0` |
| `https://api.ucloud.cn` | ❌ `RetCode 171 Signature VerifyAC Error` |
| `https://api.ucloudstack.com` | ❌ 连接超时 |

关键教训：**`RetCode 171` 不一定是签名算错。** 本项目签名对官方测试向量 5/5 PASS，
报 171 的真实原因是凭证不属于那个网关。排查顺序应为：先验签名自检 → 再换 endpoint →
最后才怀疑密钥本身。

**坑 2：地域编码与 UCloud 主站不同。**

传主站地域码会报 `RetCode 230 Params [Region] not available`。
实测账户可用地域（`GetRegion`，`RetCode 0`，共 11 个可用区）：

| Region | Zone | 默认 |
|--------|------|------|
| `cn-wlcb` | cn-wlcb-01 | ✅ |
| `cn-bj2` | cn-bj2-02 | |
| `cn-sh2` | cn-sh2-01 | |
| `cn-gd` | cn-gd-02 | |
| `us-den` | us-den-01 | |

已把 `DEFAULT_REGION` 设为账户默认的 `cn-wlcb`（内蒙乌兰察布）。
endpoint 可用 `COMPSHARE_ENDPOINT` 环境变量覆盖。

### 3.1 连通性验证结果（2026-08-10 实测）

```
whoami  -> public_key=4ebn*************************0aQ9, private_key=<hidden>
regions -> RetCode 0, 11 个可用区
list    -> RetCode 0, TotalCount 0（无运行中实例，未产生账单）
```

**账户当前无运行实例。**

| 文件 | 作用 | 自检状态 |
|------|------|----------|
| `puvnet/cloud/compshare.py` | 签名 + 只读查询 + 受保护的开关机 | ✅ 签名 5/5 PASS，API 连通 |
| `.env.local` | 真实凭证（已配置，被 gitignore 屏蔽） | ✅ 已验证可用 |
| `.env.local.example` | 凭证模板 | — |
| `.gitignore` | 屏蔽凭证/数据/权重 | ✅ |
| `tests/test_cloud_guard.py` | dry-run 护栏验证 | ✅ PASS（真凭证下复测通过） |

### 3.1 安全设计（三道防线）

1. **凭证不入库**：`.env.local`、`*.pem`、`*.key` 全部被 `.gitignore` 屏蔽
2. **凭证不入日志**：`CompShareClient.__repr__` 里私钥恒为 `<hidden>`，
   公钥经 `mask()` 脱敏；已实测 `私钥是否泄漏进 repr = False`
3. **变更动作默认 dry-run**：`MUTATING_ACTIONS` 白名单内的动作（创建/开机/关机/
   删除/变配）在 `confirm=False` 时**只返回将要发送的参数，不发网络请求**。
   已实测创建与关机均被拦下。

### 3.2 签名算法验证

来源：<https://docs.ucloud.cn/api/summary/signature.md>

规则：参数按名升序 → 拼成 `key1value1key2value2...`（不做 URL 转义）→
末尾追加 PrivateKey → 取 SHA1。**PrivateKey 只参与本地计算，不随请求发送。**

用官方测试向量自检（不需要真实凭证）：

```powershell
python -m puvnet.cloud.compshare selfcheck
```

实测输出：

```
[PASS] 官方测试向量
       期望 cba5cf5ec4d4233d206b1b54951e3787350a642f
       实际 cba5cf5ec4d4233d206b1b54951e3787350a642f
[PASS] 参数顺序无关（升序排序生效）
[PASS] bool 编码为 true/false，实际 true/false
[PASS] float 编码 42.0->42, 1.5->1.5
[PASS] float 不用科学计数法 1e-7->0.0000001
selfcheck PASS
```

三个易错点已覆盖：`str(True)` 在 Python 里是 `"True"` 而文档要求 `"true"`；
`42.0` 必须编码成 `"42"`；小数不能用科学计数法。

---

## 4. 开机前置条件

### 4.1 已满足（实测）

- [x] `scripts/evaluate.py` 写完，自检 6/6 PASS，走官方 CD/HD 语义
- [x] `configs/b001_reproduce.yaml` 定稿（正式规模，100 epoch）
- [x] 本地跑通冒烟 R-001，记录显存峰值（1.469 GB @batch16）与单 epoch 时长（6.4 s @1900）
- [x] 据实测推算云端总时长与费用（见 §1.1）
- [x] 凭证配好并验证（`.env.local`，只读接口全 RetCode 0）
- [x] 5090 规格可用性 + 价格 + 镜像 全部 API 实测确认（§0）

### 4.2 开机后必须立刻做的四件事（顺序不能乱）

镜像的 torch 是 **2.9.1 / Python 3.12**，本机是 **2.5.1 / Python 3.10**。
这是一次**跨大版本迁移**，"本机对、云端静默算错"是真实风险。

1. **环境指纹**：`python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`
   —— 必须看到 `sm_120`（即 `(12, 0)`），否则立刻停手排查。
2. **全套自检重跑，一个都不许跳**：
   | 模块 | 期望 |
   |---|---|
   | `puvnet/metrics/pointcloud.py` | 8/8 ALL PASS |
   | `puvnet/losses/upsampling.py` | 11/11 ALL PASS |
   | `puvnet/data/pu_dataset.py` | 7/7 ALL PASS |
   | `puvnet/viz/visualize.py` | ALL PASS |
   | `scripts/evaluate.py` | 6/6 ALL PASS |
   | `puvnet/models/pu_transformer.py` | 参数量 = **1,152,803** |
   | `puvnet/models/pu_gan.py` | D = **255,426**，G+D = **1,408,229** |
   参数量是**跨版本一致性的硬锚点**：数字对不上说明有层的默认行为变了。
3. **跨设备数值一致性**：用同一 seed 跑 R-001 冒烟，与本机结果对表。
   本机 5 epoch 的 best `monitor_cd = 0.003533`。**不要求逐位相同**
   （cuDNN/TF32 差异合理），但量级与下降趋势必须一致；若差超 2 倍视为异常。
4. **加速比实测**：R-001 同配置计时，与本机 6.4 s/epoch 对比，算真实加速比。
   写入 `EXPERIMENT_LOG.md`。**若 < 1.2，重新评估上云价值**（§1.1 的 1.5× 只是估算）。

### 4.3 尚未解决

- [ ] 数据上传方案未定（PU1K.zip 972 MB + PU-GAN h5 340 MB = 1.3 GB）。
      候选：① 云端直接从原始 URL 下载（已实测四数据集可直连，最省事）；
      ② 本地上传。**优先 ①**，避免上行带宽瓶颈。
- [ ] `CreateCompShareInstance` 真实调用尚未验证（当前被 dry-run 拦住，符合预期）。
      首次开机需确认必填参数集与返回的 `UHostIds` 结构。
- [ ] 多机并行的 run 编排与结果回收脚本未写。

## 5. 凭证配置步骤

```powershell
# 1. 复制模板
Copy-Item .env.local.example .env.local

# 2. 用编辑器打开 .env.local，填入控制台里的真实公钥/私钥
#    控制台：https://console.compshare.cn

# 3. 验证凭证可用（只读，不产生账单）
python -m puvnet.cloud.compshare whoami    # 看脱敏后的公钥
python -m puvnet.cloud.compshare regions   # 拉地域列表
python -m puvnet.cloud.compshare list      # 列出已有实例
python -m puvnet.cloud.compshare price     # 查价，只读
```

只读探测脚本（全部零花费，可反复跑）：

```powershell
$env:PYTHONPATH='E:\AE-CC托管\puv-net'
python scripts/probe_compshare_5090.py           # Action 名探测
python scripts/verify_compshare_gputype.py       # GpuType 有效性 + 价格对照
python scripts/list_compshare_images.py          # CompShare 镜像池筛选
```

> ⚠️ **不要把私钥粘贴到对话、issue、截图或提交信息里。**
> 一旦贴出即应视为泄露，去控制台吊销重新生成。

## 6. 成本控制纪律

- **按量计费 + 关机不收费**（官网明示；镜像/云盘存储可能仍计费，需实测确认）
- 训练脚本必须支持 checkpoint 续跑，避免掉线重头再来
  —— `scripts/train_pu.py` 每 epoch 刷 `ckpt/last.pt`，已满足
- 跑完立刻关机，不留空转实例
- 每次上云前在 `EXPERIMENT_LOG.md` 写明：本次要跑哪个 run、预计多久、预计花多少
- **按量单价 2.77 元/时**：一次 6.5 h 的完整 run ≈ 18 元；
  10 组消融 ≈ 180 元。这个量级下**不必为省钱牺牲实验完整性**，
  但也不要开着机去 debug 代码 —— debug 一律在本机做。

