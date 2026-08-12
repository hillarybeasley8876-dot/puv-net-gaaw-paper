# PU-GCN 官方评测协议 —— 一手代码证据

> 抓取时间：2026-08-11
> 仓库：`https://github.com/guochengqian/PU-GCN` (branch `master`, tree sha `0a8daac57dbda037689d797c679d3540ee253317`)
> 抓取方式：`api.github.com/.../git/trees/master?recursive=1` 取文件树 → `raw.githubusercontent.com` 取 blob 原文
> 此文件用途：作为 `puvnet/metrics/pointcloud.py` 指标协议的**唯一权威依据**。任何协议变更必须先更新此文件。

---

## 0. 为什么必须查一手代码

`puvnet/metrics/pointcloud.py` 原先在 `chamfer_distance` 的 docstring 里写了：

```
⚠️ 本项目主表统一用 squared=False（L2 距离），与 PU-GCN evaluate.py 一致。
```

**这条注释是未核实就写下的，且是错的。** 它导致 E-000 校准跑出的所谓"作弊上界" CD=11.39e-3
比文献 SOTA 0.451e-3 差 25 倍，一度被误判为评测管线有 bug。

教训：看起来有依据的注释（"与 XXX 一致"）会让后来所有人不再去查，是最危险的一类错误。

---

## 1. CD 定义（决定性证据）

### 1.1 距离本体是**平方距离**，不开方

`tf_ops/nn_distance/tf_nndistance_cpu.py`（与 CUDA 版 `tf_nndistance_g.cu` 语义一致）：

```python
def nn_distance_cpu(pc1, pc2):
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1,2), [1,1,M,1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2,1), [1,N,1,1])
    pc_diff = pc1_expand_tile - pc2_expand_tile          # B,N,M,C
    pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1)       # B,N,M   <-- 平方距离，无 sqrt
    dist1 = tf.reduce_min(pc_dist, axis=2)               # B,N
    dist2 = tf.reduce_min(pc_dist, axis=1)               # B,M
    return dist1, idx1, dist2, idx2
```

该文件自带的 `verify_nn_distance_cup()` 用 `np.sum((a-b)**2)` 做参照，**再次确认无开方**。

### 1.2 聚合方式是**双向 mean 求和**

`evaluate.py`：

```python
cd_forward, _, cd_backward, _ = tf_nndistance.nn_distance(pred_tensor, gt_tensor)
...
cd_backward_value = np.mean(cd_backward_value)
cd_forward_value  = np.mean(cd_forward_value)
row["CD"] = cd_forward_value + cd_backward_value
```

### 1.3 结论

```
CD_official(pred, gt) = mean_{x in pred} min_{y in gt} ||x-y||^2
                      + mean_{y in gt}   min_{x in pred} ||y-x||^2
```

论文报告值 = `CD_official * 1000`（`evaluate.py` 末尾 `row["CD (1e-3)"] = avg_cd_value*1000.`）。

**本项目原实现（L2 + 双向 mean 求和）与此不符 → 必须改为 squared。**

---

## 2. HD 定义

`Common/metrics.py`：

```python
def hausdorff_from_nn_distances(forward_distances, backward_distances):
    """...The symmetric Hausdorff distance is the maximum of
    the two directional maxima, not the sum of the two maxima."""
    return float(max(np.max(forward), np.max(backward)))
```

`evaluate.py` 传入的 `cd_forward_value / cd_backward_value` 就是 §1.1 的**平方距离数组**：

```python
hd_value = hausdorff_from_nn_distances(cd_forward_value, cd_backward_value)
```

### 结论

```
HD_official(pred, gt) = max( max_x min_y ||x-y||^2 ,  max_y min_x ||y-x||^2 )
```

**HD 与 CD 同处平方距离量纲。** 本项目原实现用 L2 距离的 max → 量纲与 CD 不一致，也与文献不一致，必须改。

---

## 3. 归一化：pred 与 gt **各自独立**归一化

`evaluate.py`：

```python
pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_placeholder)
gt_tensor,   centroid, furthest_distance = normalize_point_cloud(gt_placeholder)
```

两次调用，返回的 `centroid / furthest_distance` 被后一次覆盖且从未使用。
即**评测阶段 pred 和 gt 分别用自己的质心与最远距离归一化**。

> ⚠️ 注意区分两个阶段：
> - **训练阶段**（`puvnet/data/pu_dataset.py`）：input 与 gt **必须共享** gt 的 center/scale，
>   否则监督信号本身错位。这一点已由 dataset 自检 [6] 用 CD 对照验证过，保持不变。
> - **评测阶段**（`scripts/evaluate.py`）：按官方协议 pred / gt **各自独立**归一化。
>
> 这两者不矛盾：评测比的是两个已完成点云的形状一致性，官方协议如此，为了对表必须照做。

`Common/ops.py::normalize_point_cloud` 的语义（按 PU-GAN/PU-GCN 系列惯例）：中心化 + 除以最远点距离，
与本项目 `puvnet/metrics/pointcloud.py::normalize_point_cloud` 一致（最大半径而非包围盒对角线）。

---

## 4. P2F：官方由 CGAL C++ 单独算，不在 Python 里

`evaluation_code/evaluation.cpp` 实测确认：该程序**只输出 point2surface 距离**，
不计算 CD / HD / uniformity。

```cpp
Face_location location = shortest_paths.locate<AABB_face_graph_traits>(pred_points[i], tree);
pred_map_points[i] = shortest_paths.point(location.first, location.second);
const double distance = CGAL::sqrt(CGAL::squared_distance(pred_points[i], pred_map_points[i]));
nearest_distance.push_back(static_cast<float>(distance));
...
distace_output << ... << distance << std::endl;   // 写入 *_point2mesh_distance.xyz
```

- **P2F 是 L2 距离（开方了）**，注意与 CD/HD 的平方量纲不同。
- `evaluate.py` 读该文件第 4 列，取 `np.nanmean` / `np.nanstd`，再 `*1000`。
- **P2F 计算发生在原始坐标尺度上**（`evaluation.cpp` 直接读 mesh 的 `.off` 与预测 `.xyz`，
  中间无任何归一化）。→ 本项目 `scripts/evaluate.py` 算 P2F 时也必须用**未归一化**的
  预测点与原始 mesh。

### 命令行

```
./evaluation mesh_path prediction_path
```

多线程宏 `#define THREAD 40`；`calculate_density` 用于 uniformity 的 disk 统计（本项目未复现）。

---

## 5. Uniformity：官方需要 CGAL 侧产出的 disk 文件

`evaluate.py::analyze_uniform` 依赖三个由 C++ 侧生成的文件：

```python
idx_file        = pred_path[:-4] + "_disk_idx.txt"
radius_file     = pred_path[:-4] + '_radius.txt'
map_points_file = pred_path[:-4] + '_point2mesh_distance.txt'
```

且 `precentages = np.array([0.008, 0.012])`（**0.8% 与 1.2%**，只有两档，不是五档）。

核心公式：

```python
coverage   = np.square(density - expect_number) / expect_number      # expect = p * N
disk_area  = math.pi * (radius[j] ** 2) / map_point.shape[0]
expect_d   = math.sqrt(2 * disk_area / 1.732)                        # 1.732 ≈ sqrt(3)
dis        = np.square(shortest_dis - expect_d) / expect_d           # shortest_dis = 1-NN 距离
uniform_dis.append(coverage * dis_mean)
uniform_measure[j,0] = np.mean(uniform_dis)
```

即官方 uniformity = **coverage 项 × 点间距偏差项** 的乘积，在 1000 个测地 disk 上取均值，
且 disk 内点是**投影到 mesh 表面后的点**（`points = load(map_points_file)[:, 4:]`），
disk 内成员由**测地距离**判定（`Surface_mesh_shortest_path`）。

### 对本项目的影响

本项目 `uniformity_nuc` 是**简化实现**（欧氏球邻域 + 已有点作种子 + 五档半径 + 只用比值方差），
与官方公式**结构不同**（缺 coverage×spacing 乘积、缺测地距离、缺表面投影、档位不同）。

→ **结论不变且必须写进论文 Limitations：本项目 NUC 只可用于本文内部方法间相对比较，
绝不可与文献 uniformity 绝对值对表。** 若必须对表，需在 Linux 上编译 CGAL 评测程序。

---

## 6. 对本项目的修改清单（本文件驱动）

| 项 | 原实现 | 官方协议 | 处理 |
|---|---|---|---|
| CD 距离本体 | L2（开方） | **平方距离** | 改默认 `squared=True` |
| CD 聚合 | 双向 mean 求和 | 双向 mean 求和 | 一致，保留 |
| HD 距离本体 | L2（开方） | **平方距离** | 改默认 `squared=True` |
| HD 聚合 | max(两方向 max) | max(两方向 max) | 一致，保留 |
| 评测归一化 | （原假设共享） | **pred/gt 各自独立** | `scripts/evaluate.py` 改为独立归一化 |
| P2F 距离 | L2 | L2 | 一致 |
| P2F 坐标系 | 需确认 | **原始尺度，不归一化** | 明确固定 |
| Uniformity | 简化欧氏版 | CGAL 测地 disk 版 | 保留简化版，只做相对比较，写 Limitations |

---

## 7. 待验证推论（尚未实测，不得当作结论）

- `debug_cd_scale.py` 曾测得同一样本 squared 版归一化 CD = **0.0966e-3**，
  与文献 SOTA 0.451e-3 同量级 → 与本文件 §1 结论方向一致。
  但那只是**单样本 + gt_half 构造**，不构成校准通过。
- 校准是否真正通过，取决于**改协议后重跑 E-000**：性能上界必须优于 0.451e-3，
  且 baseline 复现（B-001）需接近文献值。未跑完前不得声称"协议已对齐"。
