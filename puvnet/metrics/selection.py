# -*- coding: utf-8 -*-
"""多指标综合选点 + 平台区统计。

背景（B-001 实测，见 docs/EXPERIMENT_LOG.md）：
    cd 最优在 ep41 / hd 最优在 ep48 / nuc 最优在 ep98，三者完全错开。
    ep41 的 best.pt 其 NUC=0.612210，比 ep98 的 0.539498 差 13.5%。
    → cd-only 选点有系统性偏差，论文里 uniformity 列会无故难看。

本模块提供两件事：
    1) composite_score()  — 归一化后的综合分，用于选 best checkpoint
    2) plateau_stats()    — 平台区均值±σ，用于论文主表报数

=============================================================================
两个设计红线（都踩过坑，别改）
=============================================================================
红线 1：**禁止信息泄漏**。
    综合分的归一化基准只能用「当前 epoch 及之前」已观测到的值。
    若用全程 min/max 归一化，则 ep10 的得分会依赖 ep99 的数据 —— best.pt
    在训练中途就无法确定，断点续跑结果也不可复现。

红线 2：**尺度基准必须冻结，不能用 running median**。
    ！！这是 2026-08-11 冒烟实测抓出的真 bug，别改回去 ！！
    最初实现用 running median 作尺度。它在平台期没问题（B-001 未暴露），
    但在**单调下降的训练曲线**上会持续滞后于当前值，后果有两个：
      (a) 样本数少时 median == 当前值，rel 恒等于 1.0，
          ep000 与 ep002 拿到一模一样的 score=1.000000，
          而 ep002 三项指标全面变差却抢走了 best；
      (b) 下降期 median 长期高于当前值，rel 被系统性压低，
          **早期 epoch 被优待** —— 8 epoch 冒烟里综合分选了 ep3，
          而三项指标全场最低的是 ep7（cd-only 选对了）。
    修法：warmup 结束时算一次中位数作为尺度，**之后冻结**。
    既只用过去数据（无泄漏），又不随训练进度漂移。

红线 3：**量纲必须分流**。
    cd≈0.002 / hd≈0.008 / nuc≈0.55，跨两个数量级。直接加权 = NUC 独裁
    （它的绝对波动比 CD 大 300 倍）。所以一律先除以各自的冻结尺度，
    转成「相对典型值的倍数」再加权。
    这是 F_metric 混轴 bug 的同源问题，第三次遇到，记牢。

=============================================================================
自检：python -m puvnet.metrics.selection
=============================================================================
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

# 综合分默认权重。三项都是「越小越好」。
# cd 是主指标（论文一号表头），给最大权重；hd 反映最坏局部误差；
# nuc 反映均匀性，是 PU 任务的第二核心诉求，不能给 0。
DEFAULT_WEIGHTS = {"cd": 0.5, "hd": 0.2, "nuc": 0.3}

_EPS = 1e-12


def _median(vals: Sequence[float]) -> float:
    """朴素中位数，避免为此依赖 numpy（本模块要能被极简环境导入）。"""
    s = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not s:
        return float("nan")
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
    s = [v for v in vals if v is not None and math.isfinite(v)]
    if not s:
        return float("nan"), float("nan")
    m = sum(s) / len(s)
    if len(s) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in s) / (len(s) - 1)   # 样本方差 (ddof=1)
    return m, math.sqrt(var)


class CompositeSelector:
    """在线多指标选点器。

    用法（训练循环内，每次验证后调用一次）：
        sel = CompositeSelector()
        ...
        score = sel.update(epoch=ep, cd=..., hd=..., nuc=...)
        if sel.is_best:
            torch.save(...)

    `update` 返回本 epoch 的综合分（越小越好）。`is_best` 表示该分是否为
    截至目前的最小值。
    """

    def __init__(self, weights: dict[str, float] | None = None,
                 warmup: int = 5):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        wsum = sum(self.weights.values())
        if wsum <= 0:
            raise ValueError("权重和必须 > 0")
        # 归一化权重，保证综合分的绝对尺度与权重设置无关（便于跨 run 比较）
        self.weights = {k: v / wsum for k, v in self.weights.items()}
        # warmup：前若干 epoch 指标剧烈变化，尺度基准未定，
        # 此期间不允许出 best（否则 best.pt 常被 ep0 抢走）；
        # warmup 结束的那一刻用已观测值算中位数作为尺度，之后冻结（见红线 2）
        self.warmup = max(int(warmup), 1)
        self._hist: dict[str, list[float]] = {k: [] for k in self.weights}
        self.scales: dict[str, float] | None = None   # 冻结后的尺度基准
        self.records: list[dict] = []
        self.best_score = float("inf")
        self.best_epoch: int | None = None
        self.is_best = False

    def _freeze_scales(self) -> None:
        """用 warmup 期观测值一次性确定尺度，之后不再更新。"""
        sc = {}
        for k, vals in self._hist.items():
            m = _median(vals)
            if not math.isfinite(m) or abs(m) < _EPS:
                m = 1.0
            sc[k] = m
        self.scales = sc

    def update(self, epoch: int, **metrics) -> float | None:
        vals = {}
        for k in self.weights:
            v = metrics.get(k)
            if v is None or not math.isfinite(float(v)):
                # 缺任一指标就不参与选点：宁可不选，也不用残缺分覆盖 best
                self.is_best = False
                self.records.append({"epoch": epoch, "score": None,
                                     "reason": f"缺失指标 {k}"})
                return None
            vals[k] = float(v)
            self._hist[k].append(float(v))

        n_seen = len(self._hist["cd"])
        if self.scales is None:
            if n_seen < self.warmup:
                # 尺度未定，本 epoch 不出分也不出 best
                self.is_best = False
                self.records.append({"epoch": epoch, "score": None,
                                     "raw": vals, "reason": "warmup 期，尺度未冻结"})
                return None
            self._freeze_scales()

        parts, score = {}, 0.0
        for k, v in vals.items():
            rel = v / self.scales[k]             # 相对冻结基准的倍数
            parts[k] = rel
            score += self.weights[k] * rel

        self.is_best = score < self.best_score
        if self.is_best:
            self.best_score = score
            self.best_epoch = epoch

        self.records.append({"epoch": epoch, "score": score,
                             "rel": parts, "raw": vals,
                             "is_best": self.is_best})
        return score

    def summary(self) -> dict:
        return {"weights": self.weights, "warmup": self.warmup,
                "scales": self.scales,
                "best_epoch": self.best_epoch,
                "best_score": (self.best_score
                               if math.isfinite(self.best_score) else None),
                "n_records": len(self.records)}


def plateau_stats(records: Iterable[dict], keys: Sequence[str] = ("cd", "hd", "nuc"),
                  frac: float = 0.5, prefix: str = "monitor_") -> dict:
    """平台区均值±σ —— 论文主表报数唯一出口。

    为什么不报最优单点：B-001 的 ep41 cd=0.002024 相对 ep50-99 平台
    （均值 0.002166 / σ 0.000050）低 6.6% ≈ −2.8σ，是一次幸运波动。
    拿它当 baseline 等于给自己的改进方法人为设一道高门槛。

    参数
    ----
    frac : 取训练后 frac 比例的 epoch 作为平台区。0.5 = 后半段。
    prefix : records 里的字段前缀（训练期是 "monitor_"，正式评测是 ""）。
    """
    recs = [r for r in records if isinstance(r, dict)]
    if not recs:
        return {"error": "空记录"}
    n = len(recs)
    start = int(n * (1.0 - frac))
    seg = recs[start:]
    out = {"n_total": n, "plateau_from_index": start,
           "plateau_n": len(seg), "frac": frac,
           "epoch_range": [seg[0].get("epoch"), seg[-1].get("epoch")]}
    for k in keys:
        col = [r.get(prefix + k) for r in seg]
        m, sd = _mean_std(col)
        best_v, best_e = None, None
        for r in recs:
            v = r.get(prefix + k)
            if v is not None and math.isfinite(float(v)):
                if best_v is None or float(v) < best_v:
                    best_v, best_e = float(v), r.get("epoch")
        out[k] = {"plateau_mean": m, "plateau_std": sd,
                  "best": best_v, "best_epoch": best_e,
                  # 最优单点偏离平台多少个 σ —— 用来判断它是不是幸运波动
                  "best_sigma_from_mean": ((best_v - m) / sd
                                           if (best_v is not None and sd
                                               and math.isfinite(sd)
                                               and sd > 0) else None)}
    return out


def convergence_check(records: Iterable[dict], keys: Sequence[str] = ("cd", "hd", "nuc"),
                      window: int = 10, prefix: str = "monitor_") -> dict:
    """收敛判据：比较最后两个 window 的均值变化率。

    B-001 实测 ep70-79 → ep90-99：cd −0.82%（收敛）/ hd −5.27% / nuc −3.84%
    （仍在降）→ 这是把 epoch 从 100 提到 150 的直接依据。
    """
    recs = [r for r in records if isinstance(r, dict)]
    if len(recs) < 2 * window:
        return {"error": f"记录数 {len(recs)} 不足 2×window={2*window}"}
    a, b = recs[-2 * window:-window], recs[-window:]
    out = {"window": window,
           "range_a": [a[0].get("epoch"), a[-1].get("epoch")],
           "range_b": [b[0].get("epoch"), b[-1].get("epoch")]}
    for k in keys:
        ma, _ = _mean_std([r.get(prefix + k) for r in a])
        mb, _ = _mean_std([r.get(prefix + k) for r in b])
        rel = ((mb - ma) / ma * 100.0) if (ma and math.isfinite(ma)
                                           and abs(ma) > _EPS) else None
        out[k] = {"mean_a": ma, "mean_b": mb, "change_pct": rel,
                  # 阈值 1%：变化小于 1% 视为已收敛。依据见上方 B-001 实测。
                  "converged": (rel is not None and abs(rel) < 1.0)}
    return out


# =============================================================================
def self_check() -> bool:
    print("=" * 70)
    print("selection.py 自检")
    print("=" * 70)
    ok = True

    # --- 1. 量纲分流：NUC 不得因绝对值大而独裁 ---
    # 构造：cd 显著变好，nuc 基本不动。综合分应该跟着 cd 走。
    sel = CompositeSelector(warmup=0)
    for ep in range(20):
        cd = 0.0030 - ep * 0.00005      # 0.0030 -> 0.00205，降 31%
        sel.update(epoch=ep, cd=cd, hd=0.008, nuc=0.55)
    good = sel.best_epoch == 19
    print(f"[{'PASS' if good else 'FAIL'}] 量纲分流：cd 单调下降 -> best_epoch="
          f"{sel.best_epoch} (期望 19)")
    ok &= good

    # 反向：nuc 显著变好，cd 不动 -> 也应能选到
    sel2 = CompositeSelector(warmup=0)
    for ep in range(20):
        sel2.update(epoch=ep, cd=0.002, hd=0.008, nuc=0.70 - ep * 0.01)
    good = sel2.best_epoch == 19
    print(f"[{'PASS' if good else 'FAIL'}] 量纲分流：nuc 单调下降 -> best_epoch="
          f"{sel2.best_epoch} (期望 19)")
    ok &= good

    # --- 2. 无信息泄漏：截断历史不改变已定的 best ---
    full = CompositeSelector(warmup=0)
    seq = [(0.0030, 0.010, 0.60), (0.0025, 0.009, 0.58), (0.0021, 0.008, 0.56),
           (0.0022, 0.0085, 0.57), (0.0020, 0.0078, 0.55), (0.0023, 0.009, 0.58)]
    for ep, (c, h, u) in enumerate(seq):
        full.update(epoch=ep, cd=c, hd=h, nuc=u)
    trunc = CompositeSelector(warmup=0)
    for ep, (c, h, u) in enumerate(seq[:3]):
        trunc.update(epoch=ep, cd=c, hd=h, nuc=u)
    # 前 3 个 epoch 的分数必须逐字节一致（不受后续数据影响）
    same = all(abs(full.records[i]["score"] - trunc.records[i]["score"]) < 1e-15
               for i in range(3))
    print(f"[{'PASS' if same else 'FAIL'}] 无信息泄漏：前 3 epoch 分数与截断运行一致")
    ok &= same

    # --- 3. B-001 真实错开场景：综合分不应选在 nuc 极差的 ep41 ---
    # 简化重演：ep41 cd 极好但 nuc 差 13.5%；ep98 cd 略差但 nuc 最好
    sel3 = CompositeSelector(warmup=0)
    for ep in range(100):
        cd = 0.002166
        nuc = 0.5684
        if ep == 41:
            cd, nuc = 0.002024, 0.612210       # cd −6.6%，nuc +7.7%
        if ep == 98:
            cd, nuc = 0.002159, 0.539498       # cd −0.3%，nuc −5.1%
        sel3.update(epoch=ep, cd=cd, hd=0.008, nuc=nuc)
    good = sel3.best_epoch == 98
    print(f"[{'PASS' if good else 'FAIL'}] B-001 重演：best_epoch={sel3.best_epoch} "
          f"(期望 98，即不被 ep41 的 cd 谷底带偏)")
    ok &= good

    # --- 4. 缺失指标不得覆盖 best ---
    sel4 = CompositeSelector(warmup=0)
    sel4.update(epoch=0, cd=0.003, hd=0.01, nuc=0.6)
    sel4.update(epoch=1, cd=0.002, hd=0.008, nuc=0.55)
    be = sel4.best_epoch
    s = sel4.update(epoch=2, cd=None, hd=0.001, nuc=0.1)   # 缺 cd
    good = (s is None) and (not sel4.is_best) and (sel4.best_epoch == be)
    print(f"[{'PASS' if good else 'FAIL'}] 缺失指标：返回 None 且 best_epoch 不变"
          f"({sel4.best_epoch})")
    ok &= good

    # --- 5. warmup 生效 ---
    sel5 = CompositeSelector(warmup=5)
    for ep in range(4):
        sel5.update(epoch=ep, cd=0.001, hd=0.001, nuc=0.1)   # 极好但在 warmup 内
    good = sel5.best_epoch is None
    print(f"[{'PASS' if good else 'FAIL'}] warmup=5：前 4 epoch 不出 best "
          f"(best_epoch={sel5.best_epoch})")
    ok &= good

    # --- 5b. 回归：单调下降曲线必须选末轮（running median bug 的回归用例）---
    # 2026-08-11 冒烟实测：running median 作尺度时，此场景会错选 ep3。
    # 原因见模块 docstring 红线 2。修复后必须选 ep7（三项全场最低）。
    smoke = [(0.057753, 0.296146, 49.020770), (0.026277, 0.144287, 24.182458),
             (0.041057, 0.218854, 43.962146), (0.018961, 0.106452, 10.752345),
             (0.017190, 0.093651, 12.682869), (0.014407, 0.075460, 9.817337),
             (0.011577, 0.056459, 5.948643), (0.009947, 0.054421, 4.501305)]
    sel5b = CompositeSelector(warmup=2)
    for ep, (c, h, u) in enumerate(smoke):
        sel5b.update(epoch=ep, cd=c, hd=h, nuc=u)
    good = sel5b.best_epoch == 7
    print(f"[{'PASS' if good else 'FAIL'}] 单调下降回归：best_epoch={sel5b.best_epoch} "
          f"(期望 7 = 三项全场最低；running median 版会错选 3)")
    ok &= good

    # 同场景下不得出现两个 epoch 同分（median==当前值导致 rel 恒为 1.0 的症状）
    scores = [r["score"] for r in sel5b.records if r.get("score") is not None]
    good = len(scores) == len(set(scores))
    print(f"[{'PASS' if good else 'FAIL'}] 无同分退化：{len(scores)} 个分数互不相同")
    ok &= good

    # 尺度确实被冻结
    sc = sel5b.scales
    good = sc is not None and all(math.isfinite(v) and v > 0 for v in sc.values())
    print(f"[{'PASS' if good else 'FAIL'}] 尺度已冻结：{ {k: round(v, 6) for k, v in sc.items()} }")
    ok &= good

    # --- 6. plateau_stats 复现 B-001 已知数字 ---
    recs = [{"epoch": e, "monitor_cd": 0.002166, "monitor_hd": 0.008091,
             "monitor_nuc": 0.568401} for e in range(100)]
    recs[41]["monitor_cd"] = 0.002024
    st = plateau_stats(recs, frac=0.5)
    good = (st["plateau_n"] == 50 and st["epoch_range"] == [50, 99]
            and abs(st["cd"]["plateau_mean"] - 0.002166) < 1e-9
            and st["cd"]["best_epoch"] == 41)
    print(f"[{'PASS' if good else 'FAIL'}] plateau_stats：后 50 epoch (ep{st['epoch_range'][0]}"
          f"-{st['epoch_range'][1]})，cd 平台均值={st['cd']['plateau_mean']:.6f}，"
          f"最优 ep{st['cd']['best_epoch']}")
    ok &= good

    # σ 计算与「幸运波动」判据
    recs2 = [{"epoch": e, "monitor_cd": 0.002166 + (e % 5 - 2) * 0.00005}
             for e in range(100)]
    recs2[41]["monitor_cd"] = 0.002024
    st2 = plateau_stats(recs2, keys=("cd",), frac=0.5)
    sig = st2["cd"]["best_sigma_from_mean"]
    good = sig is not None and sig < -1.0
    print(f"[{'PASS' if good else 'FAIL'}] 幸运波动判据：best 偏离平台 {sig:.2f}σ (应显著为负)")
    ok &= good

    # --- 7. convergence_check 复现 B-001 结论 ---
    # 造 cd 已收敛、nuc 仍在降
    recs3 = []
    for e in range(100):
        cd = 0.00220 if e < 90 else 0.002182           # −0.82%
        nuc = 0.5735 if e < 90 else 0.5515             # −3.84%
        recs3.append({"epoch": e, "monitor_cd": cd, "monitor_nuc": nuc})
    cv = convergence_check(recs3, keys=("cd", "nuc"), window=10)
    good = (cv["cd"]["converged"] is True and cv["nuc"]["converged"] is False)
    print(f"[{'PASS' if good else 'FAIL'}] 收敛判据：cd 变化 {cv['cd']['change_pct']:.2f}% "
          f"(收敛={cv['cd']['converged']}) / nuc 变化 {cv['nuc']['change_pct']:.2f}% "
          f"(收敛={cv['nuc']['converged']})")
    ok &= good

    # --- 8. 权重归一化不改变排序 ---
    a = CompositeSelector(weights={"cd": 1, "hd": 1, "nuc": 1}, warmup=0)
    b = CompositeSelector(weights={"cd": 10, "hd": 10, "nuc": 10}, warmup=0)
    for ep, (c, h, u) in enumerate(seq):
        a.update(epoch=ep, cd=c, hd=h, nuc=u)
        b.update(epoch=ep, cd=c, hd=h, nuc=u)
    good = a.best_epoch == b.best_epoch and abs(a.best_score - b.best_score) < 1e-12
    print(f"[{'PASS' if good else 'FAIL'}] 权重等比缩放：best_epoch/score 一致 "
          f"({a.best_epoch}/{b.best_epoch})")
    ok &= good

    print("=" * 70)
    print("ALL PASS" if ok else "存在 FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
