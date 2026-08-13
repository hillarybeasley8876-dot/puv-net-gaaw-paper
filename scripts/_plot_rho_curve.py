"""
第 4 章 ρ 曲线图（图 4-1）—— GA-PUT 训练期对抗权重自适应机制的最硬证据。

数据源：runs/rho_trace_B2.json
- 150 epoch 完整训练
- ρ_hat = target_ratio / w_auto，调和口径代表值
- 平台区 [75, 149] n=75
- B1 固定 w_adv=8.27 标横线对比

口径 caveat（必须在图注与正文同时声明）：
  w_auto 是 epoch 内 batch 的算术平均，ρ_hat 是基于它的代表值，
  不是 mean(rho)。仅可用于数量级演化趋势，不可作单 batch 精确值引用。
"""
import json, os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "runs", "rho_trace_B2.json")
OUTDIR = os.path.join(ROOT, "paper_assets_TRIAL", "figures_ch4")
os.makedirs(OUTDIR, exist_ok=True)

with open(DATA, encoding="utf-8") as f:
    d = json.load(f)

epochs = [r["epoch"] for r in d["rows"]]
rho = [r["rho_hat"] for r in d["rows"]]
w_auto = [r["w_auto"] for r in d["rows"]]
warmup = [r["warmup"] for r in d["rows"]]

# 平台区上下标底色
plat_lo, plat_hi = d["plateau"][0], d["plateau"][1]

# 固定权重对照：B1 的 w_adv=8.27 -> rho = target_ratio / 0.1 / 8.27 = 0.01209
rho_fixed = d["doc_calibration"]["rho"]
b2_below = d["doc_calibration"]["b2_frac_below"]

# 平台区中位 rho
plat_rho_median = d["stats_rho_plateau"]["median"]
plat_rho_p10 = d["stats_rho_plateau"]["p10"]
plat_rho_p90 = d["stats_rho_plateau"]["p90"]

# === 稳健性证据（对应 scripts/_verify_h1_robustness.py 的 R1/R2/R3）===
# 目的：让「阶段性单调演化」肉眼可见，使 H1 不再仅依赖 max/min 极差。
SEG = 25
seg_edges = list(range(0, 150, SEG))
seg_med = []
for s in seg_edges:
    v = [r["rho_hat"] for r in d["rows"] if s <= r["epoch"] < s + SEG]
    seg_med.append(st.median(v))

_srt = sorted(rho)


def _q(p):
    i = p * (len(_srt) - 1)
    lo, hi = int(i), min(int(i) + 1, len(_srt) - 1)
    return _srt[lo] + (_srt[hi] - _srt[lo]) * (i - lo)


rob_span = _q(0.90) / _q(0.10)          # 抗异常点极差
early_med = st.median([r["rho_hat"] for r in d["rows"] if r["epoch"] < 25])
stage_ratio = plat_rho_median / early_med  # 早期 → 平台区

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1.2]})

# === 上图：ρ_hat 演化（log-y） ===
ax1.axvspan(plat_lo, plat_hi, color="#f0f0f0", zorder=0, label="plateau [75, 149]")
ax1.axhline(rho_fixed, color="#d62728", linestyle="--", linewidth=1.2,
            label=r"B1 fixed $\rho=0.0121$ ($w_{adv}=8.27$)")
ax1.semilogy(epochs, rho, color="#1f77b4", linewidth=1.3,
             label=r"GA-PUT (B2) $\hat\rho$, 150-epoch trace")

# 分段中位数阶梯：证明单调演化，不依赖极值点（Spearman rho = 1.000）
step_x, step_y = [], []
for s, m in zip(seg_edges, seg_med):
    step_x += [s, s + SEG]
    step_y += [m, m]
ax1.plot(step_x, step_y, color="#ff7f0e", linewidth=2.0, alpha=0.9,
         solid_capstyle="butt", zorder=5,
         label=r"per-25-epoch median (monotone, Spearman $\rho$=1.000)")

# 平台区中位数横线
ax1.axhline(plat_rho_median, color="#1f77b4", linestyle=":", linewidth=1.0,
            label=f"plateau median $\\hat\\rho$ = {plat_rho_median:.0f}")

ax1.set_ylabel(r"$\hat\rho = r_{\text{target}}/\bar w_{\text{auto}}$  (log scale)")
ax1.set_ylim(1e-3, 3e6)
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(__import__('math').log10(x))}}}$"
                                            if x >= 1 else f"${x:g}$"))
ax1.legend(loc="upper left", fontsize=7.2, frameon=True, framealpha=0.95,
           ncol=2, columnspacing=1.0, handlelength=1.8, borderpad=0.4)
ax1.grid(True, which="both", linestyle=":", alpha=0.4)

# 标注关键跨度（同时给出抗异常点极差，防「极差由离群点刷出」的质疑）
# 放在右下空白区，避免遮挡曲线与图例。
ax1.text(0.985, 0.045,
         "span %.2f\u00d710$^6$  (max/min)\n"
         "robust %.1f\u00d710$^5$  (p90/p10)\n"
         "stage %.1f\u00d710$^5$  (early\u2192plateau)"
         % (3.92388, rob_span / 1e5, stage_ratio / 1e5),
         transform=ax1.transAxes, fontsize=7.2, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", alpha=0.92))

# === 下图：w_auto 演化（log-y） ===
ax2.axvspan(plat_lo, plat_hi, color="#f0f0f0", zorder=0)
ax2.axhline(8.27, color="#d62728", linestyle="--", linewidth=1.2,
            label=r"B1 fixed $w_{adv}=8.27$")
ax2.semilogy(epochs, w_auto, color="#2ca02c", linewidth=1.3,
             label=r"GA-PUT (B2) $\bar w_{\text{auto}}$")
ax2.set_xlabel("epoch")
ax2.set_ylabel(r"$\bar w_{\text{auto}}$  (log scale)")
ax2.set_xlim(0, 149)
ax2.set_ylim(1e-6, 1e2)
ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
ax2.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
ax2.grid(True, which="both", linestyle=":", alpha=0.4)

# 头部小标题与底部 caveat
fig.suptitle("Fig. 4-1  Training-dynamics evidence of gradient-adaptive adversarial weighting",
             fontsize=10, y=0.995)
_CAV = (
    "Caveat 1 (measurement): $\\bar w_{\\text{auto}}$ is the per-epoch arithmetic mean over batches; "
    "$\\hat\\rho$ is a representative value derived from it, not mean($\\rho$).\n"
    "This figure characterises order-of-magnitude trends only; do not read single-batch values from it.\n"
    "Caveat 2 (scope of the H1 verdict): H1 is accepted from this single 150-epoch run of "
    "ABL_B2_adv_adaptive (seed 20260811). It establishes that the gradient-norm\n"
    "ratio is non-stationary within this run; it is not evidence of across-seed reproducibility, and no "
    "across-seed variance is reported anywhere in this thesis."
)
# 单个文本块 + 显式换行：两个独立 fig.text 在 bbox_inches="tight" 下会因
# 画布重裁而相互重叠压成一团（实测不可读），故合并为一块并自行控制换行。
fig.text(0.5, 0.004, _CAV, ha="center", va="bottom",
         fontsize=6.8, style="italic", color="#555", linespacing=1.6)

plt.tight_layout(rect=(0, 0.115, 1, 0.97))
out = os.path.join(OUTDIR, "F4_1_rho_curve_B2.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("wrote", out)
print("b2_frac_below_8.27 =", b2_below, "  ->", f"{b2_below*100:.1f}% of epochs")
print("plateau rho_hat: median={:.1f}  p10={:.1f}  p90={:.1f}".format(
    plat_rho_median, plat_rho_p10, plat_rho_p90))
print("B1 fixed rho =", rho_fixed)
print("--- H1 robustness overlay ---")
print("  robust span p90/p10   = %.6g x" % rob_span)
print("  early->plateau ratio  = %.6g x" % stage_ratio)
print("  segment medians       = %s"
      % ", ".join("%.4g" % m for m in seg_med))
