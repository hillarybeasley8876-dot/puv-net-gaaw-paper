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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1.2]})

# === 上图：ρ_hat 演化（log-y） ===
ax1.axvspan(plat_lo, plat_hi, color="#f0f0f0", zorder=0, label="plateau [75, 149]")
ax1.axhline(rho_fixed, color="#d62728", linestyle="--", linewidth=1.2,
            label=r"B1 fixed $\rho=0.0121$ ($w_{adv}=8.27$)")
ax1.semilogy(epochs, rho, color="#1f77b4", linewidth=1.3,
             label=r"GA-PUT (B2) $\hat\rho$, 150-epoch trace")
ax1.scatter([plat_rho_p10], [plat_rho_p10], color="#1f77b4", s=0)  # noop, placeholders
# 平台区中位数横线
ax1.axhline(plat_rho_median, color="#1f77b4", linestyle=":", linewidth=1.0,
            label=f"plateau median $\\hat\\rho$ = {plat_rho_median:.0f}")

ax1.set_ylabel(r"$\hat\rho = r_{\text{target}}/\bar w_{\text{auto}}$  (log scale)")
ax1.set_ylim(1e-3, 1e5)
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(__import__('math').log10(x))}}}$"
                                            if x >= 1 else f"${x:g}$"))
ax1.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
ax1.grid(True, which="both", linestyle=":", alpha=0.4)

# 标注关键跨度
ax1.annotate(f"3.92×10⁶× span\n[min 0.006 → max 2.26×10⁴]",
             xy=(75, 1e3), xytext=(20, 1e4),
             fontsize=8, ha="left",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

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
fig.text(0.5, 0.01,
         "Caveat: $\\bar w_{\\text{auto}}$ is the per-epoch arithmetic mean over batches; "
         "$\\hat\\rho$ is a representative value derived from it, not mean($\\rho$). "
         "This figure characterises order-of-magnitude trends only; do not read single-batch values from it.",
         ha="center", fontsize=7, style="italic", color="#555", wrap=True)

plt.tight_layout(rect=(0, 0.04, 1, 0.97))
out = os.path.join(OUTDIR, "F4_1_rho_curve_B2.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("wrote", out)
print("b2_frac_below_8.27 =", b2_below, "  ->", f"{b2_below*100:.1f}% of epochs")
print("plateau rho_hat: median={:.1f}  p10={:.1f}  p90={:.1f}".format(
    plat_rho_median, plat_rho_p10, plat_rho_p90))
print("B1 fixed rho =", rho_fixed)
