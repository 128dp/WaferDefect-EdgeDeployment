"""
Generates a clear 3-metric comparison table + bubble chart
explaining Model Size, GFLOPs, and Latency for presentation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
models   = ["ResNet-18", "MobileNetV2", "EfficientNet-B0", "ShuffleNetV2"]
colors   = ["#4C72B0", "#DD8452", "#C44E52", "#8172B2"]
params_m = [11.18,  2.23,  4.02,  0.35]   # millions of parameters
size_mb  = [42.72,  8.75, 15.61,  1.47]   # model file size (MB)
flops_g  = [0.1489, 0.0266, 0.0344, 0.0036]  # GFLOPs
latency  = [4.82,   4.91,   7.70,   4.81]    # ms on desktop CPU
macro_f1 = [0.913,  0.930,  0.931,  0.899]   # macro F1

x = np.arange(len(models))
bar_w = 0.22

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Model Comparison — WM-811K Wafer Defect Detection\nUnderstanding Parameters, Size, GFLOPs, and Latency",
             fontsize=14, fontweight="bold", y=0.98)

# ── Top-left: Parameters (what they are) ──────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
bars = ax1.bar(models, params_m, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars, params_m):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v}M", ha="center", fontsize=9, fontweight="bold")
ax1.set_title("① Parameters (M)\n= Number of learnable weights", fontsize=10, fontweight="bold")
ax1.set_ylabel("Millions of Parameters")
ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ax1.set_ylim(0, 14)
ax1.annotate("More params =\nbigger model,\nmore expressive",
             xy=(0, 11.18), xytext=(1.5, 12.5),
             arrowprops=dict(arrowstyle="->", color="gray"),
             fontsize=8, color="gray")

# ── Top-middle: Model Size ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
bars2 = ax2.bar(models, size_mb, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars2, size_mb):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v} MB", ha="center", fontsize=9, fontweight="bold")
# Draw storage limit lines
ax2.axhline(y=2,  color="red",    linestyle="--", linewidth=1.2, label="MCU limit (~2 MB)")
ax2.axhline(y=10, color="orange", linestyle="--", linewidth=1.2, label="Mobile limit (~10 MB)")
ax2.set_title("② Model Size (MB)\n= Storage / RAM needed on device", fontsize=10, fontweight="bold")
ax2.set_ylabel("Size (MB)")
ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ax2.legend(fontsize=7, loc="upper right")
ax2.set_ylim(0, 50)

# ── Top-right: GFLOPs ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
bars3 = ax3.bar(models, [f*1000 for f in flops_g], color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars3, flops_g):
    ax3.text(b.get_x() + b.get_width()/2, v*1000 + 0.5,
             f"{v*1000:.1f}M", ha="center", fontsize=9, fontweight="bold")
ax3.set_title("③ GFLOPs (compute cost)\n= Arithmetic ops per image (hardware-agnostic)", fontsize=10, fontweight="bold")
ax3.set_ylabel("MFLOPs (millions)")
ax3.set_xticks(x); ax3.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ratio = flops_g[0] / flops_g[3]
ax3.annotate(f"ShuffleNetV2 needs\n{ratio:.0f}× less compute\nthan ResNet-18",
             xy=(3, flops_g[3]*1000), xytext=(1.8, 80),
             arrowprops=dict(arrowstyle="->", color="green"),
             fontsize=8, color="green", fontweight="bold")

# ── Bottom-left: Latency ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
bars4 = ax4.bar(models, latency, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars4, latency):
    ax4.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v} ms", ha="center", fontsize=9, fontweight="bold")
ax4.set_title("④ Inference Latency (ms)\n= Actual time per image (desktop CPU, 1 thread)", fontsize=10, fontweight="bold")
ax4.set_ylabel("Latency (ms)")
ax4.set_xticks(x); ax4.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ax4.set_ylim(0, 10)
ax4.annotate("* Low GFLOPs ≠ fast on desktop\n  (depthwise conv not GPU-optimized)\n  On ARM chip: ShuffleNetV2 fastest",
             xy=(0.5, 0.15), xycoords="axes fraction", fontsize=8, color="gray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# ── Bottom-middle: Macro F1 ───────────────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
bars5 = ax5.bar(models, macro_f1, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars5, macro_f1):
    ax5.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax5.set_title("⑤ Macro F1 Score\n= Accuracy across all 8 defect classes equally", fontsize=10, fontweight="bold")
ax5.set_ylabel("Macro F1")
ax5.set_xticks(x); ax5.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
ax5.set_ylim(0.85, 0.95)

# ── Bottom-right: Summary table ───────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
table_data = [
    ["ResNet-18",       "11.2 M", "42.7 MB", "148.9 M", "4.8 ms", "0.913", "☁ Cloud/Server"],
    ["MobileNetV2",     " 2.2 M", " 8.8 MB", " 26.6 M", "4.9 ms", "0.930", "📱 Smartphone"],
    ["EfficientNet-B0", " 4.0 M", "15.6 MB", " 34.4 M", "7.7 ms", "0.931", "📱 Edge Server"],
    ["ShuffleNetV2",    " 0.35M", " 1.5 MB", "  3.6 M", "4.8 ms", "0.899", "⚙ MCU / IoT"],
]
col_labels = ["Model", "Params", "Size", "FLOPs", "Latency", "F1", "Target Device"]
tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.8)
# Colour header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# Highlight ShuffleNetV2 row
for j in range(len(col_labels)):
    tbl[4, j].set_facecolor("#EAF4E8")
ax6.set_title("Summary Table", fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "results/plots/metrics_explained.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[Plot] Saved → {out}")
