"""
Compression summary diagram for ResNet-18 on WM-811K.
Compares: dense → pruned → pruned+quantised across F1 and model size.

Output: results/plots/resnet18_compression_summary.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
variants = [
    "Dense FP32\n(Optuna-tuned)",
    "Pruned 76.6%\n(FP32)",
    "Pruned 76.6%\n+ Static INT8",
]
f1_scores  = [0.9007, 0.9020, 0.9035]
sizes_mb   = [42.72,  42.72,  10.79]
colors_bar = ["#4C72B0", "#DD8452", "#55A868"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ResNet-18 Compression Summary — WM-811K Wafer Defect Detection",
             fontsize=13, fontweight="bold", y=1.01)

x = np.arange(len(variants))
bar_w = 0.5

# ── Panel 1: Macro F1 ─────────────────────────────────────────────────────────
bars1 = ax1.bar(x, [v * 100 for v in f1_scores], width=bar_w,
                color=colors_bar, edgecolor="k", linewidth=0.6, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=9)
ax1.set_ylabel("Test Macro F1 (%)", fontsize=10)
ax1.set_ylim(89.5, 91.5)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_title("Macro F1 (test set)", fontsize=11)

for bar, val in zip(bars1, f1_scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Annotate improvement arrows
for i in range(1, len(f1_scores)):
    delta = (f1_scores[i] - f1_scores[i - 1]) * 100
    mid_x = (x[i - 1] + x[i]) / 2
    ax1.annotate(f"+{delta:.2f}%",
                 xy=(mid_x, min(f1_scores[i - 1], f1_scores[i]) * 100 - 0.1),
                 ha="center", va="top", fontsize=8, color="green",
                 fontweight="bold")

# ── Panel 2: Model size ───────────────────────────────────────────────────────
bars2 = ax2.bar(x, sizes_mb, width=bar_w,
                color=colors_bar, edgecolor="k", linewidth=0.6, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(variants, fontsize=9)
ax2.set_ylabel("Model Size (MB)", fontsize=10)
ax2.set_ylim(0, 52)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.set_title("Model Size (MB)", fontsize=11)

for bar, val in zip(bars2, sizes_mb):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.2f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Size reduction annotation
ax2.annotate("", xy=(x[2], sizes_mb[2] + 1), xytext=(x[0], sizes_mb[0] + 1),
             arrowprops=dict(arrowstyle="<->", color="tomato", lw=1.5))
ax2.text((x[0] + x[2]) / 2, sizes_mb[0] + 2.5,
         f"4× smaller\n({(1 - sizes_mb[2]/sizes_mb[0])*100:.0f}% reduction)",
         ha="center", va="bottom", fontsize=9, color="tomato", fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_bar, variants)]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
out_path = Path(__file__).parent.parent / "results" / "plots" / "resnet18_compression_summary.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] → {out_path}")
