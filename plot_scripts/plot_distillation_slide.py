"""
Knowledge distillation comparison chart for presentation slide.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models   = ["EfficientNet-B0\n(Teacher)", "ShuffleNetV2\n(Standalone)", "ShuffleNetV2\n(Distilled)"]
f1       = [0.9213, 0.8743, 0.8926]
size_mb  = [15.61,  1.47,   1.47]
colors   = ["#C44E52", "#DD8452", "#2CA02C"]

x = np.arange(len(models))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Knowledge Distillation — EfficientNet-B0 Teacher -> ShuffleNetV2 Student",
             fontsize=13, fontweight="bold")

# ── Left: Macro F1 ────────────────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, f1, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, f1):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0.84, 0.94)
ax.set_ylabel("Macro F1", fontsize=11)
ax.set_title("Macro F1 Score", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# ── Right: Model size ─────────────────────────────────────────────────────────
ax2 = axes[1]
bars2 = ax2.bar(x, size_mb, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars2, size_mb):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val} MB", ha="center", fontsize=11, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylabel("Model Size (MB)", fontsize=11)
ax2.set_title("Model Size (MB)", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "results/plots/distillation_efficientnet_slide.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[Plot] Saved -> {out}")
