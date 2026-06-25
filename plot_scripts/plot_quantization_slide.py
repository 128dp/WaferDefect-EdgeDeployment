"""
Generates a clean quantization bar chart: FP32 vs INT8 values only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("INT8 Post-Training Quantization — ResNet-18", fontsize=13, fontweight="bold")

metrics   = ["Accuracy (%)", "Macro F1 × 100", "Size (MB)"]
fp32_vals = [93.59,           91.16,             42.72]
int8_vals = [93.57,           91.13,             42.71]

x = np.arange(len(metrics))
w = 0.32
b1 = ax.bar(x - w/2, fp32_vals, w, label="FP32", color="#4C72B0", edgecolor="white")
b2 = ax.bar(x + w/2, int8_vals, w, label="INT8", color="#DD8452", edgecolor="white")

for bar, val in zip(b1, fp32_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val}", ha="center", fontsize=10, fontweight="bold", color="#4C72B0")
for bar, val in zip(b2, int8_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val}", ha="center", fontsize=10, fontweight="bold", color="#DD8452")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Value", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "results/plots/quantization_slide.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[Plot] Saved → {out}")
