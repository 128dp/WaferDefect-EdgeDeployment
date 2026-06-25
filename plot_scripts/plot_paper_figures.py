"""
Generate the three paper figures that were missing or outdated:
  1. Pareto curve  — test macro F1 vs ONNX latency, bubble size = params
  2. ONNX latency  — grouped bar chart, FP32 vs ORT for all 5 models
  3. Distillation  — bar chart comparing standalone / R18-distilled / EffNet-distilled

Run from the project root:
    python plot_scripts/plot_paper_figures.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared data ───────────────────────────────────────────────────────────────

MODELS = ["ResNet-34", "EfficientNet-B0", "MobileNetV2", "ResNet-18", "ShuffleNetV2"]
PARAMS  = [21.29,  4.02,  2.23, 11.18, 0.35]           # millions
TEST_F1 = [0.9204, 0.9153, 0.9146, 0.9007, 0.8926]     # test macro F1
FP32_MS = [11.986, 14.285, 11.588,  8.715, 9.745]      # PyTorch CPU latency
ORT_MS  = [10.091,  1.741,  0.724,  5.361, 0.184]      # ONNX Runtime latency
COLORS  = ["#e07b54", "#4c9be8", "#56b06e", "#9b7dd4", "#e8b84b"]

# ── 1. Pareto Curve ───────────────────────────────────────────────────────────

def plot_pareto():
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (name, params, f1, ort) in enumerate(zip(MODELS, PARAMS, TEST_F1, ORT_MS)):
        size = max(80, params * 18)          # bubble area proportional to params
        ax.scatter(ort, f1 * 100, s=size, color=COLORS[i], alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)

        # Label offsets — avoid overlap
        offsets = {
            "ResNet-34":       ( 0.3,  0.02),
            "EfficientNet-B0": ( 0.1, -0.35),
            "MobileNetV2":     ( 0.05, 0.15),
            "ResNet-18":       ( 0.2,  0.10),
            "ShuffleNetV2":    ( 0.1, -0.32),
        }
        dx, dy = offsets[name]
        ax.annotate(
            f"{name}\n({params}M)",
            xy=(ort, f1 * 100),
            xytext=(ort + dx, f1 * 100 + dy),
            fontsize=8.5,
            ha="left",
        )

    # Bubble legend for size reference
    for ref_p, label in [(1, "1M"), (5, "5M"), (20, "20M")]:
        ax.scatter([], [], s=max(80, ref_p * 18), color="grey", alpha=0.5,
                   label=f"{label} params")

    ax.set_xlabel("ONNX Runtime CPU Latency (ms / image, single-threaded)", fontsize=10)
    ax.set_ylabel("Test Macro F1 (%)", fontsize=10)
    ax.set_title("Accuracy–Latency Trade-off (bubble size ∝ parameters)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(88.5, 93.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(title="Model size", fontsize=8, title_fontsize=8,
              loc="lower right", framealpha=0.8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT / "pareto_curve_final.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── 2. ONNX Latency Bar Chart ─────────────────────────────────────────────────

def plot_onnx_latency():
    # Sort by FP32 latency descending for visual clarity
    order = sorted(range(len(MODELS)), key=lambda i: FP32_MS[i], reverse=True)
    names  = [MODELS[i]  for i in order]
    fp32   = [FP32_MS[i] for i in order]
    ort    = [ORT_MS[i]  for i in order]
    speedups = [fp32[j] / ort[j] for j in range(len(names))]

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, fp32, w, label="PyTorch FP32 (CPU)",
                   color="#9b7dd4", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, ort,  w, label="ONNX Runtime (single-thread)",
                   color="#4c9be8", alpha=0.85, edgecolor="white")

    # Annotate speedup above each ORT bar
    for j, (bar, spd) in enumerate(zip(bars2, speedups)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{spd:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4c9be8")

    # Annotate FP32 bars with value
    for bar, val in zip(bars1, fp32):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, color="#9b7dd4")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Inference Latency (ms / image)", fontsize=10)
    ax.set_title("CPU Inference Latency: PyTorch FP32 vs ONNX Runtime",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(0, max(fp32) * 1.22)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Secondary note
    ax.annotate("ORT speedup shown above ONNX bars",
                xy=(0.01, 0.97), xycoords="axes fraction",
                fontsize=8, color="grey", va="top")

    plt.tight_layout()
    path = OUT / "onnx_latency_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── 3. Distillation Comparison Bar Chart ─────────────────────────────────────

def plot_distillation():
    configs = ["Standalone\n(no distillation)", "Distilled\n(ResNet-18 teacher)",
               "Distilled\n(EfficientNet-B0 teacher)"]
    f1s     = [0.8926, 0.8907, 0.9029]
    colors  = ["#9b9b9b", "#e07b54", "#4c9be8"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(configs))
    bars = ax.bar(x, [v * 100 for v in f1s], width=0.5,
                  color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)

    baseline = f1s[0] * 100
    ax.axhline(baseline, color="#9b9b9b", linestyle="--", linewidth=1.2,
               label=f"Standalone baseline ({baseline:.2f}%)")

    for bar, val in zip(bars, f1s):
        diff = val * 100 - baseline
        sign = "+" if diff >= 0 else ""
        label = f"{val*100:.2f}%"
        if diff != 0:
            label += f"\n({sign}{diff:.2f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                label,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9.5)
    ax.set_ylabel("Test Macro F1 (%)", fontsize=10)
    ax.set_title("ShuffleNetV2: Standalone vs Knowledge Distillation",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(88.0, 92.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Teacher quality annotation
    ax.annotate(
        "Teacher F1: 0.9007\n(weaker → hurts student)",
        xy=(1, f1s[1] * 100), xytext=(1.28, 89.2),
        arrowprops=dict(arrowstyle="->", color="#e07b54", lw=1.2),
        fontsize=8, color="#e07b54",
    )
    ax.annotate(
        "Teacher F1: 0.9153\n(stronger → helps student)",
        xy=(2, f1s[2] * 100), xytext=(1.65, 91.4),
        arrowprops=dict(arrowstyle="->", color="#4c9be8", lw=1.2),
        fontsize=8, color="#4c9be8",
    )

    plt.tight_layout()
    path = OUT / "distillation_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_pareto()
    plot_onnx_latency()
    plot_distillation()
    print("\nAll 3 figures saved to results/plots/")
