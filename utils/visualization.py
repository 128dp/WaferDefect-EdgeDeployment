"""
Visualization utilities for wafer defect classification results.
All functions save to disk and do not display interactively (headless-safe).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, save_path)


# ── Training History ──────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict,
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="s", markersize=3)
    axes[0].set_title(f"{model_name} – Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"],      label="Train Acc",   marker="o", markersize=3)
    axes[1].plot(epochs, history["val_macro_f1"],   label="Val Macro F1", marker="s", markersize=3)
    axes[1].set_title(f"{model_name} – Accuracy / F1", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History – {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


# ── Model Comparison Bar Charts ───────────────────────────────────────────────

def plot_model_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    names = list(results.keys())
    n = len(names)
    x = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 0.5, n))
    bar_kw = dict(width=0.6, edgecolor="black", linewidth=0.5)

    acc     = [results[m].get("accuracy", 0) * 100 for m in names]
    f1      = [results[m].get("macro_f1", 0) * 100 for m in names]
    size    = [results[m].get("size_mb", 0)         for m in names]
    latency = [results[m].get("latency_ms", 0)      for m in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        (acc,     "Accuracy (%)",       "Test Accuracy"),
        (f1,      "Macro F1 (%)",       "Macro F1 Score"),
        (size,    "Model Size (MB)",    "Model Size (FP32)"),
        (latency, "CPU Latency (ms)",   "Inference Latency (CPU)"),
    ]
    for ax, (vals, ylabel, title) in zip(axes.flat, specs):
        bars = ax.bar(x, vals, color=colors, **bar_kw)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle(
        "Model Comparison – WM-811K Wafer Defect Detection (Edge Benchmark)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path)


# ── Accuracy vs. Latency Pareto Curve ────────────────────────────────────────

def plot_pareto(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, r in results.items():
        lat = r.get("latency_ms", 0)
        acc = r.get("accuracy", 0) * 100
        size = r.get("params_m", 5) * 60   # bubble area ∝ parameter count
        ax.scatter(lat, acc, s=size, alpha=0.75, label=name,
                   edgecolors="black", linewidth=0.8)
        ax.annotate(name, (lat, acc), textcoords="offset points",
                    xytext=(7, 4), fontsize=9)

    ax.set_xlabel("CPU Inference Latency (ms / image)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Accuracy vs. Latency Pareto Frontier – WM-811K\n"
        "(bubble size ∝ parameter count, ideal: top-left)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, save_path)


# ── Class Distribution ────────────────────────────────────────────────────────

def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> None:
    counts = np.bincount(labels, minlength=len(class_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Defect Class")
    plt.xticks(rotation=30, ha="right")
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            str(cnt), ha="center", va="bottom", fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.close(fig)
