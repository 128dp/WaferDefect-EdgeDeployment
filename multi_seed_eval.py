"""
Multi-seed evaluation for statistical significance reporting.

Trains each specified model N times (default: 3 seeds) and reports
mean ± std for test-set accuracy and macro F1 — suitable for a
paper result table.

Usage
-----
    python multi_seed_eval.py                        # resnet18, 3 seeds
    python multi_seed_eval.py --models resnet18 efficientnet_b0
    python multi_seed_eval.py --all                  # all 5 models
    python multi_seed_eval.py --seeds 42 123 456 789 --epochs 40

Outputs
-------
    results/multi_seed_results.json   — per-seed + aggregate stats
    results/multi_seed_summary.png    — bar chart with error bars
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import Config
from dataset import load_wm811k
from models import build_model

ALL_MODELS = ["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2"]
DEFAULT_SEEDS = [42, 123, 456]


# ── Minimal train + test (no CSV / plot overhead) ────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        criterion(out, labels).backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    acc = correct / total
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return acc, f1


def train_and_eval_one_seed(model_name: str, seed: int, epochs: int) -> dict:
    """Train from scratch with a fixed seed and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = Config(model_name=model_name, epochs=epochs, seed=seed)
    train_loader, _, test_loader, class_weights = load_wm811k(cfg)

    model = build_model(model_name, cfg.num_classes, pretrained=True).to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.lr * 0.01
    )

    print(f"    seed={seed} — training {epochs} epochs...")
    for epoch in tqdm(range(1, epochs + 1), desc=f"    [{model_name} seed={seed}]", leave=False):
        _train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        scheduler.step()

    acc, f1 = _evaluate(model, test_loader, cfg.device)
    print(f"    seed={seed}  test acc={acc*100:.2f}%  macro F1={f1:.4f}")
    return {"seed": seed, "test_acc": round(acc, 4), "test_macro_f1": round(f1, 4)}


# ── Per-model multi-seed run ──────────────────────────────────────────────────

def run_model(model_name: str, seeds: list[int], epochs: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Multi-seed eval: {model_name}  ({len(seeds)} seeds, {epochs} epochs each)")
    print(f"{'='*60}")

    seed_results = [train_and_eval_one_seed(model_name, s, epochs) for s in seeds]

    accs = [r["test_acc"] for r in seed_results]
    f1s  = [r["test_macro_f1"] for r in seed_results]

    summary = {
        "model": model_name,
        "n_seeds": len(seeds),
        "epochs_per_seed": epochs,
        "seed_results": seed_results,
        "test_acc_mean":     round(float(np.mean(accs)), 4),
        "test_acc_std":      round(float(np.std(accs)),  4),
        "test_f1_mean":      round(float(np.mean(f1s)),  4),
        "test_f1_std":       round(float(np.std(f1s)),   4),
    }

    print(f"\n  {model_name} — Test Macro F1: "
          f"{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}  |  "
          f"Acc: {summary['test_acc_mean']*100:.2f}% ± {summary['test_acc_std']*100:.2f}%")

    return summary


# ── Summary plot ──────────────────────────────────────────────────────────────

def plot_summary(all_summaries: list[dict], save_path: str):
    models = [s["model"] for s in all_summaries]
    means  = [s["test_f1_mean"] * 100 for s in all_summaries]
    stds   = [s["test_f1_std"]  * 100 for s in all_summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color="steelblue", alpha=0.8,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.3,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Test Macro F1 (%)")
    ax.set_title("Multi-seed Test Macro F1 (mean ± std)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(100, max(means) + max(stds) + 6))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Summary plot → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Multi-seed evaluation for WM-811K models")
    p.add_argument("--models", nargs="+", metavar="MODEL",
                   help="Models to evaluate (default: resnet18)")
    p.add_argument("--all",    action="store_true",
                   help="Evaluate ALL 5 architectures")
    p.add_argument("--seeds",  nargs="+", type=int, default=DEFAULT_SEEDS,
                   help=f"Random seeds (default: {DEFAULT_SEEDS})")
    p.add_argument("--epochs", type=int, default=40,
                   help="Training epochs per seed (default: 40)")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.all:
        models_to_run = ALL_MODELS
    elif args.models:
        models_to_run = args.models
    else:
        models_to_run = ["resnet18"]

    results_dir = Path("results") / "multi_seed"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for model_name in models_to_run:
        summary = run_model(model_name, args.seeds, args.epochs)
        all_summaries.append(summary)

    # Save JSON
    out_path = results_dir / "multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[Saved] Results → {out_path}")

    # Print paper-ready table
    print(f"\n{'='*60}")
    print(f"  {'Model':<20}  {'Macro F1':>12}  {'Accuracy':>12}")
    print(f"  {'-'*54}")
    for s in all_summaries:
        f1_str  = f"{s['test_f1_mean']*100:.2f} ± {s['test_f1_std']*100:.2f}%"
        acc_str = f"{s['test_acc_mean']*100:.2f} ± {s['test_acc_std']*100:.2f}%"
        print(f"  {s['model']:<20}  {f1_str:>12}  {acc_str:>12}")
    print(f"{'='*60}")

    # Plot (only meaningful with 2+ models)
    if len(all_summaries) > 1:
        plot_summary(all_summaries, str(results_dir / "multi_seed_summary.png"))


if __name__ == "__main__":
    main()
