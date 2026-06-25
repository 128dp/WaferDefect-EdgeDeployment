"""
Full model comparison pipeline for WM-811K wafer defect detection on edge devices.

Runs three phases:
  1. Train each architecture (skip with --benchmark_only)
  2. Evaluate on the held-out test set
  3. Benchmark edge performance (FLOPs, size, quantization, ONNX latency)

Then produces:
  results/model_comparison.png   — 4-panel bar chart
  results/pareto_curve.png       — Accuracy vs. Latency Pareto frontier
  results/comparison_summary.json

Usage
-----
    python compare_models.py                         # full pipeline, 40 epochs each
    python compare_models.py --quick                 # 10 epochs (demo mode)
    python compare_models.py --benchmark_only        # skip training, just benchmark
    python compare_models.py --models resnet18 mobilenet_v2
"""

import argparse
import json
from pathlib import Path

from benchmark import run_benchmark
from config import Config
from evaluate import evaluate
from models import MODEL_REGISTRY
from train import train
from utils import plot_model_comparison, plot_pareto

COMPARISON_MODELS = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2"]


def run_comparison(models: list, cfg: Config, benchmark_only: bool = False) -> dict:
    print(f"\n{'#'*65}")
    print(f"  WM-811K — Model Comparison for Edge Deployment")
    print(f"  Architectures : {models}")
    print(f"  Device  : {cfg.device}  |  Epochs: {cfg.epochs}  |  Image: {cfg.image_size}×{cfg.image_size}")
    print(f"{'#'*65}\n")

    comparison = {}

    # ── Phase 1 & 2: Train + Evaluate ─────────────────────────────────────────
    if not benchmark_only:
        for name in models:
            print(f"\n{'▶'*3}  Training: {name}  {'◀'*3}")
            cfg.model_name = name
            train(cfg)

            print(f"\n{'▶'*3}  Evaluating: {name}  {'◀'*3}")
            metrics = evaluate(cfg)
            comparison[name] = {
                "accuracy":    metrics["accuracy"],
                "macro_f1":    metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }

    # ── Load eval results if benchmark_only ───────────────────────────────────
    if benchmark_only:
        for name in models:
            hist_path = Path(cfg.results_dir) / "training" / f"{name}_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                # history.json stores per-epoch lists; pick the best val_f1 epoch
                val_f1s = hist.get("val_macro_f1", [])
                val_accs = hist.get("val_acc", [])
                if val_f1s:
                    best_idx = int(max(range(len(val_f1s)), key=lambda i: val_f1s[i]))
                    comparison.setdefault(name, {}).update({
                        "accuracy":  val_accs[best_idx] if val_accs else 0,
                        "macro_f1":  val_f1s[best_idx],
                    })

    # ── Phase 3: Edge Benchmark ────────────────────────────────────────────────
    print(f"\n{'▶'*3}  Edge Benchmarking  {'◀'*3}")
    bench = run_benchmark(models, cfg)

    for name in models:
        entry = comparison.setdefault(name, {})
        entry.update({
            "params_m":       bench[name]["params_m"],
            "size_mb":        bench[name]["size_fp32_mb"],
            "size_int8_mb":   bench[name]["size_int8_mb"],
            "onnx_size_mb":   bench[name]["onnx_size_mb"],
            "flops_g":        bench[name]["flops_g"],
            "latency_ms":     bench[name]["latency_fp32_ms"],
            "latency_onnx_ms":bench[name]["latency_onnx_ms"],
        })

    # ── Phase 4: Plots ─────────────────────────────────────────────────────────
    plot_model_comparison(
        comparison,
        save_path=f"{cfg.results_dir}/plots/model_comparison.png",
    )
    plot_pareto(
        comparison,
        save_path=f"{cfg.results_dir}/plots/pareto_curve.png",
    )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = Path(cfg.results_dir) / "benchmark" / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Compare] Summary → {summary_path}")

    # ── Final table ───────────────────────────────────────────────────────────
    from tabulate import tabulate
    headers = [
        "Model", "Acc (%)", "Macro F1",
        "Params (M)", "FP32 (MB)", "INT8 (MB)", "ONNX (MB)",
        "FP32 ms", "ONNX ms", "GFLOPs",
    ]
    rows = []
    for name, r in comparison.items():
        acc = f"{r['accuracy']*100:.2f}" if r.get("accuracy", 0) > 0 else "–"
        f1  = f"{r['macro_f1']:.4f}"     if r.get("macro_f1",  0) > 0 else "–"
        rows.append([
            name, acc, f1,
            r.get("params_m",        "–"),
            r.get("size_mb",         "–"),
            r.get("size_int8_mb",    "–"),
            r.get("onnx_size_mb",    "–"),
            r.get("latency_ms",      "–"),
            r.get("latency_onnx_ms", "–"),
            r.get("flops_g",         "–"),
        ])

    print(f"\n{'='*110}")
    print("  FINAL RESULTS — WM-811K Wafer Defect Detection on Edge Devices")
    print(f"{'='*110}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nPlots saved to: {cfg.results_dir}/")

    return comparison


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="WM-811K full model comparison")
    p.add_argument(
        "--models", nargs="+", default=COMPARISON_MODELS,
        help="Architectures to compare (default: all 4)",
    )
    p.add_argument("--quick",          action="store_true", help="10 epochs each")
    p.add_argument("--benchmark_only", action="store_true", help="Skip training")
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--image_size",     type=int,   default=64)
    p.add_argument("--data_path",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    cfg.image_size = args.image_size
    if args.data_path:  cfg.data_path = args.data_path
    if args.quick:      cfg.epochs    = 10
    if args.epochs:     cfg.epochs    = args.epochs

    run_comparison(
        models=args.models,
        cfg=cfg,
        benchmark_only=args.benchmark_only,
    )
