"""
Quantization for WM-811K wafer defect models.

Covers two quantization strategies relevant for edge deployment:

1. Post-Training Dynamic Quantization (PTDQ)
   - Quantizes Linear layers to INT8 at runtime
   - No calibration data needed
   - Fast to apply, good for RNN/FC-heavy models

2. Post-Training Static Quantization (PTSQ)
   - Quantizes Conv2d + Linear layers to INT8 statically
   - Requires a calibration pass with representative data
   - Better size and latency reduction than dynamic
   - Most practical for CNN-heavy architectures like ours

Metrics reported
----------------
  - FP32 baseline accuracy / F1
  - INT8 accuracy / F1  (quantization accuracy drop)
  - Model size: FP32 vs INT8 (MB)
  - CPU inference latency: FP32 vs INT8 (ms/image)

Usage
-----
    python quantize.py --model resnet18
    python quantize.py --model mobilenet_v2 --method static
    python quantize.py --model all --method both
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from dataset import load_wm811k
from models import build_model, MODEL_REGISTRY
from utils import evaluate_model, plot_confusion_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_size_mb(model: nn.Module) -> float:
    tmp = Path("_tmp_quant.pt")
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 ** 2)
    tmp.unlink()
    return round(size, 2)


def measure_latency_ms(model: nn.Module, image_size: int, n: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 3)


# ── Quantization methods ──────────────────────────────────────────────────────

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization via torchao — quantizes Conv2d + Linear weights and activations."""
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    model = model.cpu().eval()
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return model


def apply_static_quantization(model: nn.Module, calib_loader, cfg: Config) -> nn.Module:
    """INT8 static quantization via FX graph mode — auto-fuses Conv+BN+ReLU and calibrates."""
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import QConfigMapping, get_default_qconfig

    model = model.cpu().eval()
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig("x86"))
    example_inputs = (torch.randn(1, 3, cfg.image_size, cfg.image_size),)

    model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    print("  [Static] Running calibration pass...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(calib_loader, desc="  Calibrating", leave=False)):
            model_prepared(images.cpu())
            if i >= 20:
                break

    model_static = convert_fx(model_prepared)
    return model_static


# ── Per-model quantization run ────────────────────────────────────────────────

def quantize_model(model_name: str, method: str, cfg: Config, ckpt_override: str = None) -> dict:
    print(f"\n{'-'*60}")
    print(f"  Quantizing: {model_name}  |  method: {method}")
    print(f"{'-'*60}")

    # Load data (need test set + calibration subset)
    train_loader, _, test_loader, _ = load_wm811k(cfg)

    # Load trained model
    model = build_model(model_name, cfg.num_classes, pretrained=False)
    ckpt_path = Path(ckpt_override) if ckpt_override else Path(cfg.checkpoint_dir) / f"{model_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  WARNING — no checkpoint. Run: python train.py --model {model_name}")

    model = model.cpu()
    model.eval()

    # FP32 baseline
    print("  Evaluating FP32 baseline...")
    fp32_metrics = evaluate_model(model, test_loader, "cpu", cfg.class_names)
    fp32_size    = get_size_mb(model)
    fp32_latency = measure_latency_ms(model, cfg.image_size)

    print(f"  FP32 — Acc: {fp32_metrics['accuracy']*100:.2f}%  |  "
          f"F1: {fp32_metrics['macro_f1']:.4f}  |  "
          f"Size: {fp32_size:.2f} MB  |  Latency: {fp32_latency:.2f} ms")

    results = {
        "model": model_name,
        "fp32_accuracy":    round(fp32_metrics["accuracy"], 4),
        "fp32_macro_f1":    round(fp32_metrics["macro_f1"], 4),
        "fp32_size_mb":     fp32_size,
        "fp32_latency_ms":  fp32_latency,
    }

    # Dynamic INT8
    if method in ("dynamic", "both"):
        print("\n  Applying dynamic INT8 quantization...")
        model_dyn = apply_dynamic_quantization(build_model(model_name, cfg.num_classes, pretrained=False))
        ckpt = torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
        if ckpt:
            _m = build_model(model_name, cfg.num_classes, pretrained=False)
            _m.load_state_dict(ckpt["model_state"])
            model_dyn = apply_dynamic_quantization(_m)

        dyn_metrics  = evaluate_model(model_dyn, test_loader, "cpu", cfg.class_names)
        dyn_size     = get_size_mb(model_dyn)
        dyn_latency  = measure_latency_ms(model_dyn, cfg.image_size)
        acc_drop     = (fp32_metrics["accuracy"] - dyn_metrics["accuracy"]) * 100

        print(f"  INT8 Dynamic — Acc: {dyn_metrics['accuracy']*100:.2f}%  |  "
              f"F1: {dyn_metrics['macro_f1']:.4f}  |  "
              f"Size: {dyn_size:.2f} MB  |  Latency: {dyn_latency:.2f} ms  |  "
              f"Acc drop: {acc_drop:+.2f}%")

        results.update({
            "dyn_accuracy":    round(dyn_metrics["accuracy"], 4),
            "dyn_macro_f1":    round(dyn_metrics["macro_f1"], 4),
            "dyn_size_mb":     dyn_size,
            "dyn_latency_ms":  dyn_latency,
            "dyn_acc_drop":    round(acc_drop, 3),
            "dyn_size_reduction_pct": round((1 - dyn_size / fp32_size) * 100, 1),
        })

        ckpt_dyn = Path(cfg.checkpoint_dir) / f"{model_name}_int8_dynamic.pt"
        torch.save(model_dyn.state_dict(), ckpt_dyn)

    # Static INT8
    if method in ("static", "both"):
        print("\n  Applying static INT8 quantization (with calibration)...")
        _m = build_model(model_name, cfg.num_classes, pretrained=False)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            _m.load_state_dict(ckpt["model_state"])
        try:
            model_static = apply_static_quantization(_m, train_loader, cfg)
            st_metrics  = evaluate_model(model_static, test_loader, "cpu", cfg.class_names)
            st_size     = get_size_mb(model_static)
            st_latency  = measure_latency_ms(model_static, cfg.image_size)
            acc_drop_st = (fp32_metrics["accuracy"] - st_metrics["accuracy"]) * 100

            print(f"  INT8 Static  — Acc: {st_metrics['accuracy']*100:.2f}%  |  "
                  f"F1: {st_metrics['macro_f1']:.4f}  |  "
                  f"Size: {st_size:.2f} MB  |  Latency: {st_latency:.2f} ms  |  "
                  f"Acc drop: {acc_drop_st:+.2f}%")

            results.update({
                "static_accuracy":    round(st_metrics["accuracy"], 4),
                "static_macro_f1":    round(st_metrics["macro_f1"], 4),
                "static_size_mb":     st_size,
                "static_latency_ms":  st_latency,
                "static_acc_drop":    round(acc_drop_st, 3),
                "static_size_reduction_pct": round((1 - st_size / fp32_size) * 100, 1),
            })
        except Exception as e:
            print(f"  [WARN] Static quantization failed for {model_name}: {e}")
            print("  This is expected for some architectures — use dynamic instead.")

    # Save summary
    out_path = Path(cfg.results_dir) / "benchmark" / f"{model_name}_quantization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Quantize] Results saved → {out_path}")

    return results


# ── Multi-model run ───────────────────────────────────────────────────────────

def run_quantization(model_names: list, method: str, cfg: Config, ckpt_override: str = None):
    from tabulate import tabulate
    all_results = {}
    for name in model_names:
        all_results[name] = quantize_model(name, method, cfg, ckpt_override)

    # Summary table
    print(f"\n{'='*80}")
    print("  QUANTIZATION SUMMARY")
    print(f"{'='*80}")
    headers = ["Model", "FP32 Acc", "FP32 MB", "INT8 Acc", "INT8 MB", "Size ↓%", "Acc drop"]
    rows = []
    for name, r in all_results.items():
        int8_acc  = r.get("dyn_accuracy",    r.get("static_accuracy", "–"))
        int8_mb   = r.get("dyn_size_mb",     r.get("static_size_mb",  "–"))
        size_drop = r.get("dyn_size_reduction_pct", r.get("static_size_reduction_pct", "–"))
        acc_drop  = r.get("dyn_acc_drop",    r.get("static_acc_drop", "–"))
        rows.append([
            name,
            f"{r['fp32_accuracy']*100:.2f}%",
            f"{r['fp32_size_mb']:.2f}",
            f"{int8_acc*100:.2f}%" if isinstance(int8_acc, float) else "–",
            f"{int8_mb:.2f}"       if isinstance(int8_mb,  float) else "–",
            f"{size_drop}%"        if isinstance(size_drop, (int, float)) else "–",
            f"{acc_drop:+.2f}%"   if isinstance(acc_drop,  float) else "–",
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Quantization for WM-811K wafer defect models")
    p.add_argument("--model",  default="resnet18",
                   help=f"Model name or 'all'. Options: {list(MODEL_REGISTRY.keys())}")
    p.add_argument("--method", default="dynamic", choices=["dynamic", "static", "both"])
    p.add_argument("--ckpt",   default=None,
                   help="Override checkpoint path (e.g. checkpoints/resnet18_pruned_76.pt)")
    p.add_argument("--data_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    if args.data_path: cfg.data_path = args.data_path

    names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

