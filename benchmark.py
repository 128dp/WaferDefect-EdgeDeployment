"""
Edge device benchmarking script.

Measures for each model architecture:
  - Parameter count
  - GFLOPs  (via thop)
  - FP32 model size on disk
  - INT8 dynamic quantization size
  - ONNX export size
  - CPU inference latency  — FP32 (PyTorch) and ONNX Runtime (single-threaded,
    simulates Raspberry Pi / Jetson CPU workload)

Usage
-----
    python benchmark.py                    # benchmark all 5 architectures
    python benchmark.py --model resnet18
    python benchmark.py --model all --image_size 64 --iterations 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from models import build_model, MODEL_REGISTRY


# ── Measurement helpers ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    tmp = Path("_tmp_bench.pt")
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 ** 2)
    tmp.unlink()
    return round(size, 2)


def get_flops_g(model: nn.Module, image_size: int) -> float:
    try:
        from thop import profile
        dummy = torch.randn(1, 3, image_size, image_size)
        model.eval()
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return round(flops / 1e9, 4)
    except ImportError:
        print("  [WARN] thop not installed — skipping FLOPs count. Run: pip install thop")
        return -1.0


def measure_latency_pytorch(model: nn.Module, image_size: int, n_iters: int) -> float:
    """Average CPU latency in ms using PyTorch (FP32)."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(20):        # warm-up
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 3)


def measure_latency_onnx(onnx_path: str, image_size: int, n_iters: int) -> float:
    """Average CPU latency in ms using ONNX Runtime (single-threaded).

    Single-threaded mimics edge device constraints (Raspberry Pi, Jetson CPU).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnxruntime not installed. Run: pip install onnxruntime")
        return -1.0

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # single-core, like an edge MCU
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(20):                    # warm-up
        sess.run(None, {in_name: dummy})

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return round(float(np.mean(times)), 3)


def export_onnx(model: nn.Module, image_size: int, out_path: str, opset: int) -> float:
    """Export to ONNX and return total file size in MB (graph + external data)."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["wafer_map"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    # PyTorch 2.x dynamo exporter uses external data format — sum both files
    total = os.path.getsize(out_path)
    data_file = out_path + ".data"
    if os.path.exists(data_file):
        total += os.path.getsize(data_file)
    size = total / (1024 ** 2)
    print(f"  [ONNX] Exported → {out_path}  ({size:.2f} MB)")
    return round(size, 2)


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization on Linear layers (conservative but portable)."""
    return torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )


# ── Per-model benchmark ───────────────────────────────────────────────────────

def benchmark_model(name: str, cfg: Config) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Benchmarking: {name}")
    print(f"{'─'*55}")

    model = build_model(name, cfg.num_classes, pretrained=True)
    model.eval()

    params     = count_parameters(model)
    size_fp32  = get_model_size_mb(model)
    flops_g    = get_flops_g(model, cfg.image_size)
    lat_fp32   = measure_latency_pytorch(model, cfg.image_size, cfg.benchmark_iterations)

    # INT8 quantization
    model_int8 = quantize_dynamic_int8(model)
    size_int8  = get_model_size_mb(model_int8)
    lat_int8_pt = measure_latency_pytorch(model_int8, cfg.image_size, cfg.benchmark_iterations)

    # ONNX export + ONNX Runtime latency
    onnx_dir = Path(cfg.results_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(onnx_dir / f"{name}.onnx")
    try:
        onnx_size = export_onnx(model, cfg.image_size, onnx_path, cfg.onnx_opset)
        lat_onnx  = measure_latency_onnx(onnx_path, cfg.image_size, cfg.benchmark_iterations)
    except Exception as e:
        print(f"  [WARN] ONNX export/inference failed: {e}")
        onnx_size = -1.0
        lat_onnx  = -1.0

    result = {
        "model":              name,
        "params_m":           round(params / 1e6, 2),
        "flops_g":            flops_g,
        "size_fp32_mb":       size_fp32,
        "size_int8_mb":       size_int8,
        "onnx_size_mb":       onnx_size,
        "latency_fp32_ms":    lat_fp32,
        "latency_int8_ms":    lat_int8_pt,
        "latency_onnx_ms":    lat_onnx,
        "size_reduction_pct": round((1 - size_int8 / max(size_fp32, 1e-6)) * 100, 1),
        "onnx_speedup":       round(lat_fp32 / max(lat_onnx, 1e-6), 2) if lat_onnx > 0 else -1,
    }

    print(f"  Params            : {result['params_m']:.2f} M")
    print(f"  GFLOPs            : {flops_g}")
    print(f"  FP32 size         : {size_fp32:.2f} MB")
    print(f"  INT8 size         : {size_int8:.2f} MB  ({result['size_reduction_pct']}% reduction)")
    print(f"  FP32 latency (PT) : {lat_fp32:.2f} ms/img")
    print(f"  INT8 latency (PT) : {lat_int8_pt:.2f} ms/img")
    print(f"  ONNX latency      : {lat_onnx:.2f} ms/img  (speedup ×{result['onnx_speedup']})")

    return result


# ── Multi-model run ───────────────────────────────────────────────────────────

def run_benchmark(model_names: list, cfg: Config) -> dict:
    from tabulate import tabulate

    all_results = {}
    for name in model_names:
        all_results[name] = benchmark_model(name, cfg)

    # Summary table
    headers = [
        "Model", "Params (M)", "GFLOPs",
        "FP32 (MB)", "INT8 (MB)", "ONNX (MB)",
        "FP32 ms", "ONNX ms",
    ]
    rows = [[
        r["model"],
        r["params_m"],
        r["flops_g"],
        r["size_fp32_mb"],
        r["size_int8_mb"],
        r["onnx_size_mb"],
        r["latency_fp32_ms"],
        r["latency_onnx_ms"],
    ] for r in all_results.values()]

    print(f"\n{'='*90}")
    print("  EDGE BENCHMARK SUMMARY — WM-811K Models")
    print(f"{'='*90}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    out_path = Path(cfg.results_dir) / "benchmark" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Benchmark] Full results saved → {out_path}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Edge benchmark for WM-811K models")
    p.add_argument(
        "--model", default="all",
        help=f"Model name or 'all'. Options: {list(MODEL_REGISTRY.keys())}",
    )
    p.add_argument("--image_size",  type=int, default=None)
    p.add_argument("--iterations",  type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    if args.image_size:  cfg.image_size           = args.image_size
    if args.iterations:  cfg.benchmark_iterations = args.iterations

    names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    run_benchmark(names, cfg)
