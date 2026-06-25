# Lightweight Neural Network Compression for On-Device Wafer Map Defect Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![License](https://img.shields.io/badge/License-MIT-green)

URECA Undergraduate Research Programme — Nanyang Technological University, 2024-25

---

## Overview

This project investigates model compression techniques for eight-class semiconductor wafer map defect classification on the WM-811K benchmark. Five CNN architectures are trained under Optuna Bayesian hyperparameter optimisation, then compressed using three strategies:

| Technique | What was done |
|---|---|
| **Magnitude Pruning** | Joint Optuna search over sparsity + fine-tuning hyperparameters for ResNet-18 |
| **INT8 Quantisation** | Static post-training quantisation (FX-graph-mode) on dense and pruned ResNet-18 |
| **Knowledge Distillation** | EfficientNet-B0 and ResNet-18 as teachers for ShuffleNetV2 student |
| **Edge Benchmark** | ONNX Runtime single-threaded CPU latency for all five architectures |

**Key results:**
- 76.6% sparse ResNet-18 + static INT8 → **4× size reduction** (42.72 MB → 10.79 MB), F1: 0.9007 → **0.9035**
- ShuffleNetV2 + EfficientNet-B0 distillation → F1 = **0.9029** at **0.18 ms** ORT latency (52.96× speedup)
- Multi-seed evaluation (3 seeds) across all 5 architectures confirms result stability

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Dataset

Download **WM-811K** (`LSWMD.pkl`) and place it at `data/LSWMD.pkl`:
- Kaggle: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

---

## Reproducing Results

```bash
# 1. Hyperparameter search + full training (repeat for each model)
python optimize.py --model resnet18 --trials 30
python train.py --model resnet18

# 2. Pruning — joint Optuna search over sparsity + fine-tuning
python prune.py --model resnet18 --trials 40

# 3. Quantisation
python quantize.py --model resnet18 --method static
python quantize.py --model resnet18 --method static --ckpt checkpoints/resnet18_pruned_76.pt

# 4. Knowledge distillation
python distill.py --teacher efficientnet_b0 --student shufflenet_v2 --T 4.0 --alpha 0.7
python distill.py --teacher resnet18 --student shufflenet_v2 --T 4.0 --alpha 0.7

# 5. Edge benchmark (ONNX export + ORT latency)
python benchmark.py --model all --iterations 200

# 6. Multi-seed stability evaluation
python multi_seed_eval.py --models resnet18 resnet34 efficientnet_b0 mobilenet_v2 shufflenet_v2 --seeds 42 123 456 --epochs 40
```

---

## Results

### Baseline Comparison

| Model | Params (M) | Size (MB) | Test F1 | ORT latency (ms) |
|---|---|---|---|---|
| ResNet-34 | 21.29 | 81.34 | 0.9204 | 10.09 |
| EfficientNet-B0 | 4.02 | 15.61 | 0.9153 | 1.74 |
| MobileNetV2 | 2.23 | 8.75 | 0.9146 | 0.72 |
| ResNet-18 | 11.18 | 42.72 | 0.9007 | 5.36 |
| ShuffleNetV2 | 0.35 | 1.47 | 0.8926 | 0.18 |

### ResNet-18 Compression

| Variant | Test F1 | Size (MB) | Reduction |
|---|---|---|---|
| Dense FP32 | 0.9007 | 42.72 | 1× |
| Dense + Static INT8 | 0.8998 | 10.79 | 4× |
| Pruned 76.6% + Static INT8 | **0.9035** | **10.79** | **4×** |

### Knowledge Distillation (ShuffleNetV2 student)

| Teacher | Test F1 | vs Standalone |
|---|---|---|
| None (standalone) | 0.8926 | — |
| ResNet-18 | 0.8907 | −0.19% |
| EfficientNet-B0 | **0.9029** | **+1.03%** |

### Multi-seed Stability (3 seeds: 42, 123, 456)

| Model | Macro F1 | Std |
|---|---|---|
| EfficientNet-B0 | 92.39% | ±0.29% |
| ResNet-18 | 92.28% | ±0.31% |
| MobileNetV2 | 92.05% | ±0.33% |
| ResNet-34 | 91.91% | ±0.82% |
| ShuffleNetV2 | 88.35% | ±0.88% |

---

## Checkpoints

| File | Description | Size |
|---|---|---|
| `resnet18_best.pt` | Dense Optuna-tuned ResNet-18 | 42.7 MB |
| `resnet18_pruned_76.pt` | 76.6% sparse, Optuna fine-tuned | 42.7 MB |
| `resnet34_best.pt` | Dense Optuna-tuned ResNet-34 | 81.3 MB |
| `efficientnet_b0_best.pt` | Dense Optuna-tuned EfficientNet-B0 | 15.6 MB |
| `mobilenet_v2_best.pt` | Dense Optuna-tuned MobileNetV2 | 8.8 MB |
| `shufflenet_v2_best.pt` | Dense Optuna-tuned ShuffleNetV2 | 1.5 MB |
| `shufflenet_v2_distilled_best.pt` | EfficientNet-B0 distilled student | 1.5 MB |
| `shufflenet_v2_distilled_resnet18_best.pt` | ResNet-18 distilled student | 1.5 MB |

> Files >50 MB are tracked with Git LFS. Run `git lfs pull` after cloning.

---

## Project Structure

```
├── train.py               # Main training loop
├── optimize.py            # Optuna hyperparameter search
├── prune.py               # L1 magnitude pruning
├── quantize.py            # Static/dynamic INT8 quantisation
├── distill.py             # Knowledge distillation
├── benchmark.py           # ONNX export + latency benchmark
├── multi_seed_eval.py     # Multi-seed stability evaluation
├── evaluate.py            # Model evaluation utilities
├── config.py              # Central configuration
├── dataset.py             # WM-811K data loading
├── models/                # Architecture factory
├── utils/                 # Metrics and visualisation
├── plot_scripts/          # Figure generation scripts
├── results/               # JSON results + plots
└── checkpoints/           # Saved model weights
```

---

## Paper

See the accompanying URECA report (published separately).


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Dataset

Download **WM-811K** (`LSWMD.pkl`) from:
- Kaggle: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- Place at `data/LSWMD.pkl`

## Reproducing Results

```bash
# 1. Hyperparameter optimisation (all 5 architectures)
python optimize.py --model resnet18 --trials 30

# 2. Full training with best hyperparameters
python train.py --model resnet18

# 3. Pruning (joint Optuna search)
python prune.py --model resnet18 --trials 40

# 4. Quantisation
python quantize.py --model resnet18 --method static
python quantize.py --model resnet18 --method static --ckpt checkpoints/resnet18_pruned_76.pt

# 5. Knowledge distillation
python distill.py --teacher efficientnet_b0 --student shufflenet_v2 --T 4.0 --alpha 0.7

# 6. Edge benchmark (ONNX export + latency)
python benchmark.py --model all --iterations 200

# 7. Multi-seed stability
python multi_seed_eval.py --models resnet18 shufflenet_v2 --seeds 42 123 456 --epochs 40
```

## Checkpoints

Pre-trained checkpoints are stored in `checkpoints/`. Key files:

| File | Description | Size |
|---|---|---|
| `resnet18_best.pt` | Dense Optuna-tuned ResNet-18 | 42.7 MB |
| `resnet18_pruned_76.pt` | 76.6% sparse, Optuna fine-tuned | 42.7 MB |
| `resnet34_best.pt` | Dense Optuna-tuned ResNet-34 | 81.3 MB |
| `efficientnet_b0_best.pt` | Dense Optuna-tuned EfficientNet-B0 | 15.6 MB |
| `mobilenet_v2_best.pt` | Dense Optuna-tuned MobileNetV2 | 8.8 MB |
| `shufflenet_v2_best.pt` | Dense Optuna-tuned ShuffleNetV2 | 1.5 MB |
| `shufflenet_v2_distilled_best.pt` | EfficientNet-B0 distilled | 1.5 MB |
| `shufflenet_v2_distilled_resnet18_best.pt` | ResNet-18 distilled | 1.5 MB |

> **Note:** Files >50 MB are tracked with Git LFS. Run `git lfs pull` after cloning.

## Results Summary

| Model | Params (M) | Test F1 | Size (MB) | ORT latency (ms) |
|---|---|---|---|---|
| ResNet-34 | 21.29 | 0.9204 | 81.34 | 10.09 |
| EfficientNet-B0 | 4.02 | 0.9153 | 15.61 | 1.74 |
| MobileNetV2 | 2.23 | 0.9146 | 8.75 | 0.72 |
| ResNet-18 | 11.18 | 0.9007 | 42.72 | 5.36 |
| ShuffleNetV2 | 0.35 | 0.8926 | 1.47 | 0.18 |
| ResNet-18 pruned 76.6% + INT8 | 11.18 | **0.9035** | **10.79** | — |
| ShuffleNetV2 + EffNet distillation | 0.35 | **0.9029** | 1.47 | **0.18** |

## Paper

See the accompanying URECA report (published separately).

## Project Structure

```
├── train.py               # Main training loop
├── optimize.py            # Optuna hyperparameter search
├── prune.py               # L1 magnitude pruning
├── quantize.py            # Static/dynamic INT8 quantisation
├── distill.py             # Knowledge distillation
├── benchmark.py           # ONNX export + latency benchmark
├── multi_seed_eval.py     # Multi-seed stability evaluation
├── evaluate.py            # Model evaluation utilities
├── config.py              # Central configuration
├── dataset.py             # WM-811K data loading
├── models/                # Architecture factory
├── utils/                 # Metrics and visualisation
├── plot_scripts/          # Figure generation scripts
├── results/               # JSON results + figures
└── checkpoints/           # Saved model weights
```
