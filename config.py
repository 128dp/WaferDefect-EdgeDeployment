"""
Central configuration for WM-811K Wafer Defect Detection.
All hyperparameters and path settings live here.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ── Class vocabulary (alphabetical one-hot order from the raw dataset) ─────────
ALL_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "none", "Random", "Scratch",
]
NONE_CLASS_IDX = 6  # index of "none" in ALL_CLASSES

# 8 defect classes after removing "none"
DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch",
]


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path: str = "data/LSWMD.pkl"
    image_size: int = 64          # resize all wafer maps to this resolution
    num_classes: int = 8          # 8 defect types (none excluded)
    exclude_none: bool = True     # exclude the dominant "none" class
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    seed: int = 42

    # ── Model ──────────────────────────────────────────────────────────────────
    # Options: resnet18 | resnet34 | mobilenet_v2 | efficientnet_b0 | shufflenet_v2
    model_name: str = "resnet18"
    pretrained: bool = True       # start from ImageNet weights

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"     # cosine | step | none
    num_workers: int = 0          # 0 is safest on Windows

    # ── Paths ──────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # ── Hardware ───────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Edge Benchmarking ──────────────────────────────────────────────────────
    benchmark_iterations: int = 200
    onnx_opset: int = 18   # PyTorch 2.11 dynamo exporter targets opset 18

    # ── Class Names ────────────────────────────────────────────────────────────
    class_names: List[str] = field(
        default_factory=lambda: list(DEFECT_CLASSES)
    )

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        for d in [
            self.results_dir,
            f"{self.results_dir}/training",
            f"{self.results_dir}/optuna",
            f"{self.results_dir}/pruning",
            f"{self.results_dir}/distillation",
            f"{self.results_dir}/benchmark",
            f"{self.results_dir}/plots",
            f"{self.results_dir}/multi_seed",
            f"{self.results_dir}/onnx",
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
