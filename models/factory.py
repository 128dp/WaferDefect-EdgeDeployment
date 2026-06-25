"""
Model factory for WM-811K wafer defect classification.

All models use ImageNet pretrained weights by default, with the final
classification head swapped out for 8-class wafer defect detection.

Available architectures
-----------------------
resnet18       : 11.2 M params – strong baseline
resnet34       : 21.8 M params – stronger baseline
mobilenet_v2   :  3.4 M params – edge-friendly
efficientnet_b0:  5.3 M params – compact + accurate
shufflenet_v2  :  1.4 M params – ultra-lightweight (MCU-grade)
"""

from typing import Dict

import torch.nn as nn
import torchvision.models as tvm

# Approximate stats at 64×64 input (FLOPs via thop, params from torchvision)
MODEL_REGISTRY: Dict[str, dict] = {
    "resnet18":         {"family": "ResNet",       "params_m": 11.2, "note": "Baseline"},
    "resnet34":         {"family": "ResNet",       "params_m": 21.8, "note": "Stronger baseline"},
    "mobilenet_v2":     {"family": "MobileNet",    "params_m": 3.4,  "note": "Edge-friendly"},
    "efficientnet_b0":  {"family": "EfficientNet", "params_m": 5.3,  "note": "Compact + accurate"},
    "shufflenet_v2":    {"family": "ShuffleNet",   "params_m": 1.4,  "note": "Ultra-lightweight"},
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate a model with a custom head for wafer defect classification.

    Parameters
    ----------
    name        : architecture key from MODEL_REGISTRY
    num_classes : number of output classes (8 for WM-811K defects)
    pretrained  : load ImageNet pretrained weights before fine-tuning

    Returns
    -------
    nn.Module with the final classifier replaced
    """
    w = "DEFAULT" if pretrained else None

    if name == "resnet18":
        m = tvm.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet34":
        m = tvm.resnet34(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "shufflenet_v2":
        # x0_5 variant: ~1.4 M params — most suitable for microcontrollers
        m = tvm.shufflenet_v2_x0_5(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )

    return m
