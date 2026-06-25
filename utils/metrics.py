"""
Evaluation metrics for WM-811K wafer defect classification.

Uses macro F1 as the primary metric — more meaningful than accuracy
given the severe class imbalance (Donut ~0.3%, Near-full ~0.1%).
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tabulate import tabulate
from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str],
) -> Dict:
    """Run inference on a DataLoader and compute full classification metrics.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, per_class_f1,
        report (str), confusion_matrix, y_true, y_pred
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None,       zero_division=0),
        "report":       classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def format_metrics_table(results: Dict[str, Dict], class_names: List[str]) -> str:
    """Format a multi-model comparison as a printable grid table."""
    headers = [
        "Model", "Acc (%)", "Macro F1", "Wtd F1",
        "Params (M)", "Size (MB)", "Latency (ms)",
    ]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r.get('accuracy', 0) * 100:.2f}",
            f"{r.get('macro_f1', 0):.4f}",
            f"{r.get('weighted_f1', 0):.4f}",
            r.get("params_m", "–"),
            r.get("size_mb", "–"),
            r.get("latency_ms", "–"),
        ])
    return tabulate(rows, headers=headers, tablefmt="grid")
