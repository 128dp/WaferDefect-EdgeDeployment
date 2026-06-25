"""
Evaluation script — loads a trained checkpoint and reports full test-set metrics.

Usage
-----
    python evaluate.py                                    # evaluate ResNet-18
    python evaluate.py --model mobilenet_v2
    python evaluate.py --model resnet18 --ckpt checkpoints/resnet18_best.pt

Outputs
-------
    Console  : accuracy, macro F1, per-class classification report
    results/ : <model>_confusion_matrix.png
"""

import argparse
from pathlib import Path

import torch

from config import Config
from dataset import load_wm811k
from models import build_model
from utils import evaluate_model, plot_confusion_matrix, format_metrics_table


def evaluate(cfg: Config, ckpt_path: str = None):
    print(f"\n{'='*55}")
    print(f"  WM-811K – Evaluation: {cfg.model_name}")
    print(f"{'='*55}")

    # Data (test split only — val/train loaders are discarded)
    _, _, test_loader, _ = load_wm811k(cfg)

    # Model
    model = build_model(cfg.model_name, cfg.num_classes, pretrained=False)

    if ckpt_path is None:
        ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.model_name}_best.pt"

    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Eval] Checkpoint loaded: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"[Eval] WARNING — no checkpoint at '{ckpt_path}'. Evaluating with random weights.")

    model = model.to(cfg.device)

    # Compute metrics
    metrics = evaluate_model(model, test_loader, cfg.device, cfg.class_names)

    print(f"\n  Accuracy     : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
    print(f"\n{'-'*55}")
    print("[Per-Class Report]")
    print(metrics["report"])

    # Confusion matrix plot
    cm_path = f"{cfg.results_dir}/training/{cfg.model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg.class_names,
        title=f"Confusion Matrix – {cfg.model_name} (Test Set)",
        save_path=cm_path,
    )

    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate WM-811K defect detector")
    p.add_argument("--model",     default="resnet18")
    p.add_argument("--ckpt",      default=None, help="Path to .pt checkpoint")
    p.add_argument("--data_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config(model_name=args.model)
    if args.data_path:
        cfg.data_path = args.data_path
    evaluate(cfg, ckpt_path=args.ckpt)
