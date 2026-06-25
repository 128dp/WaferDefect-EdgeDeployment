"""
Training script for WM-811K wafer defect classification.

Usage
-----
    python train.py                              # train ResNet-18 (default)
    python train.py --model mobilenet_v2
    python train.py --model efficientnet_b0 --epochs 40 --lr 5e-4
    python train.py --model shufflenet_v2 --quick   # 10 epochs for a quick demo

The best checkpoint (by validation macro F1) is saved to:
    checkpoints/<model_name>_best.pt

Training log CSV and history plot are saved to:
    results/<model_name>_training_log.csv
    results/<model_name>_training_history.png
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import Config
from dataset import load_wm811k
from models import build_model
from utils import plot_training_history

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── Single epoch helpers ──────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: Config, use_wandb: bool = True):
    print(f"\n{'='*62}")
    print(f"  WM-811K Wafer Defect Detection – Training Run")
    print(f"  Model   : {cfg.model_name}")
    print(f"  Device  : {cfg.device}")
    print(f"  Epochs  : {cfg.epochs}  |  LR: {cfg.lr}  |  Batch: {cfg.batch_size}")
    print(f"  Image   : {cfg.image_size}×{cfg.image_size}  |  Pretrained: {cfg.pretrained}")
    print(f"{'='*62}\n")

    _run = None
    if _WANDB and use_wandb:
        _run = wandb.init(
            project="URECA",
            name=cfg.model_name,
            config={
                "model": cfg.model_name,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "scheduler": cfg.scheduler,
                "image_size": cfg.image_size,
                "pretrained": cfg.pretrained,
            },
            reinit=True,
        )

    # Data
    train_loader, val_loader, _, class_weights = load_wm811k(cfg)

    # Model
    model = build_model(cfg.model_name, cfg.num_classes, cfg.pretrained)
    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params/1e6:.2f} M\n")

    # Loss: class-weighted cross-entropy combats imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # LR Scheduler
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # Tracking
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_macro_f1": [],
    }
    best_f1, best_epoch = 0.0, 0
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.model_name}_best.pt"

    log_path = Path(cfg.results_dir) / "training" / f"{cfg.model_name}_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"]
        )

    # Header row
    print(f"{'Ep':>4}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}  {'V-F1':>7}  {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        vl_loss, vl_acc, vl_f1 = _val_epoch(model, val_loader, criterion, cfg.device)
        lr_now = optimizer.param_groups[0]["lr"]

        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)

        print(
            f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
            f"{vl_loss:>8.4f}  {vl_acc*100:>6.2f}%  {vl_f1:>6.4f}  {lr_now:>10.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, vl_loss, vl_acc, vl_f1, lr_now]
            )

        if _run:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": vl_loss,
                "val/acc": vl_acc,
                "val/macro_f1": vl_f1,
                "lr": lr_now,
            })

        if vl_f1 > best_f1:
            best_f1, best_epoch = vl_f1, epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_f1": vl_f1},
                ckpt_path,
            )
            print(f"  ** Best F1 = {best_f1:.4f} at epoch {best_epoch} → {ckpt_path}")

    print(f"\n[Done] Best Val Macro F1 = {best_f1:.4f} (epoch {best_epoch})")
    print(f"[Done] Training log  → {log_path}")

    if _run:
        wandb.summary["best_val_f1"] = best_f1
        wandb.summary["best_epoch"] = best_epoch
        wandb.finish()

    # Plots & JSON history
    plot_training_history(
        history, cfg.model_name,
        save_path=f"{cfg.results_dir}/training/{cfg.model_name}_training_history.png",
    )
    hist_path = Path(cfg.results_dir) / "training" / f"{cfg.model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train WM-811K wafer defect detector")
    p.add_argument("--model",       default="resnet18", help="Architecture name")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--scheduler",   default=None, help="cosine | step | none")
    p.add_argument("--image_size",  type=int,   default=None)
    p.add_argument("--data_path",   default=None)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--quick",       action="store_true", help="10 epochs (fast demo)")
    p.add_argument("--no_wandb",    action="store_true", help="Disable W&B logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config(model_name=args.model)
    if args.epochs:         cfg.epochs     = args.epochs
    if args.lr:             cfg.lr           = args.lr
    if args.weight_decay:   cfg.weight_decay = args.weight_decay
    if args.batch_size:     cfg.batch_size   = args.batch_size
    if args.scheduler:      cfg.scheduler    = args.scheduler
    if args.image_size:     cfg.image_size = args.image_size
    if args.data_path:      cfg.data_path  = args.data_path
    if args.no_pretrained:  cfg.pretrained = False
    if args.quick:          cfg.epochs     = 10
    train(cfg, use_wandb=not args.no_wandb)
