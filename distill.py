"""
Knowledge Distillation for WM-811K Wafer Defect Detection on Edge Devices.

Teacher : ResNet-18  (11.2 M params — high accuracy)
Student : ShuffleNetV2-x0.5  (0.35 M params — 32× smaller, MCU-grade)

The student learns from two signals simultaneously:
  1. Hard labels   — standard cross-entropy against ground truth
  2. Soft targets  — KL divergence against the teacher's temperature-scaled
                     probability distribution (captures inter-class relationships)

Loss = α · KL(softStudent ∥ softTeacher) · T²  +  (1−α) · CE(student, labels)

  T (temperature) : higher T → softer distributions, more knowledge transferred
  α (alpha)       : weight on the soft-target loss term

Usage
-----
    python distill.py                              # default settings
    python distill.py --quick                      # 10 epochs (demo)
    python distill.py --teacher resnet34 --student mobilenet_v2
    python distill.py --T 6 --alpha 0.8 --epochs 40

Outputs
-------
    checkpoints/<student>_distilled_best.pt
    results/<student>_distilled_training_history.png
    results/<student>_distilled_confusion_matrix.png
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import Config
from dataset import load_wm811k
from evaluate import evaluate
from models import build_model
from utils import plot_training_history


# ── Distillation loss ─────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    T: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Combined soft-target KL loss + hard-label cross-entropy.

    Parameters
    ----------
    T     : temperature — higher values produce softer probability distributions
    alpha : weight on the KL (soft) term; (1-alpha) weights the CE (hard) term
    """
    # Soft targets: scale logits by T before softmax
    soft_log_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher     = F.softmax(teacher_logits  / T, dim=1)
    kl_loss = F.kl_div(soft_log_student, soft_teacher, reduction="batchmean") * (T ** 2)

    # Hard targets: standard weighted cross-entropy
    ce_loss = F.cross_entropy(student_logits, labels, weight=class_weights)

    return alpha * kl_loss + (1.0 - alpha) * ce_loss


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(student, teacher, loader, optimizer, class_weights, device, T, alpha):
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pass  # teacher stays in eval — no gradient needed

    for images, labels in tqdm(loader, desc="  Distill", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        student_out = student(images)
        with torch.no_grad():
            teacher_out = teacher(images)

        loss = distillation_loss(student_out, teacher_out, labels, class_weights, T, alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (student_out.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for images, labels in tqdm(loader, desc="  Val    ", leave=False):
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        preds = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1


# ── Main distillation function ────────────────────────────────────────────────

def distill(cfg: Config, teacher_name: str, student_name: str, T: float, alpha: float):
    run_name = f"{student_name}_distilled_{teacher_name}"

    print(f"\n{'='*65}")
    print(f"  Knowledge Distillation")
    print(f"  Teacher : {teacher_name}")
    print(f"  Student : {student_name}")
    print(f"  T (temp): {T}   |   α (alpha): {alpha}")
    print(f"  Epochs  : {cfg.epochs}  |  Device: {cfg.device}")
    print(f"{'='*65}\n")

    # Data
    train_loader, val_loader, _, class_weights = load_wm811k(cfg)
    cw = class_weights.to(cfg.device)

    # Teacher — load best checkpoint
    teacher = build_model(teacher_name, cfg.num_classes, pretrained=False).to(cfg.device)
    teacher_ckpt = Path(cfg.checkpoint_dir) / f"{teacher_name}_best.pt"
    if teacher_ckpt.exists():
        ckpt = torch.load(teacher_ckpt, map_location=cfg.device)
        teacher.load_state_dict(ckpt["model_state"])
        print(f"[KD] Teacher loaded from {teacher_ckpt}  (epoch {ckpt.get('epoch','?')})")
    else:
        print(f"[KD] WARNING — no teacher checkpoint at '{teacher_ckpt}'.")
        print(f"[KD] Run: python train.py --model {teacher_name} first.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False     # teacher is frozen

    # Student
    student = build_model(student_name, cfg.num_classes, cfg.pretrained).to(cfg.device)
    n_s = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"[KD] Teacher params : {n_t/1e6:.2f} M")
    print(f"[KD] Student params : {n_s/1e6:.2f} M  ({n_t/n_s:.1f}× compression)\n")

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )
    ce_criterion = nn.CrossEntropyLoss(weight=cw)   # for validation only

    # Logging
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_macro_f1": [],
    }
    best_f1, best_epoch = 0.0, 0
    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}_best.pt"
    log_path  = Path(cfg.results_dir) / "distillation" / f"{run_name}_training_log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"]
        )

    print(f"{'Ep':>4}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}  {'V-F1':>7}")
    print("-" * 55)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            student, teacher, train_loader, optimizer, cw, cfg.device, T, alpha
        )
        vl_loss, vl_acc, vl_f1 = _val_epoch(student, val_loader, ce_criterion, cfg.device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)

        print(
            f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
            f"{vl_loss:>8.4f}  {vl_acc*100:>6.2f}%  {vl_f1:>6.4f}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc, vl_loss, vl_acc, vl_f1])

        if vl_f1 > best_f1:
            best_f1, best_epoch = vl_f1, epoch
            torch.save(
                {"model_state": student.state_dict(), "epoch": epoch, "val_f1": vl_f1},
                ckpt_path,
            )
            print(f"  ** Best F1 = {best_f1:.4f} at epoch {best_epoch} → {ckpt_path}")

    print(f"\n[KD] Done. Best Val Macro F1 = {best_f1:.4f} (epoch {best_epoch})")

    # Save plots
    plot_training_history(
        history, run_name,
        save_path=f"{cfg.results_dir}/distillation/{run_name}_training_history.png",
    )
    with open(Path(cfg.results_dir) / "distillation" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    print(f"\n[KD] Evaluating distilled student on test set...")
    cfg.model_name = student_name
    # point evaluate() to the distilled checkpoint
    from evaluate import evaluate as _eval
    from utils import plot_confusion_matrix, evaluate_model
    from dataset import load_wm811k as _load

    _, _, test_loader, _ = _load(cfg)
    student_eval = build_model(student_name, cfg.num_classes, pretrained=False).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    student_eval.load_state_dict(ckpt["model_state"])
    metrics = evaluate_model(student_eval, test_loader, cfg.device, cfg.class_names)

    print(f"\n  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(metrics["report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"], cfg.class_names,
        title=f"Confusion Matrix – {run_name} (Test Set)",
        save_path=f"{cfg.results_dir}/distillation/{run_name}_confusion_matrix.png",
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Knowledge Distillation for WM-811K")
    p.add_argument("--teacher",      default="resnet18")
    p.add_argument("--student",      default="shufflenet_v2")
    p.add_argument("--T",            type=float, default=4.0,  help="Distillation temperature")
    p.add_argument("--alpha",        type=float, default=0.7,  help="Weight on soft-target loss")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None, help="Student learning rate")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--quick",        action="store_true",      help="10 epochs (demo mode)")
    p.add_argument("--data_path",    default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    if args.epochs:       cfg.epochs       = args.epochs
    if args.data_path:    cfg.data_path    = args.data_path
    if args.quick:        cfg.epochs       = 10
    if args.lr:           cfg.lr           = args.lr
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.batch_size:   cfg.batch_size   = args.batch_size

    distill(cfg, teacher_name=args.teacher, student_name=args.student,
            T=args.T, alpha=args.alpha)
