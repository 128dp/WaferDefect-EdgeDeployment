"""
Hyperparameter optimisation for WM-811K wafer defect classification
using Optuna (TPE sampler + Median pruner).

Usage
-----
    python optimize.py                                      # ResNet-18, 30 trials, 15 epochs each
    python optimize.py --model mobilenet_v2
    python optimize.py --models resnet18 efficientnet_b0    # subset of models
    python optimize.py --trials 50 --epochs 20
    python optimize.py --all                                # all 5 models sequentially
    python optimize.py --model resnet18 --trials 30 --epochs 15 --study_name my_study

Search space
------------
    lr            : log-uniform [1e-5, 1e-2]
    weight_decay  : log-uniform [1e-6, 1e-2]
    batch_size    : categorical [32, 64, 128]
    scheduler     : categorical [cosine, step, none]

Outputs
-------
    results/optuna_<model>_study.db    SQLite storage (resume-safe)
    results/optuna_<model>_results.json  Best params + history
    results/optuna_<model>_importance.png  Hyperparameter importance plot
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from config import Config
from dataset import load_wm811k
from models import build_model


# ── Single-epoch helpers (lightweight copies to avoid side-effects) ───────────

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
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
    for images, labels in loader:
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


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(model_name: str, n_epochs: int, data_fraction: float = 1.0):
    """
    Returns an Optuna objective function that:
      1. Samples hyperparameters from the search space
      2. Trains for `n_epochs` epochs on `data_fraction` of the training set
      3. Reports intermediate F1 each epoch (enables pruning)
      4. Returns the best validation macro-F1 for the trial

    data_fraction < 1.0 speeds up trials significantly on CPU — the relative
    ranking of hyperparameters is preserved even on a subset.
    """

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────────────────────
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step", "none"])

        # ── Build config from sampled params ─────────────────────────────────
        cfg = Config(
            model_name=model_name,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            scheduler=scheduler_name,
            epochs=n_epochs,
        )

        # ── Data ─────────────────────────────────────────────────────────────
        train_loader, val_loader, _, class_weights = load_wm811k(cfg)

        # Subsample training set to speed up CPU-bound trials
        if data_fraction < 1.0:
            import torch.utils.data as tud
            n_sub = max(1, int(len(train_loader.dataset) * data_fraction))
            indices = torch.randperm(len(train_loader.dataset), generator=
                                     torch.Generator().manual_seed(trial.number))[:n_sub].tolist()
            subset = tud.Subset(train_loader.dataset, indices)
            train_loader = tud.DataLoader(
                subset, batch_size=batch_size, shuffle=True,
                num_workers=cfg.num_workers
            )

        # ── Model ─────────────────────────────────────────────────────────────
        model = build_model(cfg.model_name, cfg.num_classes, cfg.pretrained)
        model = model.to(cfg.device)

        # ── Loss ──────────────────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))

        # ── Optimiser ─────────────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        if cfg.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
            )
        elif cfg.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        else:
            sched = None

        # ── Training loop ─────────────────────────────────────────────────────
        best_f1 = 0.0
        for epoch in range(1, cfg.epochs + 1):
            _train_epoch(model, train_loader, criterion, optimizer, cfg.device)
            _, _, val_f1 = _val_epoch(model, val_loader, criterion, cfg.device)

            if sched:
                sched.step()

            best_f1 = max(best_f1, val_f1)

            # Report intermediate value so the pruner can cut bad trials early
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_f1

    return objective


# ── All known models ─────────────────────────────────────────────────────────

ALL_MODELS = ["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Optuna hyperparameter optimisation for WM-811K")
    p.add_argument("--model",       default="resnet18",
                   help="Single architecture to optimise (ignored when --models or --all is set)")
    p.add_argument("--models",      nargs="+", metavar="MODEL",
                   help="One or more architectures to optimise sequentially, e.g. --models resnet18 efficientnet_b0")
    p.add_argument("--all",         action="store_true",
                   help="Run optimisation for ALL models sequentially (ignores --model / --models)")
    p.add_argument("--trials",      type=int, default=30,
                   help="Number of Optuna trials per model (default: 30)")
    p.add_argument("--epochs",      type=int, default=15,
                   help="Epochs per trial (default: 15 — short proxy for full training)")
    p.add_argument("--study_name",  default=None,
                   help="Optuna study name (defaults to optuna_<model>; ignored with --all)")
    p.add_argument("--jobs",        type=int, default=1,
                   help="Parallel jobs — keep at 1 on Windows to avoid CUDA conflicts")
    p.add_argument("--subset",      type=float, default=0.4,
                   help="Fraction of training data used per trial (default: 0.4 — 4-5× faster on CPU). "
                        "Set to 1.0 for full dataset.")
    return p.parse_args()


# ── Core per-model study runner ───────────────────────────────────────────────

def run_study(model_name: str, n_trials: int, n_epochs: int, n_jobs: int,
              data_fraction: float = 0.4,
              study_name: str | None = None) -> optuna.Study:
    """Create (or resume) an Optuna study for one model and return it."""
    study_name = study_name or f"optuna_{model_name}"
    results_dir = Path("results") / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)

    storage_path = results_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    print(f"\n{'='*62}")
    print(f"  Optuna Hyperparameter Search")
    print(f"  Model   : {model_name}")
    print(f"  Trials  : {n_trials}  |  Epochs/trial: {n_epochs}  |  Data subset: {data_fraction*100:.0f}%")
    print(f"  Storage : {storage_path}")
    print(f"{'='*62}\n")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # If the DB exists but has an incompatible search space, remove it and start fresh.
    if storage_path.exists():
        try:
            _test = optuna.load_study(study_name=study_name, storage=storage_url)
            _ = _test.trials  # force-load to surface any schema issues
        except Exception:
            print(f"  [Info] Removing incompatible study DB and starting fresh.")
            storage_path.unlink(missing_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    _wb_run = None
    if _WANDB:
        _wb_run = wandb.init(
            project="URECA",
            name=f"optuna_{model_name}",
            config={"model": model_name, "n_trials": n_trials,
                    "n_epochs": n_epochs, "data_fraction": data_fraction},
            reinit=True,
        )

        def _wandb_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            if trial.value is not None:
                wandb.log({"trial": trial.number, "val_macro_f1": trial.value,
                           **{f"param/{k}": v for k, v in trial.params.items()}})

    objective = make_objective(model_name, n_epochs, data_fraction)
    callbacks = [_wandb_callback] if _wb_run else []
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=callbacks)

    # ── Report ────────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*62}")
    print(f"  Best trial #{best.number}  Val macro-F1 = {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*62}\n")

    # ── Save per-model results ────────────────────────────────────────────────
    out = {
        "model": model_name,
        "best_trial": best.number,
        "best_val_macro_f1": best.value,
        "best_params": best.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    json_path = results_dir / f"{study_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] Results  → {json_path}")

    # ── Importance plot ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.figure.tight_layout()
        img_path = results_dir / f"{study_name}_importance.png"
        fig.figure.savefig(img_path, dpi=150)
        print(f"[Saved] Importance plot → {img_path}")
    except Exception as e:
        print(f"[Warning] Could not save importance plot: {e}")

    if _wb_run:
        wandb.summary["best_val_f1"] = best.value
        wandb.summary["best_trial"] = best.number
        for k, v in best.params.items():
            wandb.summary[f"best/{k}"] = v
        wandb.finish()

    return study


def main():
    args = _parse_args()
    if args.all:
        models_to_run = ALL_MODELS
    elif args.models:
        models_to_run = args.models
    else:
        models_to_run = [args.model]

    all_best: dict[str, dict] = {}
    for model_name in models_to_run:
        study_name = None if args.all else args.study_name
        study = run_study(
            model_name=model_name,
            n_trials=args.trials,
            n_epochs=args.epochs,
            n_jobs=args.jobs,
            data_fraction=args.subset,
            study_name=study_name,
        )
        all_best[model_name] = {
            "best_val_macro_f1": study.best_value,
            "best_params": study.best_params,
        }

    # ── Cross-model summary (only meaningful when --all is used) ──────────────
    if args.all:
        results_dir = Path("results") / "optuna"
        summary_path = results_dir / "optuna_all_models_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_best, f, indent=2)
        print(f"\n{'='*62}")
        print(f"  Cross-model summary")
        print(f"  {'Model':<20}  {'Best F1':>8}  Best LR")
        print(f"  {'-'*50}")
        for m, info in sorted(all_best.items(), key=lambda x: x[1]["best_val_macro_f1"], reverse=True):
            lr_str = f"{info['best_params'].get('lr', 'N/A'):.2e}"
            print(f"  {m:<20}  {info['best_val_macro_f1']:>8.4f}  {lr_str}")
        print(f"{'='*62}")
        print(f"[Saved] Summary → {summary_path}")


if __name__ == "__main__":
    main()
