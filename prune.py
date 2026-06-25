"""
Magnitude-based pruning for WM-811K wafer defect models.

Applies global L1-norm unstructured pruning to all Conv2d and Linear weights,
then fine-tunes the sparse model to recover accuracy. Iterates across multiple
sparsity levels to show the accuracy–compression tradeoff curve.

Pruning pipeline
----------------
  1. Load trained dense model
  2. Apply global L1-unstructured pruning at target sparsity s
  3. Fine-tune for fine_tune_epochs epochs
  4. Remove pruning masks (make sparsity permanent)
  5. Measure accuracy, F1, model size, and effective parameter count
  6. Repeat for s ∈ {0.3, 0.5, 0.7}

Usage
-----
    python prune.py --model resnet18 --sparsity 0.5
    python prune.py --model resnet18 --sweep          # 0.3, 0.5, 0.7
    python prune.py --model resnet18 --sweep --quick  # 5 fine-tune epochs each
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import Config
from dataset import load_wm811k
from models import build_model
from utils import evaluate_model, plot_confusion_matrix


# ── Pruning helpers ───────────────────────────────────────────────────────────

def apply_global_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """Apply global L1-unstructured pruning across all Conv2d + Linear layers."""
    params_to_prune = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    torch_prune.global_unstructured(
        params_to_prune,
        pruning_method=torch_prune.L1Unstructured,
        amount=sparsity,
    )
    return model


def remove_pruning_masks(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing masks and zeroing pruned weights."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                torch_prune.remove(m, "weight")
            except ValueError:
                pass
    return model


def count_nonzero_params(model: nn.Module):
    """Count total and non-zero parameters (after pruning masks applied)."""
    total, nonzero = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight_mask * m.weight_orig if hasattr(m, "weight_mask") else m.weight
            total   += w.numel()
            nonzero += w.nonzero().size(0)
    return total, nonzero


def get_actual_sparsity(model: nn.Module) -> float:
    total, nonzero = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight_mask * m.weight_orig if hasattr(m, "weight_mask") else m.weight
            total   += w.numel()
            nonzero += (w != 0).sum().item()
    return 1.0 - (nonzero / max(total, 1))


# ── Fine-tuning loop ──────────────────────────────────────────────────────────

def fine_tune(model, train_loader, val_loader, cfg: Config, n_epochs: int, fine_tune_lr: float = None):
    """Short fine-tuning pass after pruning to recover accuracy."""
    _, _, _, class_weights = load_wm811k.__wrapped__(cfg) if hasattr(load_wm811k, "__wrapped__") else (None, None, None, None)
    # Re-load class weights cheaply
    import pickle, sys, warnings, pandas as pd, pandas.core.indexes
    sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(cfg.data_path, "rb") as f:
            import pickle as _pkl
            df = _pkl.load(f, encoding="latin1")
    df_l = df[df["failureType"].apply(lambda x: x.size > 0 and str(x[0][0]).strip() != "")].copy()
    df_l["label_str"] = df_l["failureType"].apply(lambda x: str(x[0][0]))
    df_l = df_l[df_l["label_str"] != "none"]
    lmap = {n: i for i, n in enumerate(cfg.class_names)}
    df_l["label"] = df_l["label_str"].apply(lambda s: lmap[s])
    from sklearn.model_selection import train_test_split
    idx = list(range(len(df_l)))
    labels = df_l["label"].tolist()
    tr_idx, _ = train_test_split(idx, test_size=cfg.test_ratio, stratify=labels, random_state=cfg.seed)
    tr_labels = [labels[i] for i in tr_idx]
    counts = np.bincount(tr_labels, minlength=cfg.num_classes).astype(float)
    w = 1.0 / np.maximum(counts, 1.0)
    cw = torch.FloatTensor((w / w.sum()) * cfg.num_classes).to(cfg.device)

    criterion = nn.CrossEntropyLoss(weight=cw)
    lr = fine_tune_lr if fine_tune_lr is not None else cfg.lr * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        model.train()
        for images, labels_b in tqdm(train_loader, desc=f"  Fine-tune {epoch}/{n_epochs}", leave=False):
            images, labels_b = images.to(cfg.device), labels_b.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_b)
            loss.backward()
            optimizer.step()

        # Quick val check
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for images, labels_b in val_loader:
                images = images.to(cfg.device)
                preds_all.extend(model(images).argmax(1).cpu().numpy())
                labels_all.extend(labels_b.numpy())
        f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        print(f"    epoch {epoch}/{n_epochs}  val macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_f1


# ── Single sparsity run ───────────────────────────────────────────────────────

def prune_and_evaluate(
    model_name: str, sparsity: float, cfg: Config,
    train_loader, val_loader, test_loader,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = None,
) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Pruning {model_name}  |  target sparsity = {sparsity*100:.0f}%")
    print(f"{'─'*55}")

    # Load dense checkpoint
    ckpt_path = Path(cfg.checkpoint_dir) / f"{model_name}_best.pt"
    model = build_model(model_name, cfg.num_classes, pretrained=False).to(cfg.device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  WARNING — no checkpoint. Run: python train.py --model {model_name}")

    # Prune
    model = apply_global_pruning(model, sparsity)
    actual_sparsity = get_actual_sparsity(model)
    print(f"  Actual sparsity after pruning: {actual_sparsity*100:.1f}%")

    # Eval immediately after pruning (before fine-tuning)
    pre_metrics = evaluate_model(model, test_loader, cfg.device, cfg.class_names)
    print(f"  Acc before fine-tune: {pre_metrics['accuracy']*100:.2f}%  |  Macro F1: {pre_metrics['macro_f1']:.4f}")

    # Fine-tune
    model, ft_f1 = fine_tune(model, train_loader, val_loader, cfg, n_epochs=fine_tune_epochs, fine_tune_lr=fine_tune_lr)

    # Remove masks permanently
    model = remove_pruning_masks(model)

    # Final evaluation
    post_metrics = evaluate_model(model, test_loader, cfg.device, cfg.class_names)
    print(f"  Acc after  fine-tune: {post_metrics['accuracy']*100:.2f}%  |  Macro F1: {post_metrics['macro_f1']:.4f}")

    # Save pruned checkpoint
    tag = f"{model_name}_pruned_{int(sparsity*100)}"
    save_path = Path(cfg.checkpoint_dir) / f"{tag}.pt"
    torch.save({"model_state": model.state_dict(), "sparsity": sparsity}, save_path)

    plot_confusion_matrix(
        post_metrics["confusion_matrix"], cfg.class_names,
        title=f"Confusion Matrix – {tag}",
        save_path=f"{cfg.results_dir}/pruning/{tag}_confusion_matrix.png",
    )

    return {
        "model":               model_name,
        "sparsity_target":     sparsity,
        "sparsity_actual":     round(actual_sparsity, 4),
        "acc_before_finetune": round(pre_metrics["accuracy"],  4),
        "f1_before_finetune":  round(pre_metrics["macro_f1"],  4),
        "acc_after_finetune":  round(post_metrics["accuracy"], 4),
        "f1_after_finetune":   round(post_metrics["macro_f1"], 4),
        "fine_tune_epochs":    fine_tune_epochs,
    }


# ── Sparsity sweep ────────────────────────────────────────────────────────────

def run_sweep(model_name: str, sparsities: list, cfg: Config, fine_tune_epochs: int):
    from tabulate import tabulate

    train_loader, val_loader, test_loader, _ = load_wm811k(cfg)

    # Dense baseline
    print("\n[Pruning] Evaluating dense baseline...")
    dense_model = build_model(model_name, cfg.num_classes, pretrained=False).to(cfg.device)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{model_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        dense_model.load_state_dict(ckpt["model_state"])
    baseline = evaluate_model(dense_model, test_loader, cfg.device, cfg.class_names)
    print(f"  Dense baseline — Acc: {baseline['accuracy']*100:.2f}%  Macro F1: {baseline['macro_f1']:.4f}")

    all_results = [{
        "model": model_name, "sparsity_target": 0.0, "sparsity_actual": 0.0,
        "acc_after_finetune": round(baseline["accuracy"], 4),
        "f1_after_finetune":  round(baseline["macro_f1"], 4),
        "acc_before_finetune": round(baseline["accuracy"], 4),
        "f1_before_finetune":  round(baseline["macro_f1"], 4),
        "fine_tune_epochs": 0,
    }]

    for s in sparsities:
        result = prune_and_evaluate(
            model_name, s, cfg, train_loader, val_loader, test_loader, fine_tune_epochs
        )
        all_results.append(result)

    # Summary table
    headers = ["Sparsity", "Actual %", "Acc (pre-FT)", "Acc (post-FT)", "Macro F1"]
    rows = [[
        f"{r['sparsity_target']*100:.0f}%" ,
        f"{r['sparsity_actual']*100:.1f}%",
        f"{r['acc_before_finetune']*100:.2f}%",
        f"{r['acc_after_finetune']*100:.2f}%",
        f"{r['f1_after_finetune']:.4f}",
    ] for r in all_results]

    print(f"\n{'='*65}")
    print(f"  Pruning Sweep Results — {model_name}")
    print(f"{'='*65}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Accuracy vs Sparsity plot
    sparsities_plot = [r["sparsity_actual"] * 100 for r in all_results]
    accs_pre  = [r["acc_before_finetune"] * 100 for r in all_results]
    accs_post = [r["acc_after_finetune"]  * 100 for r in all_results]
    f1s       = [r["f1_after_finetune"]   * 100 for r in all_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sparsities_plot, accs_pre,  "o--", label="Accuracy (before fine-tune)", color="tomato")
    ax.plot(sparsities_plot, accs_post, "s-",  label="Accuracy (after fine-tune)",  color="steelblue")
    ax.plot(sparsities_plot, f1s,       "^-",  label="Macro F1 (after fine-tune)",  color="seagreen")
    ax.set_xlabel("Actual Sparsity (%)")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Accuracy vs Sparsity — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f"{cfg.results_dir}/pruning/{model_name}_pruning_sweep.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {plot_path}")

    # Save JSON
    out_path = Path(cfg.results_dir) / "pruning" / f"{model_name}_pruning_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Pruning] Results saved → {out_path}")

    return all_results


# ── Optuna joint search (sparsity + fine-tune hyperparams) ──────────────────

def run_optuna_prune(
    model_name: str,
    cfg: Config,
    n_trials: int = 40,
    f1_tolerance: float = 0.005,
) -> None:
    """
    Jointly optimise pruning sparsity + fine-tuning hyperparams with Optuna.
    Objective  : maximise val macro-F1.
    Best result: highest sparsity among trials with F1 >= baseline - f1_tolerance.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    train_loader, val_loader, _, class_weights = load_wm811k(cfg)
    cw = class_weights.to(cfg.device)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{model_name}_best.pt"

    # Baseline F1 on val set
    _base = build_model(model_name, cfg.num_classes, pretrained=False).to(cfg.device)
    _base.load_state_dict(torch.load(ckpt_path, map_location=cfg.device)["model_state"])
    _base.eval()
    _preds, _labels = [], []
    with torch.no_grad():
        for imgs, lbs in val_loader:
            _preds.extend(_base(imgs.to(cfg.device)).argmax(1).cpu().numpy())
            _labels.extend(lbs.numpy())
    baseline_f1 = f1_score(_labels, _preds, average="macro", zero_division=0)
    del _base
    print(f"\n  Baseline val F1 = {baseline_f1:.4f}  (tolerance = -{f1_tolerance:.4f})")

    criterion = nn.CrossEntropyLoss(weight=cw)

    def objective(trial: optuna.Trial) -> float:
        sparsity   = trial.suggest_float("sparsity",        0.10, 0.90)
        lr         = trial.suggest_float("fine_tune_lr",    1e-5, 1e-2, log=True)
        ft_epochs  = trial.suggest_int(  "fine_tune_epochs", 5,   20)
        sched_name = trial.suggest_categorical("scheduler", ["cosine", "step", "none"])

        model = build_model(model_name, cfg.num_classes, pretrained=False).to(cfg.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device)["model_state"])
        model = apply_global_pruning(model, sparsity)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=lr * 0.01)
        elif sched_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, ft_epochs // 3), gamma=0.1)
        else:
            sched = None

        best_f1 = 0.0
        for epoch in range(1, ft_epochs + 1):
            model.train()
            for imgs, lbs in train_loader:
                imgs, lbs = imgs.to(cfg.device), lbs.to(cfg.device)
                optimizer.zero_grad()
                criterion(model(imgs), lbs).backward()
                optimizer.step()
            if sched:
                sched.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for imgs, lbs in val_loader:
                    preds_all.extend(model(imgs.to(cfg.device)).argmax(1).cpu().numpy())
                    labels_all.extend(lbs.numpy())
            val_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
            best_f1 = max(best_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_f1

    study_name   = f"prune_optuna_{model_name}"
    storage_path = Path(cfg.results_dir) / "optuna" / f"{study_name}.db"
    storage_url  = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage_url,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    # Best = highest sparsity with F1 within tolerance
    threshold    = baseline_f1 - f1_tolerance
    valid        = [t for t in study.trials if t.value is not None and t.value >= threshold]
    best_trial   = max(valid, key=lambda t: t.params["sparsity"]) if valid else study.best_trial

    print(f"\n{'='*65}")
    print(f"  Optuna Pruning Search — {model_name}")
    print(f"  Baseline F1   : {baseline_f1:.4f}  |  Tolerance: -{f1_tolerance:.4f}")
    print(f"  Valid trials  : {len(valid)} / {len(study.trials)}")
    print(f"  Best trial #{best_trial.number}")
    for k, v in best_trial.params.items():
        print(f"    {k:<20}: {v}")
    print(f"  Best val F1   : {best_trial.value:.4f}")
    print(f"{'='*65}")

    out = {
        "model":          model_name,
        "baseline_f1":    round(baseline_f1, 4),
        "f1_tolerance":   f1_tolerance,
        "best_trial":     best_trial.number,
        "best_params":    best_trial.params,
        "best_val_f1":    round(best_trial.value, 4),
        "n_valid_trials": len(valid),
        "all_trials": [
            {"number": t.number, "value": t.value,
             "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    results_path = Path(cfg.results_dir) / "pruning" / f"{model_name}_optuna_prune_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] → {results_path}")

    # Pareto scatter: sparsity vs F1
    try:
        completed = [t for t in study.trials if t.value is not None]
        xs = [t.params["sparsity"] * 100 for t in completed]
        ys = [t.value * 100            for t in completed]
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(xs, ys, c=ys, cmap="RdYlGn", edgecolors="k", linewidths=0.4, s=60)
        ax.axhline(baseline_f1 * 100, color="steelblue", linestyle="--", label=f"Baseline F1 ({baseline_f1*100:.2f}%)")
        ax.axhline(threshold * 100,  color="tomato",    linestyle=":",  label=f"Threshold ({threshold*100:.2f}%)")
        best_x = best_trial.params["sparsity"] * 100
        best_y = best_trial.value * 100
        ax.scatter([best_x], [best_y], marker="*", s=250, color="gold", edgecolors="k", zorder=5, label=f"Best ({best_x:.0f}%, F1={best_y:.2f}%)")
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel("Val Macro F1 (%)")
        ax.set_title(f"Optuna Pruning Search — {model_name}", fontweight="bold")
        ax.legend(fontsize=9)
        plt.colorbar(sc, ax=ax, label="Val F1 (%)")
        plt.tight_layout()
        plot_path = Path(cfg.results_dir) / "pruning" / f"{model_name}_optuna_prune_scatter.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] Plot → {plot_path}")
    except Exception as e:
        print(f"[Warning] Could not save scatter plot: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Pruning for WM-811K wafer defect models")
    p.add_argument("--model",     default="resnet18",
                   help="Architecture to prune (ignored when --all is set)")
    p.add_argument("--all",       action="store_true",
                   help="Prune all 5 architectures sequentially")
    p.add_argument("--sparsity",  type=float, default=0.5,
                   help="Single sparsity level (ignored if --sweep / --find_optimal)")
    p.add_argument("--sparsities", nargs="+", type=float, metavar="S",
                   help="Custom sparsity list, e.g. --sparsities 0.1 0.2 0.3 (overrides --sweep)")
    p.add_argument("--sweep",     action="store_true",
                   help="Sweep over 30%%, 50%%, 70%% sparsity")
    p.add_argument("--find_optimal", action="store_true",
                   help="Sweep 10%%–90%% in steps of 10%% and auto-pick the best sparsity "
                        "(peak F1 within --f1_tolerance of baseline)")
    p.add_argument("--f1_tolerance", type=float, default=0.005,
                   help="Max allowed F1 drop from baseline when picking optimal sparsity (default 0.005)")
    p.add_argument("--fine_tune_epochs", type=int, default=5)
    p.add_argument("--fine_tune_lr", type=float, default=None,
                   help="Override fine-tune LR (default: cfg.lr * 0.1)")
    p.add_argument("--optuna_prune", action="store_true",
                   help="Joint Optuna search over sparsity + fine-tune hyperparams")
    p.add_argument("--prune_trials",  type=int, default=40,
                   help="Optuna trials for --optuna_prune (default: 40)")
    p.add_argument("--quick",     action="store_true",
                   help="2 fine-tune epochs per level (demo)")
    p.add_argument("--data_path", default=None)
    return p.parse_args()


ALL_PRUNE_MODELS = ["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2"]


def _find_optimal_sparsity(results: list, f1_tolerance: float) -> dict:
    """Pick the highest sparsity whose post-finetune F1 is within f1_tolerance of baseline."""
    baseline_f1 = results[0]["f1_after_finetune"]   # 0% sparsity entry
    threshold   = baseline_f1 - f1_tolerance
    # Walk from highest to lowest sparsity; take the first that clears the threshold
    candidates = [r for r in results[1:] if r["f1_after_finetune"] >= threshold]
    if not candidates:
        best = results[1]   # fallback: lowest sparsity tested
    else:
        best = max(candidates, key=lambda r: r["sparsity_actual"])
    return best


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    if args.data_path: cfg.data_path = args.data_path
    ft_epochs = 2 if args.quick else args.fine_tune_epochs

    models_to_prune = ALL_PRUNE_MODELS if args.all else [args.model]

    for model_name in models_to_prune:
        if args.optuna_prune:
            run_optuna_prune(model_name, cfg, n_trials=args.prune_trials, f1_tolerance=args.f1_tolerance)
        elif args.find_optimal:
            fine_sweep = [round(s * 0.1, 1) for s in range(1, 10)]   # 0.1 … 0.9
            all_results = run_sweep(model_name, fine_sweep, cfg, ft_epochs)
            best = _find_optimal_sparsity(all_results, args.f1_tolerance)
            print(f"\n{'='*65}")
            print(f"  Optimal sparsity for {model_name}")
            print(f"  Baseline F1  : {all_results[0]['f1_after_finetune']:.4f}")
            print(f"  Tolerance    : -{args.f1_tolerance:.4f}")
            print(f"  Best sparsity: {best['sparsity_actual']*100:.0f}%  "
                  f"→ F1={best['f1_after_finetune']:.4f}  "
                  f"Acc={best['acc_after_finetune']*100:.2f}%")
            print(f"{'='*65}")
            # Append recommendation to the JSON
            results_path = Path(cfg.results_dir) / "pruning" / f"{model_name}_pruning_results.json"
            import json as _json
            existing = _json.loads(results_path.read_text()) if results_path.exists() else all_results
            rec = {"optimal_recommendation": best}
            results_path.write_text(_json.dumps(existing + [rec] if isinstance(existing, list) else existing, indent=2))
        elif args.sparsities:
            run_sweep(model_name, args.sparsities, cfg, ft_epochs)
        elif args.sweep:
            run_sweep(model_name, [0.3, 0.5, 0.7], cfg, ft_epochs)
        else:
            train_loader, val_loader, test_loader, _ = load_wm811k(cfg)
            prune_and_evaluate(
                model_name, args.sparsity, cfg,
                train_loader, val_loader, test_loader,
                fine_tune_epochs=ft_epochs,
                fine_tune_lr=args.fine_tune_lr,
            )
