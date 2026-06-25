"""
End-to-end pipeline: hyperparameter optimisation → retrain → optimal pruning.

Steps
-----
  1. Hyperparameter optimisation (Optuna) for every model that doesn't yet have
     a completed study (skips models whose results JSON already exists).
  2. Retrain each model using the best hyperparameters found in step 1.
  3. Find the optimal pruning sparsity for ResNet-18 (or --prune_model).

Usage
-----
    python pipeline.py                        # full pipeline, all models
    python pipeline.py --skip_done            # skip models already in results/
    python pipeline.py --opt_only             # step 1 only
    python pipeline.py --prune_model resnet18 --f1_tolerance 0.005

Outputs (per model)
-------------------
    results/optuna_<model>_results.json       best hyperparams
    checkpoints/<model>_best.pt               retrained checkpoint
    results/<model>_pruning_results.json      pruning sweep + recommendation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ALL_MODELS = ["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2"]
PYTHON = sys.executable
RESULTS = Path("results")
OPTUNA  = RESULTS / "optuna"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str):
    print(f"\n{'#'*65}")
    print(f"  PIPELINE  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'#'*65}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[Pipeline WARNING] '{label}' exited with code {result.returncode}. Continuing...")


def _optuna_done(model: str) -> bool:
    """True if a completed Optuna results JSON exists for this model."""
    return (OPTUNA / f"optuna_{model}_results.json").exists()


def _load_best_params(model: str) -> dict:
    """Load best hyperparameters from an Optuna results JSON."""
    path = OPTUNA / f"optuna_{model}_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("best_params", {})


def _checkpoint_exists(model: str) -> bool:
    return (Path("checkpoints") / f"{model}_best.pt").exists()


# ── Step 1: Hyperparameter optimisation ──────────────────────────────────────

def step_optimise(models: list[str], trials: int, epochs: int, skip_done: bool):
    pending = [m for m in models if not (skip_done and _optuna_done(m))]
    if not pending:
        print("[Pipeline] All models already optimised — skipping Step 1.")
        return
    print(f"[Pipeline] Step 1 — Optimising: {pending}")
    _run(
        [PYTHON, "optimize.py", "--models"] + pending +
        ["--trials", str(trials), "--epochs", str(epochs)],
        "Hyperparameter optimisation",
    )


# ── Step 2: Retrain with best hyperparameters ─────────────────────────────────

def step_retrain(models: list[str], base_epochs: int, skip_done: bool):
    print(f"\n[Pipeline] Step 2 — Retraining all models with optimised hyperparameters")
    for model in models:
        if skip_done and _checkpoint_exists(model):
            print(f"  [{model}] checkpoint already exists — skipping retrain.")
            continue

        params = _load_best_params(model)
        if not params:
            print(f"  [{model}] No Optuna results found — using default hyperparams.")

        cmd = [PYTHON, "train.py", "--model", model, "--epochs", str(base_epochs)]
        if "lr" in params:
            cmd += ["--lr", str(params["lr"])]
        if "weight_decay" in params:
            cmd += ["--weight_decay", str(params["weight_decay"])]
        if "batch_size" in params:
            cmd += ["--batch_size", str(int(params["batch_size"]))]
        if "scheduler" in params:
            cmd += ["--scheduler", params["scheduler"]]
        _run(cmd, f"Retrain {model}")


# ── Step 3: Find optimal pruning sparsity ─────────────────────────────────────

def step_find_optimal_pruning(model: str, ft_epochs: int, f1_tolerance: float):
    print(f"\n[Pipeline] Step 3 — Finding optimal pruning sparsity for {model}")
    _run(
        [PYTHON, "prune.py",
         "--model",        model,
         "--find_optimal",
         "--fine_tune_epochs", str(ft_epochs),
         "--f1_tolerance", str(f1_tolerance)],
        f"Optimal pruning — {model}",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="WM-811K end-to-end optimisation pipeline")
    p.add_argument("--models",       nargs="+", default=ALL_MODELS,
                   help="Models to process (default: all 5)")
    p.add_argument("--opt_only",     action="store_true",
                   help="Run only Step 1 (hyperparameter optimisation)")
    p.add_argument("--skip_done",    action="store_true",
                   help="Skip steps whose outputs already exist on disk")
    # Optuna settings
    p.add_argument("--trials",       type=int, default=30,
                   help="Optuna trials per model (default: 30)")
    p.add_argument("--opt_epochs",   type=int, default=15,
                   help="Epochs per Optuna trial (default: 15)")
    # Retraining settings
    p.add_argument("--train_epochs", type=int, default=40,
                   help="Full training epochs after optimisation (default: 40)")
    # Pruning settings
    p.add_argument("--prune_model",  default="resnet18",
                   help="Model to run optimal-pruning search on (default: resnet18)")
    p.add_argument("--prune_ft_epochs", type=int, default=10,
                   help="Fine-tune epochs per pruning level (default: 10)")
    p.add_argument("--f1_tolerance", type=float, default=0.005,
                   help="Max F1 drop allowed when picking optimal sparsity (default: 0.005)")
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"\n{'='*65}")
    print(f"  WM-811K Optimisation Pipeline")
    print(f"  Models  : {args.models}")
    print(f"  Trials  : {args.trials}  |  Opt epochs: {args.opt_epochs}")
    print(f"  Train epochs: {args.train_epochs}")
    print(f"  Prune model : {args.prune_model}  (tolerance: {args.f1_tolerance})")
    print(f"{'='*65}")

    # Step 1 — Hyperparameter optimisation
    step_optimise(args.models, args.trials, args.opt_epochs, args.skip_done)

    if args.opt_only:
        print("\n[Pipeline] --opt_only set — stopping after Step 1.")
        return

    # Step 2 — Retrain with best params
    step_retrain(args.models, args.train_epochs, args.skip_done)

    # Step 3 — Find optimal pruning for the designated model
    step_find_optimal_pruning(args.prune_model, args.prune_ft_epochs, args.f1_tolerance)

    print(f"\n{'='*65}")
    print(f"  Pipeline complete.")
    print(f"  Optimised checkpoints → checkpoints/<model>_best.pt")
    print(f"  Optuna results        → results/optuna_<model>_results.json")
    print(f"  Pruning report        → results/{args.prune_model}_pruning_results.json")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
