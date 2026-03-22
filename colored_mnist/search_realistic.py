"""
search_realistic.py — Hyperparameter search with realistic model selection.

Extends search.py with two additional selection strategies from:
"In Search of Lost Domain Generalization" (Gulrajani & Lopez-Paz, 2021)

Selection methods:
    --selection_method oracle
        Use test env (e=0.9) for selection — same as original search.py
        Score = min(env0_acc, env1_acc, test_acc)

    --selection_method train_domain_val
        Hold out 20% of each train env for validation
        Score = min(val_env0_acc, val_env1_acc)
        Test acc is recorded but NEVER used for selection

    --selection_method leave_one_domain_out
        Each train env acts as validation for the other
        Score = min(loo_env0_acc, loo_env1_acc)
        Test acc is recorded but NEVER used for selection

Usage
-----
    python search_realistic.py --mode irm --selection_method train_domain_val --n_trials 50
    python search_realistic.py --mode irm --selection_method leave_one_domain_out --n_trials 50
    python search_realistic.py --mode erm --selection_method train_domain_val
    python search_realistic.py --mode grayscale --selection_method train_domain_val --n_trials 50
"""

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

# Import search spaces and fixed configs from original search.py
from search import (
    sample_irm,
    sample_grayscale,
    ERM_FIXED,
    STEPS_GRID,
)

# Import realistic data loaders and training loop
from main_realistic import (
    load_colored_mnist_train_domain_val,
    load_colored_mnist_leave_one_domain_out,
    train_one_restart_realistic,
)


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------

def selection_score_oracle(env0_acc, env1_acc, val0_acc, val1_acc, test_acc):
    """Original criterion — uses test env (unrealistic)."""
    return min(env0_acc, env1_acc, test_acc)


def selection_score_train_domain_val(env0_acc, env1_acc, val0_acc, val1_acc, test_acc):
    """Realistic — uses held-out portion of train envs."""
    return min(val0_acc, val1_acc)


def selection_score_leave_one_domain_out(env0_acc, env1_acc, val0_acc, val1_acc, test_acc):
    """Realistic — uses left-out train env as validation."""
    return min(val0_acc, val1_acc)


SELECTION_METHODS = {
    "oracle":               selection_score_oracle,
    "train_domain_val":     selection_score_train_domain_val,
    "leave_one_domain_out": selection_score_leave_one_domain_out,
}

DATA_LOADERS = {
    "oracle":               load_colored_mnist_train_domain_val,  # val split exists but ignored
    "train_domain_val":     load_colored_mnist_train_domain_val,
    "leave_one_domain_out": load_colored_mnist_leave_one_domain_out,
}


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_hparams(hparams, envs, device, seed=0):
    """
    Train once and return all accuracies.
    Returns (env0_acc, env1_acc, val0_acc, val1_acc, test_acc).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    flags = SimpleNamespace(**hparams)
    return train_one_restart_realistic(flags, envs, device, verbose=False)


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  Selection: {args.selection_method}  |  Trials: {args.n_trials}  |  Seed: {args.seed}\n")

    rng          = np.random.default_rng(args.seed)
    score_fn     = SELECTION_METHODS[args.selection_method]
    data_loader  = DATA_LOADERS[args.selection_method]

    # Load environments once and reuse across trials
    envs = data_loader(device)

    best_score   = -np.inf
    best_hparams = None
    results      = []

    if args.mode == "erm":
        candidates = [ERM_FIXED]
    else:
        sampler    = sample_irm if args.mode == "irm" else sample_grayscale
        candidates = [sampler(rng) for _ in range(args.n_trials)]

    for trial_idx, hparams in enumerate(candidates):
        t0 = time.time()
        e0, e1, v0, v1, te = evaluate_hparams(
            hparams, envs, device, seed=args.seed + trial_idx
        )
        score   = score_fn(e0, e1, v0, v1, te)
        elapsed = time.time() - t0

        result = {
            "trial":    trial_idx,
            "score":    score,
            "env0_acc": e0,
            "env1_acc": e1,
            "val0_acc": v0,
            "val1_acc": v1,
            "test_acc": te,
            "elapsed":  elapsed,
            **hparams,
        }
        results.append(result)

        marker = " ← best" if score > best_score else ""
        print(
            f"[{trial_idx:3d}]  score={score:.4f}  "
            f"e0={e0:.4f}  e1={e1:.4f}  "
            f"v0={v0:.4f}  v1={v1:.4f}  "
            f"te={te:.4f}  ({elapsed:.1f}s){marker}"
        )

        if score > best_score:
            best_score   = score
            best_hparams = hparams

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Best score ({args.mode} / {args.selection_method}): {best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in sorted(best_hparams.items()):
        print(f"    {k}: {v}")

    # Save results
    out_path = f"search_results_{args.mode}_{args.selection_method}.json"
    with open(out_path, "w") as f:
        json.dump({"best": best_hparams, "all_trials": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print best command
    cmd = (
        f"python main.py \\\n"
        f"  --hidden_dim={best_hparams['hidden_dim']} \\\n"
        f"  --l2_regularizer_weight={best_hparams['l2_regularizer_weight']} \\\n"
        f"  --lr={best_hparams['lr']} \\\n"
        f"  --penalty_anneal_iters={best_hparams['penalty_anneal_iters']} \\\n"
        f"  --penalty_weight={best_hparams['penalty_weight']} \\\n"
        f"  --steps={best_hparams['steps']}"
    )
    if best_hparams.get("grayscale_model"):
        cmd += " \\\n  --grayscale_model"
    print(f"\nBest command:\n\n{cmd}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic hyperparameter search for IRM")
    parser.add_argument("--mode",             choices=["irm", "grayscale", "erm"], default="irm")
    parser.add_argument("--selection_method", choices=["oracle", "train_domain_val", "leave_one_domain_out"],
                        default="train_domain_val")
    parser.add_argument("--n_trials",         type=int, default=50)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()
    run_search(args)
