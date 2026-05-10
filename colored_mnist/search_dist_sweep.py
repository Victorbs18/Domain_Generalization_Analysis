# search_dist_sweep.py — Hyperparameter search for Experiment 7.
#
# Searches for the best hyperparameters for a given (mode, e_train,
# selection_method) combination using random search, consistent with
# all other search scripts in this project.
#
# Usage
# -----
#   python search_dist_sweep.py --mode irm --e_train 0.2
#       --selection_method oracle --n_trials 50
#
#   # Run all combinations
#   run_dist_sweep_search.bat

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

from search import sample_irm
from main_dist_sweep import (
    E_VALUES,
    load_environments,
    train_one_restart,
)


# ---------------------------------------------------------------------------
# Selection criteria (consistent with all other search scripts)
# ---------------------------------------------------------------------------

def score_oracle(train_acc, val_acc, test_acc):
    return min(train_acc, test_acc)


def score_train_domain_val(train_acc, val_acc, test_acc):
    return val_acc


SCORE_FNS = {
    "oracle":           score_oracle,
    "train_domain_val": score_train_domain_val,
}


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_hparams(hparams, data, device, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    flags   = SimpleNamespace(**hparams)
    results = train_one_restart(flags, data, device, verbose=False)
    return results


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_fn = SCORE_FNS[args.selection_method]

    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  "
          f"e_train: {args.e_train}  |  "
          f"Selection: {args.selection_method}  |  "
          f"Trials: {args.n_trials}  |  "
          f"Seed: {args.seed}\n")

    rng  = np.random.default_rng(args.seed)
    data = load_environments(args.e_train, args.selection_method, device)

    if args.mode == "irm":
        candidates = [sample_irm(rng) for _ in range(args.n_trials)]
    else:
        # ERM — fixed hyperparameters, no penalty terms
        candidates = [{
            "hidden_dim":            256,
            "lr":                    0.001,
            "l2_regularizer_weight": 0.001,
            "penalty_anneal_iters":  0,
            "penalty_weight":        0.0,
            "steps":                 501,
            "mode":                  "erm",
            "grayscale_model":       False,
        }]

    best_score   = -np.inf
    best_hparams = None
    results      = []

    for trial_idx, hparams in enumerate(candidates):
        hparams["mode"]            = args.mode
        hparams["grayscale_model"] = False

        t0      = time.time()
        res     = evaluate_hparams(
            hparams, data, device, seed=args.seed + trial_idx)
        score   = score_fn(
            res["train_acc"], res["val_acc"], res["test_acc"])
        elapsed = time.time() - t0

        result = {
            "trial":      trial_idx,
            "score":      score,
            "train_acc":  res["train_acc"],
            "val_acc":    res["val_acc"],
            "test_acc":   res["test_acc"],
            "elapsed":    elapsed,
            **hparams,
        }
        results.append(result)

        marker = " << best" if score > best_score else ""
        print(
            f"[{trial_idx:3d}]  score={score:.4f}  "
            f"train={res['train_acc']:.4f}  "
            f"val={res['val_acc']:.4f}  "
            f"test={res['test_acc']:.4f}  "
            f"({elapsed:.1f}s){marker}"
        )

        if score > best_score:
            best_score   = score
            best_hparams = hparams

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Best score (mode={args.mode} / "
          f"e_train={args.e_train} / "
          f"{args.selection_method}): {best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in sorted(best_hparams.items()):
        print(f"    {k}: {v}")

    out_path = (f"search_results_dist_sweep"
                f"_{args.mode}"
                f"_e{args.e_train:.1f}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"best": best_hparams, "all_trials": results,
             "mode": args.mode, "e_train": args.e_train,
             "selection_method": args.selection_method},
            f, indent=2)
    print(f"\nResults saved to {out_path}")

    cmd = (
        f"python main_dist_sweep.py \\\n"
        f"  --mode={args.mode} \\\n"
        f"  --e_train={args.e_train} \\\n"
        f"  --selection_method={args.selection_method} \\\n"
        f"  --hidden_dim={best_hparams['hidden_dim']} \\\n"
        f"  --l2_regularizer_weight="
        f"{best_hparams['l2_regularizer_weight']} \\\n"
        f"  --lr={best_hparams['lr']} \\\n"
        f"  --penalty_anneal_iters="
        f"{best_hparams['penalty_anneal_iters']} \\\n"
        f"  --penalty_weight={best_hparams['penalty_weight']} \\\n"
        f"  --steps={best_hparams['steps']} \\\n"
        f"  --n_restarts=10"
    )
    print(f"\nBest command:\n\n{cmd}")

    return best_hparams, best_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search — Distribution Distance Sweep (Experiment 7)")
    parser.add_argument("--mode",             type=str,   default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--e_train",          type=float, default=0.2,
                        choices=E_VALUES)
    parser.add_argument("--selection_method", type=str,   default="oracle",
                        choices=["oracle", "train_domain_val"])
    parser.add_argument("--n_trials",         type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()
    run_search(args)