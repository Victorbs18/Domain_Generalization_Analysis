"""
search_texture.py — Hyperparameter search for two correlated spurious
features experiment.

Tests whether IRM can recover the invariant feature (digit shape) when
TWO spurious features (color + texture) vary together across environments.

Configs:
    original:  e_values=[0.1, 0.2]  — same as original IRM paper
    diverse:   e_values=[0.1, 0.5]  — high diversity
    proximate: e_values=[0.7, 0.8]  — high proximity to test (e=0.9)

Usage
-----
    python search_texture.py --config original --mode irm --selection_method oracle --n_trials 50
    python search_texture.py --config original --mode erm --selection_method oracle

    # Run all combinations
    run_texture_search.bat
"""

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

from search import sample_irm
from main_texture import (
    load_environments,
    train_one_restart,
    ENV_CONFIGS,
)


# ---------------------------------------------------------------------------
# ERM fixed hyperparameters
# ---------------------------------------------------------------------------

ERM_FIXED = {
    "hidden_dim":            256,
    "l2_regularizer_weight": 0.001,
    "lr":                    0.001,
    "penalty_anneal_iters":  0,
    "penalty_weight":        0.0,
    "steps":                 501,
    "grayscale_model":       False,
    "mode":                  "erm",
}


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------

def score_oracle(train_accs, val_accs, test_acc):
    return min(min(train_accs), test_acc)


def score_train_domain_val(train_accs, val_accs, test_acc):
    return min(val_accs)


def score_leave_one_domain_out(train_accs, val_accs, test_acc):
    return min(val_accs)


SCORE_FNS = {
    "oracle":               score_oracle,
    "train_domain_val":     score_train_domain_val,
    "leave_one_domain_out": score_leave_one_domain_out,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = ENV_CONFIGS[args.config]

    print(f"Device: {device}")
    print(f"Config: {args.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Description: {cfg['description']}")
    print(f"Mode: {args.mode}  |  "
          f"Selection: {args.selection_method}  |  "
          f"Trials: {args.n_trials}  |  "
          f"Seed: {args.seed}\n")

    score_fn = SCORE_FNS[args.selection_method]
    data     = load_environments(args.config, args.selection_method, device)

    # ------------------------------------------------------------------
    # ERM — fixed hyperparameters, no search needed
    # ------------------------------------------------------------------
    if args.mode == "erm":
        hparams = ERM_FIXED.copy()
        res     = evaluate_hparams(hparams, data, device, seed=args.seed)
        score   = score_fn(res["train_accs"], res["val_accs"], res["test_acc"])

        print(f"ERM (fixed hparams)")
        print(f"  score={score:.4f}  "
              f"train={res['mean_train']:.4f}  "
              f"val={res['mean_val']:.4f}  "
              f"te={res['test_acc']:.4f}")

        out_path = (f"search_results_texture"
                    f"_config{args.config}"
                    f"_erm"
                    f"_{args.selection_method}.json")
        with open(out_path, "w") as f:
            json.dump(
                {"best": hparams, "all_trials": [res],
                 "config": args.config, "cfg": cfg,
                 "mode": "erm",
                 "selection_method": args.selection_method},
                f, indent=2)
        print(f"\nResults saved to {out_path}")
        return hparams, score

    # ------------------------------------------------------------------
    # IRM — random search
    # ------------------------------------------------------------------
    rng          = np.random.default_rng(args.seed)
    candidates   = [sample_irm(rng) for _ in range(args.n_trials)]
    # Add mode to each candidate
    for c in candidates:
        c["mode"] = "irm"

    best_score   = -np.inf
    best_hparams = None
    results      = []

    for trial_idx, hparams in enumerate(candidates):
        t0      = time.time()
        res     = evaluate_hparams(
            hparams, data, device, seed=args.seed + trial_idx)
        score   = score_fn(
            res["train_accs"], res["val_accs"], res["test_acc"])
        elapsed = time.time() - t0

        result = {
            "trial":      trial_idx,
            "score":      score,
            "mean_train": res["mean_train"],
            "mean_val":   res["mean_val"],
            "test_acc":   res["test_acc"],
            "train_accs": res["train_accs"],
            "val_accs":   res["val_accs"],
            "elapsed":    elapsed,
            **hparams,
        }
        results.append(result)

        marker = " << best" if score > best_score else ""
        print(
            f"[{trial_idx:3d}]  score={score:.4f}  "
            f"train={res['mean_train']:.4f}  "
            f"val={res['mean_val']:.4f}  "
            f"te={res['test_acc']:.4f}  "
            f"({elapsed:.1f}s){marker}"
        )

        if score > best_score:
            best_score   = score
            best_hparams = hparams

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Best score ({args.config} / irm / {args.selection_method}): "
          f"{best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in sorted(best_hparams.items()):
        print(f"    {k}: {v}")

    out_path = (f"search_results_texture"
                f"_config{args.config}"
                f"_irm"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"best": best_hparams, "all_trials": results,
             "config": args.config, "cfg": cfg,
             "mode": "irm",
             "selection_method": args.selection_method},
            f, indent=2)
    print(f"\nResults saved to {out_path}")

    cmd = (
        f"python main_texture.py \\\n"
        f"  --config={args.config} \\\n"
        f"  --mode=irm \\\n"
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
        description="Search — two correlated spurious features")
    parser.add_argument("--config",           type=str, default="original",
                        choices=["original", "diverse", "proximate"])
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_trials",         type=int, default=50)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()
    run_search(args)