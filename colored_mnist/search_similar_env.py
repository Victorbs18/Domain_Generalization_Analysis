"""
search_similar_env.py — Hyperparameter search for similar environment range experiment.

Tests whether more environments help IRM when e values stay within
the SAME fixed range {0.1, 0.2} — isolating the pure quantity effect
from the diversity/proximity effect of main_env.py.

Usage
-----
    python search_similar_env.py --n_envs 2 --selection_method oracle --n_trials 50
    
    # Run all combinations
    run_similar_env_search.bat
"""

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

from search import sample_irm, STEPS_GRID
from main_similar_env import load_environments, train_one_restart, ENV_E_VALUES


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
    print(f"Device: {device}")
    print(f"N envs: {args.n_envs}  |  "
          f"e values: {ENV_E_VALUES[args.n_envs]}  |  "
          f"Range: fixed {{0.1, 0.2}}  |  "
          f"Selection: {args.selection_method}  |  "
          f"Trials: {args.n_trials}  |  "
          f"Seed: {args.seed}\n")

    rng      = np.random.default_rng(args.seed)
    score_fn = SCORE_FNS[args.selection_method]

    # Load environments once and reuse across all trials
    data = load_environments(args.n_envs, args.selection_method, device)

    candidates   = [sample_irm(rng) for _ in range(args.n_trials)]
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
    print(f"Best score (n_envs={args.n_envs} / {args.selection_method}): "
          f"{best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in sorted(best_hparams.items()):
        print(f"    {k}: {v}")

    out_path = (f"search_results_similar_env{args.n_envs}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"best": best_hparams, "all_trials": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    cmd = (
        f"python main_similar_env.py \\\n"
        f"  --n_envs={args.n_envs} \\\n"
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
        description="Search — similar environment range IRM")
    parser.add_argument("--n_envs",           type=int, default=2,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_trials",         type=int, default=50)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()
    run_search(args)