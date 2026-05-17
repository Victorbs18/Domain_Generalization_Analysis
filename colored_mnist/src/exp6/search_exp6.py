# search_exp6.py — Hyperparameter search for Experiment 6.
#
# Searches for the best IRM hyperparameters for a given
# (e2, selection_method) combination using random search.
# ERM uses fixed hyperparameters (no search needed).
#
# Usage
#   python src/exp6/search_exp6.py --mode irm --e2 0.5
#       --selection_method oracle --n_trials 50
#
# Output
#   results/exp6/search_results_{mode}_e{e2}_{selection_method}.json

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset    import load_mnist
from utils.trainer    import train_one_restart
from utils.selection  import load_environments
from utils.search     import sample_irm, ERM_FIXED
from utils.losses     import mean_accuracy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E1       = 0.1
E2_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEST_E   = 0.9


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_oracle(train_accs, val_accs, test_acc):
    return min(min(train_accs), test_acc)


def score_train_domain_val(train_accs, val_accs, test_acc):
    return min(val_accs)


def score_leave_one_domain_out(fold_val_accs):
    return min(fold_val_accs)


# ---------------------------------------------------------------------------
# Evaluate one set of hyperparameters
# ---------------------------------------------------------------------------

def evaluate(hparams, data, device, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    flags   = SimpleNamespace(**hparams)
    results = train_one_restart(flags, data["train"], device, verbose=False)

    # Evaluate on test env
    model = results["model"]
    model.eval()
    with torch.no_grad():
        test_logits = model(data["test"]["images"])
        test_acc    = mean_accuracy(
            test_logits, data["test"]["labels"]).item()

    # Evaluate on val env(s)
    if data["folds"] is not None:
        # LOO: run one fold per held-out env, collect val accs
        fold_val_accs = []
        for fold in data["folds"]:
            torch.manual_seed(seed)
            fold_res = train_one_restart(
                flags, fold["train"], device, verbose=False)
            fold_model = fold_res["model"]
            fold_model.eval()
            with torch.no_grad():
                fold_logits  = fold_model(fold["val"]["images"])
                fold_val_acc = mean_accuracy(
                    fold_logits, fold["val"]["labels"]).item()
            fold_val_accs.append(fold_val_acc)
        val_accs = fold_val_accs
    else:
        val_envs = data["val"] if isinstance(data["val"], list) \
            else [data["val"]]
        model.eval()
        val_accs = []
        with torch.no_grad():
            for ve in val_envs:
                vl  = model(ve["images"])
                val_accs.append(
                    mean_accuracy(vl, ve["labels"]).item())

    return results["train_accs"], val_accs, test_acc


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}  |  e2: {args.e2}  |  "
          f"Selection: {args.selection_method}  |  "
          f"Trials: {args.n_trials}")

    train_pool, test_pool = load_mnist()
    e_values = [E1, args.e2]

    data = load_environments(
        selection_method=args.selection_method,
        train_pool=train_pool,
        test_pool=test_pool,
        e_values=e_values,
        device=device,
    )
    # Always use e=0.9 as test
    from utils.dataset import make_environment
    data["test"] = make_environment(
        test_pool[0], test_pool[1], TEST_E, device)

    rng = np.random.default_rng(args.seed)

    if args.mode == "irm":
        candidates = [sample_irm(rng) for _ in range(args.n_trials)]
    else:
        candidates = [ERM_FIXED.copy()]

    best_score   = -np.inf
    best_hparams = None

    for trial_idx, hparams in enumerate(candidates):
        hparams["mode"]            = args.mode
        hparams["grayscale_model"] = False

        t0          = time.time()
        train_accs, val_accs, test_acc = evaluate(
            hparams, data, device, seed=args.seed + trial_idx)
        elapsed = time.time() - t0

        if args.selection_method == "oracle":
            score = score_oracle(train_accs, val_accs, test_acc)
        elif args.selection_method == "train_domain_val":
            score = score_train_domain_val(train_accs, val_accs, test_acc)
        else:
            score = score_leave_one_domain_out(val_accs)

        val_str = str([f"{v:.4f}" for v in val_accs])
        print(f"Trial {trial_idx:02d} | score: {score:.4f} | "
              f"val: {val_str} | test: {test_acc:.4f} | "
              f"elapsed: {elapsed:.1f}s")

        if score > best_score:
            best_score   = score
            best_hparams = hparams.copy()

    print(f"\nBest score: {best_score:.4f}")
    print(f"Best hparams: {best_hparams}")

    os.makedirs("results/exp6", exist_ok=True)
    out_path = (f"results/exp6/search_results"
                f"_{args.mode}_e{args.e2:.1f}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump({"best": best_hparams}, f, indent=2)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search — Exp 6 Distribution Distance")
    parser.add_argument("--mode",             type=str,   default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--e2",               type=float, default=0.2,
                        choices=E2_SWEEP)
    parser.add_argument("--selection_method", type=str,   default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_trials",         type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=0)
    args = parser.parse_args()
    run_search(args)
