# src/exp1/search_exp1.py
# Experiment 1: Hyperparameter search
#
# Searches for best hyperparameters for IRM, ERM and grayscale ERM under oracle model selection.
#
# Usage:
#   python src/exp1/search_exp1.py --mode irm --n_trials 50
#   python src/exp1/search_exp1.py --mode erm
#   python src/exp1/search_exp1.py --mode grayscale --n_trials 50


import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset   import load_mnist, make_environment
from utils.trainer   import train_one_restart
from utils.losses    import mean_accuracy
from utils.search    import sample_irm, sample_grayscale, ERM_FIXED
from utils.selection import score_oracle



# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Experiment 1 — Hyperparameter search")
parser.add_argument("--mode",     type=str, default="irm",
                    choices=["irm", "erm", "grayscale"])
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--seed",     type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Mode: {args.mode}  |  Trials: {args.n_trials}\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_pool, test_pool = load_mnist()
train_envs = [
    make_environment(train_pool[0][::2],  train_pool[1][::2],  0.2, device),
    make_environment(train_pool[0][1::2], train_pool[1][1::2], 0.1, device),
]
test_env = make_environment(test_pool[0], test_pool[1], 0.9, device)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
if args.mode == "erm":
    hparams = ERM_FIXED.copy()
    hparams["grayscale_model"] = False
    flags   = SimpleNamespace(**hparams)
    t0      = time.time()
    results = train_one_restart(flags, train_envs, device, verbose=False)
    with torch.no_grad():
        logits   = results["model"](test_env["images"])
        test_acc = mean_accuracy(logits, test_env["labels"]).detach().cpu().item()
    score = score_oracle(results["train_accs"], [], test_acc)
    print(f"ERM fixed hparams — score: {score:.4f}  test_acc: {test_acc:.4f}")
    out = {
        "best": hparams,
        "all_trials": [{
            "trial":    0,
            "score":    score,
            "env0_acc": results["train_accs"][0],
            "env1_acc": results["train_accs"][1],
            "test_acc": test_acc,
            "elapsed":  time.time() - t0,
            **hparams,
        }]
    }

else:
    all_trials   = []
    best_score   = -1
    best_hparams = None

    for trial in range(args.n_trials):
        hparams = (sample_irm(rng) if args.mode == "irm"
                   else sample_grayscale(rng))
        flags   = SimpleNamespace(**hparams)
        t0      = time.time()

        results = train_one_restart(flags, train_envs, device, verbose=False)

        with torch.no_grad():
            logits   = results["model"](test_env["images"])
            test_acc = mean_accuracy(logits, test_env["labels"]).detach().cpu().item()

        score   = score_oracle(results["train_accs"], [], test_acc)
        elapsed = time.time() - t0

        print(f"Trial {trial:02d} | score: {score:.4f} | "
              f"env0: {results['train_accs'][0]:.4f} | "
              f"env1: {results['train_accs'][1]:.4f} | "
              f"test: {test_acc:.4f} | "
              f"elapsed: {elapsed:.1f}s")

        all_trials.append({
            "trial":    trial,
            "score":    score,
            "env0_acc": results["train_accs"][0],
            "env1_acc": results["train_accs"][1],
            "test_acc": test_acc,
            "elapsed":  elapsed,
            **hparams,
        })

        if score > best_score:
            best_score   = score
            best_hparams = hparams

    print(f"\nBest score: {best_score:.4f}")
    print(f"Best hparams: {best_hparams}")
    out = {"best": best_hparams, "all_trials": all_trials}

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = f"results/exp1/search_results_{args.mode}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {out_path}")