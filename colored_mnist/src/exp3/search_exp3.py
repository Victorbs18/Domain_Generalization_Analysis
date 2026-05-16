# src/exp3/search_exp3.py
# Experiment 3 — Hyperparameter search for variable environments (increasing diversity)
#
# Usage:
#   python src/exp3/search_exp3.py --mode irm --n_envs 3 --selection_method train_domain_val --n_trials 50
#   python src/exp3/search_exp3.py --mode erm --n_envs 3 --selection_method oracle
#
# Note: oracle search reuses Exp 1 results for n_envs=2 only.
#       For all other n_envs oracle search is run fresh.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

from utils.dataset   import load_mnist
from utils.selection import load_environments, score_oracle, score_train_domain_val, score_leave_one_domain_out
from utils.trainer   import train_one_restart
from utils.losses    import mean_accuracy
from utils.search    import sample_irm, ERM_FIXED

# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------
ENV_E_VALUES = {
    1: [0.1],
    2: [0.1, 0.2],
    3: [0.1, 0.2, 0.3],
    4: [0.1, 0.2, 0.3, 0.4],
    5: [0.1, 0.2, 0.3, 0.4, 0.45],
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Experiment 3 — Hyperparameter search")
parser.add_argument("--mode", type=str, default="irm",
                    choices=["irm", "erm"])
parser.add_argument("--n_envs",           type=int, default=2,
                    choices=[1, 2, 3, 4, 5])
parser.add_argument("--selection_method", type=str, default="oracle",
                    choices=["oracle", "train_domain_val", "leave_one_domain_out"])
parser.add_argument("--n_trials",         type=int, default=50)
parser.add_argument("--seed",             type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Mode: {args.mode}  |  n_envs: {args.n_envs}  |  "
      f"Selection: {args.selection_method}  |  Trials: {args.n_trials}\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
e_values = ENV_E_VALUES[args.n_envs]
train_pool, test_pool = load_mnist()
data = load_environments(
    selection_method = args.selection_method,
    train_pool       = train_pool,
    test_pool        = test_pool,
    e_values         = e_values,
    device           = device,
)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
all_trials   = []
best_score   = -1
best_hparams = None

candidates = ([ERM_FIXED.copy()] if args.mode == "erm"
              else [sample_irm(rng) for _ in range(args.n_trials)])

for trial, hparams in enumerate(candidates):
    hparams["grayscale_model"] = False
    flags = SimpleNamespace(**hparams)
    t0    = time.time()

    if args.selection_method == "leave_one_domain_out":
        val_accs       = []
        fold_test_accs = []
        for fold in data["folds"]:
            results = train_one_restart(flags, fold["train"], device, verbose=False)
            with torch.no_grad():
                logits  = results["model"](fold["val"]["images"])
                val_acc = mean_accuracy(logits, fold["val"]["labels"]).detach().cpu().item()
                logits  = results["model"](data["test"]["images"])
                te_acc  = mean_accuracy(logits, data["test"]["labels"]).detach().cpu().item()
            val_accs.append(val_acc)
            fold_test_accs.append(te_acc)
        score    = score_leave_one_domain_out(val_accs)
        test_acc = float(np.mean(fold_test_accs))

    else:
        results = train_one_restart(flags, data["train"], device, verbose=False)
        with torch.no_grad():
            logits   = results["model"](data["test"]["images"])
            test_acc = mean_accuracy(logits, data["test"]["labels"]).detach().cpu().item()
        train_accs = results["train_accs"]
        val_accs   = []
        if args.selection_method == "train_domain_val":
            with torch.no_grad():
                for val_env in data["val"]:
                    logits  = results["model"](val_env["images"])
                    val_acc = mean_accuracy(logits, val_env["labels"]).detach().cpu().item()
                    val_accs.append(val_acc)
            score = score_train_domain_val(train_accs, val_accs, test_acc)
        else:
            score = score_oracle(train_accs, val_accs, test_acc)

    elapsed = time.time() - t0

    print(f"Trial {trial:02d} | score: {score:.4f} | "
          f"val: {[f'{v:.4f}' for v in val_accs]} | "
          f"test: {test_acc:.4f} | "
          f"elapsed: {elapsed:.1f}s")

    all_trials.append({
        "trial":    trial,
        "score":    score,
        "val_accs": val_accs,
        "test_acc": test_acc,
        "elapsed":  elapsed,
        **hparams,
    })

    if score > best_score:
        best_score   = score
        best_hparams = hparams

print(f"\nBest score: {best_score:.4f}")
print(f"Best hparams: {best_hparams}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs("results/exp3", exist_ok=True)
out_path = f"results/exp3/search_results_{args.mode}_{args.n_envs}_{args.selection_method}.json"
with open(out_path, "w") as f:
    json.dump({"best": best_hparams, "all_trials": all_trials}, f, indent=2)
print(f"Saved -> {out_path}")