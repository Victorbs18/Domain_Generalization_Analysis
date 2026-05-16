# src/exp1/main_exp1.py
# Experiment 1 : IRM Reproduction
#
# Reproduces the three conditions from the original IRM paper under oracle model selection:
#   - ERM baseline
#   - IRM 
#   - Grayscale ERM 
#
# Arjovsky et al. (2019). Invariant Risk Minimization.
# https://arxiv.org/abs/1907.02893

import argparse
import numpy as np
import torch
import json
from types import SimpleNamespace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset  import load_mnist, make_environment
from utils.losses   import mean_accuracy
from utils.trainer  import train_one_restart


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1 — IRM Reproduction")
    parser.add_argument("--mode",       type=str, default="irm",
                        choices=["irm", "erm", "grayscale"])
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--seed",       type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_colored_mnist(device):
    """
    Load standard Colored MNIST with 2 training environments
    and 1 fixed test environment.

    Returns list of 3 environments:
        envs[0]: train split A, e=0.2 — color 80% correlated with label
        envs[1]: train split B, e=0.1 — color 90% correlated with label
        envs[2]: test split,    e=0.9 — color 90% anti-correlated
    """
    train_pool, test_pool = load_mnist()

    return [
        make_environment(train_pool[0][::2],  train_pool[1][::2],  0.2, device),
        make_environment(train_pool[0][1::2], train_pool[1][1::2], 0.1, device),
        make_environment(test_pool[0],        test_pool[1],        0.9, device),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Load best hyperparameters from search results
    search_path = f"results/exp1/search_results_{args.mode}.json"
    with open(search_path) as f:
        hparams = json.load(f)["best"]

    # Add runtime arguments
    hparams["n_restarts"] = args.n_restarts
    hparams["seed"]       = args.seed

    flags = SimpleNamespace(**hparams)

    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    envs = load_colored_mnist(device)
    train_envs = envs[:2]
    test_env   = envs[2]

    final_train_accs = []
    final_test_accs  = []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        results = train_one_restart(flags, train_envs, device, verbose=True)

        with torch.no_grad():
            logits   = results["model"](test_env["images"])
            test_acc = mean_accuracy(logits, test_env["labels"]).detach().cpu().item()

        final_train_accs.append(results["mean_train"])
        final_test_accs.append(test_acc)

        print(f"  Train acc so far: "
              f"{np.mean(final_train_accs):.4f} +- "
              f"{np.std(final_train_accs):.4f}")
        print(f"  Test  acc so far: "
              f"{np.mean(final_test_accs):.4f} +- "
              f"{np.std(final_test_accs):.4f}")

    print("\n=== Final results ===")
    print(f"Mode: {args.mode}")
    print(f"Train acc (mean +- std): "
          f"{np.mean(final_train_accs):.4f} +- "
          f"{np.std(final_train_accs):.4f}")
    print(f"Test  acc (mean +- std): "
          f"{np.mean(final_test_accs):.4f} +- "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()