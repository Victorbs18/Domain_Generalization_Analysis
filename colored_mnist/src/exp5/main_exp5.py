# src/exp5/main_exp5.py
# Experiment 5 -- Proximity vs Diversity 2x2
#
# Usage:
#   python src/exp5/main_exp5.py --mode irm --config A --selection_method oracle
#   python src/exp5/main_exp5.py --mode irm --config B --selection_method train_domain_val
#   python src/exp5/main_exp5.py --mode erm --config A --selection_method oracle
#
# Note: Config A oracle reuses Exp 1 results — same environments (e=0.1, e=0.2)
#       ERM uses fixed hparams — no search needed

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import json
import numpy as np
import torch
from types import SimpleNamespace

from utils.dataset   import load_mnist
from utils.selection import load_environments
from utils.trainer   import train_one_restart
from utils.losses    import mean_accuracy
from utils.search    import ERM_FIXED

# ---------------------------------------------------------------------------
# 2x2 configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "A": [0.1, 0.2],  # Low diversity, Low proximity
    "B": [0.7, 0.8],  # Low diversity, High proximity
    "C": [0.1, 0.5],  # High diversity, Low proximity
    "D": [0.1, 0.8],  # High diversity, High proximity
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment 5 -- Proximity vs Diversity 2x2")
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--config",           type=str, default="A",
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val", "leave_one_domain_out"])
    parser.add_argument("--n_restarts",       type=int, default=10)
    parser.add_argument("--seed",             type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # Load best hyperparameters from search results
    # ERM uses fixed hparams — no search needed
    # Config A oracle reuses Exp 1 results — same environments
    if args.mode == "erm":
        hparams = ERM_FIXED.copy()
        hparams["grayscale_model"] = False
    elif args.config == "A" and args.selection_method == "oracle":
        with open("results/exp1/search_results_irm.json") as f:
            hparams = json.load(f)["best"]
    else:
        search_path = (f"results/exp5/search_results_irm_"
                       f"{args.config}_{args.selection_method}.json")
        with open(search_path) as f:
            hparams = json.load(f)["best"]

    hparams["n_restarts"] = args.n_restarts
    hparams["seed"]       = args.seed
    flags = SimpleNamespace(**hparams)

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Mode: {args.mode}  |  Config: {args.config}  |  "
          f"e values: {CONFIGS[args.config]}  |  "
          f"Selection: {args.selection_method}")
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    # Final model always trains on ALL training environments
    e_values = CONFIGS[args.config]
    train_pool, test_pool = load_mnist()
    data = load_environments(
        selection_method = "oracle",
        train_pool       = train_pool,
        test_pool        = test_pool,
        e_values         = e_values,
        device           = device,
    )

    final_train_accs = []
    final_test_accs  = []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        results = train_one_restart(flags, data["train"], device, verbose=True)

        with torch.no_grad():
            logits   = results["model"](data["test"]["images"])
            test_acc = mean_accuracy(logits, data["test"]["labels"]).detach().cpu().item()

        final_train_accs.append(results["mean_train"])
        final_test_accs.append(test_acc)

        print(f"  Train acc so far: "
              f"{np.mean(final_train_accs):.4f} +- "
              f"{np.std(final_train_accs):.4f}")
        print(f"  Test  acc so far: "
              f"{np.mean(final_test_accs):.4f} +- "
              f"{np.std(final_test_accs):.4f}")

    print("\n=== Final results ===")
    print(f"Mode: {args.mode}  |  Config: {args.config}  |  "
          f"e values: {CONFIGS[args.config]}  |  "
          f"Selection: {args.selection_method}")
    print(f"Train acc (mean +- std): "
          f"{np.mean(final_train_accs):.4f} +- "
          f"{np.std(final_train_accs):.4f}")
    print(f"Test  acc (mean +- std): "
          f"{np.mean(final_test_accs):.4f} +- "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()