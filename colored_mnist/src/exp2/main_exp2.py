# src/exp2/main_exp2.py
# Experiment 2 — Model Selection Methods
#
# Tests whether IRM's advantage holds under realistic model selection.
# Three selection strategies: oracle, train_domain_val, leave_one_domain_out
#
# Usage:
#   python src/exp2/main_exp2.py --mode irm --selection_method oracle
#   python src/exp2/main_exp2.py --mode irm --selection_method train_domain_val
#   python src/exp2/main_exp2.py --mode irm --selection_method leave_one_domain_out
#   python src/exp2/main_exp2.py --mode erm --selection_method train_domain_val
#   python src/exp2/main_exp2.py --mode erm --selection_method leave_one_domain_out
#   python src/exp2/main_exp2.py --mode grayscale --selection_method train_domain_val
#   python src/exp2/main_exp2.py --mode grayscale --selection_method leave_one_domain_out
#
# Note: oracle hparams reuse Exp 1 results:
#   results/exp1/search_results_{mode}.json

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment 2 — Model Selection Methods")
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm", "grayscale"])
    parser.add_argument("--selection_method", type=str, default="train_domain_val",
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
    if args.selection_method == "oracle":
        search_path = f"results/exp1/search_results_{args.mode}.json"
    else:
        search_path = f"results/exp2/search_results_{args.mode}_{args.selection_method}.json"

    with open(search_path) as f:
        hparams = json.load(f)["best"]

    hparams["n_restarts"] = args.n_restarts
    hparams["seed"]       = args.seed
    flags = SimpleNamespace(**hparams)

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Mode: {args.mode}  |  Selection: {args.selection_method}")
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    # Final model always trains on ALL training environments
    # regardless of selection method used during search
    E_VALUES = [0.2, 0.1]
    train_pool, test_pool = load_mnist()
    data = load_environments(
        selection_method = "oracle",
        train_pool       = train_pool,
        test_pool        = test_pool,
        e_values         = E_VALUES,
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
    print(f"Mode: {args.mode}  |  Selection: {args.selection_method}")
    print(f"Train acc (mean +- std): "
          f"{np.mean(final_train_accs):.4f} +- "
          f"{np.std(final_train_accs):.4f}")
    print(f"Test  acc (mean +- std): "
          f"{np.mean(final_test_accs):.4f} +- "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()