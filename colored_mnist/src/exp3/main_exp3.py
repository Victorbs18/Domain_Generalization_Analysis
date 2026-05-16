# src/exp3/main_exp3.py
# Experiment 3 — Number of Environments (Increasing Diversity)
#
# Tests whether more training environments help IRM recover under
# realistic model selection. Environments use increasing e values
# so both quantity and diversity increase together.
#
# n_envs | e values
# -------|--------------------------------------------------
#   1    | {0.1}
#   2    | {0.1, 0.2}
#   3    | {0.1, 0.2, 0.3}
#   4    | {0.1, 0.2, 0.3, 0.4}
#   5    | {0.1, 0.2, 0.3, 0.4, 0.45}
#
# Usage:
#   python src/exp3/main_exp3.py --mode irm --n_envs 3 --selection_method oracle
#   python src/exp3/main_exp3.py --mode irm --n_envs 3 --selection_method train_domain_val
#   python src/exp3/main_exp3.py --mode irm --n_envs 3 --selection_method leave_one_domain_out
#   python src/exp3/main_exp3.py --mode erm --n_envs 3 --selection_method oracle

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
def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3 — Number of Environments (Increasing Diversity)")
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--n_envs",           type=int, default=2,
                        choices=[1, 2, 3, 4, 5])
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
    # Oracle for n_envs=2 reuses Exp 1 results
    if args.mode == "erm":
        hparams = ERM_FIXED.copy()
        hparams["grayscale_model"] = False
    elif args.selection_method == "oracle" and args.n_envs == 2:
        with open("results/exp1/search_results_irm.json") as f:
            hparams = json.load(f)["best"]
    else:
        search_path = (f"results/exp3/search_results_{args.mode}_"
                       f"{args.n_envs}_{args.selection_method}.json")
        with open(search_path) as f:
            hparams = json.load(f)["best"]

    hparams["n_restarts"] = args.n_restarts
    hparams["seed"]       = args.seed
    flags = SimpleNamespace(**hparams)

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Mode: {args.mode}  |  n_envs: {args.n_envs}  |  "
          f"Selection: {args.selection_method}")
    print(f"e values: {ENV_E_VALUES[args.n_envs]}")
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    # Final model always trains on ALL training environments
    e_values = ENV_E_VALUES[args.n_envs]
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
    print(f"Mode: {args.mode}  |  n_envs: {args.n_envs}  |  "
          f"Selection: {args.selection_method}")
    print(f"e values: {ENV_E_VALUES[args.n_envs]}")
    print(f"Train acc (mean +- std): "
          f"{np.mean(final_train_accs):.4f} +- "
          f"{np.std(final_train_accs):.4f}")
    print(f"Test  acc (mean +- std): "
          f"{np.mean(final_test_accs):.4f} +- "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()