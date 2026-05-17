# main_exp6.py — Experiment 6: Performance vs Distribution Distance
#
# Trains IRM and ERM on two environments {e1=0.1, e2} where e2 is swept
# from 0.2 to 0.9. Evaluates on fixed test environment (e=0.9).
# Hyperparameters loaded from search JSON if available.
#
# Usage
#   python src/exp6/main_exp6.py --mode irm --e2 0.5
#       --selection_method oracle --n_restarts 10
#
# Output
#   results/exp6/results_{mode}_e{e2}_{selection_method}.json

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset   import load_mnist, make_environment
from utils.trainer   import train_one_restart
from utils.selection import load_environments
from utils.search    import ERM_FIXED
from utils.losses    import mean_accuracy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E1       = 0.1
E2_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEST_E   = 0.9


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Exp 6 — Performance vs Distribution Distance")
    parser.add_argument("--mode",             type=str,   default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--e2",               type=float, default=0.2,
                        choices=E2_SWEEP)
    parser.add_argument("--selection_method", type=str,   default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_restarts",       type=int,   default=10)
    parser.add_argument("--seed",             type=int,   default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load hyperparameters
# ---------------------------------------------------------------------------

def load_hparams(mode, e2, selection_method):
    path = (f"results/exp6/search_results"
            f"_{mode}_e{e2:.1f}_{selection_method}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["best"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}  |  e1={E1}  e2={args.e2}  |  "
          f"Selection: {args.selection_method}")

    # Load hparams
    hparams = load_hparams(args.mode, args.e2, args.selection_method)
    if hparams is None:
        print("No search JSON found — using defaults")
        if args.mode == "erm":
            hparams = ERM_FIXED.copy()
        else:
            hparams = {
                "hidden_dim":            256,
                "l2_regularizer_weight": 0.001,
                "lr":                    0.001,
                "penalty_anneal_iters":  100,
                "penalty_weight":        10000.0,
                "steps":                 501,
            }
    hparams["mode"]            = args.mode
    hparams["grayscale_model"] = False

    print("Flags:")
    for k, v in sorted(hparams.items()):
        print(f"\t{k}: {v}")

    # Load data
    train_pool, test_pool = load_mnist()
    e_values = [E1, args.e2]

    data = load_environments(
        selection_method=args.selection_method,
        train_pool=train_pool,
        test_pool=test_pool,
        e_values=e_values,
        device=device,
    )
    # Always evaluate on fixed test e=0.9
    test_env = make_environment(
        test_pool[0], test_pool[1], TEST_E, device)

    from types import SimpleNamespace
    flags = SimpleNamespace(**hparams)

    final_test_accs  = []
    final_train_accs = []

    for restart in range(args.n_restarts):
        print(f"\n--- Restart {restart} ---")
        torch.manual_seed(args.seed + restart)
        np.random.seed(args.seed + restart)

        results = train_one_restart(
            flags, data["train"], device, verbose=True)

        # Evaluate on test env
        model = results["model"]
        model.eval()
        with torch.no_grad():
            test_logits = model(test_env["images"])
            test_acc    = mean_accuracy(
                test_logits, test_env["labels"]).item()

        final_test_accs.append(test_acc)
        final_train_accs.append(results["mean_train"])

        print(f"  Train acc so far: "
              f"{np.mean(final_train_accs):.4f} ± "
              f"{np.std(final_train_accs):.4f}")
        print(f"  Test  acc so far: "
              f"{np.mean(final_test_accs):.4f} ± "
              f"{np.std(final_test_accs):.4f}")

    mean_test = float(np.mean(final_test_accs))
    std_test  = float(np.std(final_test_accs))
    mean_train = float(np.mean(final_train_accs))
    std_train  = float(np.std(final_train_accs))

    print(f"\n=== Final results ===")
    print(f"Mode: {args.mode}  |  e1={E1}  e2={args.e2}  |  "
          f"Selection: {args.selection_method}")
    print(f"Train acc (mean ± std): {mean_train:.4f} ± {std_train:.4f}")
    print(f"Test  acc (mean ± std): {mean_test:.4f} ± {std_test:.4f}")

    os.makedirs("results/exp6", exist_ok=True)
    out = {
        "mode":             args.mode,
        "e1":               E1,
        "e2":               args.e2,
        "selection_method": args.selection_method,
        "mean_test_acc":    mean_test,
        "std_test_acc":     std_test,
        "mean_train_acc":   mean_train,
        "std_train_acc":    std_train,
        "all_test_accs":    final_test_accs,
        "all_train_accs":   final_train_accs,
    }
    out_path = (f"results/exp6/results"
                f"_{args.mode}_e{args.e2:.1f}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
