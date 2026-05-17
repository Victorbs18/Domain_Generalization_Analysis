# main_exp8.py — Experiment 8: Two Independent Spurious Features
#
# Same design as Exp 7 but color and texture are assigned independently.
# Direct comparison with Exp 7 isolates the effect of feature independence.
#
# 4 configs matching Exp 5, 7:
#   A: {0.1, 0.2} — low proximity,  low diversity
#   B: {0.7, 0.8} — high proximity, low diversity
#   C: {0.1, 0.5} — low proximity,  high diversity
#   D: {0.1, 0.8} — high proximity, high diversity
#
# Usage
#   python src/exp8/main_exp8.py --config A --mode irm --selection_method oracle --n_restarts 10
#
# Output
#   results/exp8/results_{mode}_{config}_{selection_method}.json

import argparse
import json
import os
import sys
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset_texture_independent import make_environment_texture_independent
from utils.dataset                      import load_mnist
from utils.selection                    import load_environments
from utils.trainer                      import train_one_restart
from utils.search                       import ERM_FIXED
from utils.losses                       import mean_accuracy


TEST_E = 0.9

CONFIGS = {
    "A": {"e_values": [0.1, 0.2], "desc": "low proximity, low diversity"},
    "B": {"e_values": [0.7, 0.8], "desc": "high proximity, low diversity"},
    "C": {"e_values": [0.1, 0.5], "desc": "low proximity, high diversity"},
    "D": {"e_values": [0.1, 0.8], "desc": "high proximity, high diversity"},
}


def load_hparams(mode, config, selection_method):
    path = (f"results/exp8/search_results"
            f"_{mode}_{config}_{selection_method}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["best"]



def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = CONFIGS[args.config]

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}  |  Config: {args.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Selection: {args.selection_method}")

    hparams = load_hparams(args.mode, args.config, args.selection_method)
    if hparams is None:
        print("No search JSON found — using defaults")
        hparams = ERM_FIXED.copy() if args.mode == "erm" else {
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

    flags = SimpleNamespace(**hparams)

    train_pool, test_pool = load_mnist()
    data = load_environments(
        selection_method=args.selection_method,
        train_pool=train_pool,
        test_pool=test_pool,
        e_values=cfg["e_values"],
        device=device,
        make_env_fn=make_environment_texture_independent,
        test_e=TEST_E,
    )

    final_test_accs  = []
    final_train_accs = []

    for restart in range(args.n_restarts):
        print(f"\n--- Restart {restart} ---")
        torch.manual_seed(args.seed + restart)
        np.random.seed(args.seed + restart)

        results = train_one_restart(
            flags, data["train"], device, verbose=True)

        model = results["model"]
        model.eval()
        with torch.no_grad():
            test_logits = model(data["test"]["images"])
            test_acc    = mean_accuracy(
                test_logits, data["test"]["labels"]).item()

        final_test_accs.append(test_acc)
        final_train_accs.append(results["mean_train"])

        print(f"  Train acc so far: "
              f"{np.mean(final_train_accs):.4f} ± "
              f"{np.std(final_train_accs):.4f}")
        print(f"  Test  acc so far: "
              f"{np.mean(final_test_accs):.4f} ± "
              f"{np.std(final_test_accs):.4f}")

    mean_test  = float(np.mean(final_test_accs))
    std_test   = float(np.std(final_test_accs))
    mean_train = float(np.mean(final_train_accs))
    std_train  = float(np.std(final_train_accs))

    print(f"\n=== Final results ===")
    print(f"Mode: {args.mode}  |  Config: {args.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Selection: {args.selection_method}")
    print(f"Train acc (mean ± std): {mean_train:.4f} ± {std_train:.4f}")
    print(f"Test  acc (mean ± std): {mean_test:.4f} ± {std_test:.4f}")

    os.makedirs("results/exp8", exist_ok=True)
    out = {
        "mode":             args.mode,
        "config":           args.config,
        "e_values":         cfg["e_values"],
        "selection_method": args.selection_method,
        "mean_test_acc":    mean_test,
        "std_test_acc":     std_test,
        "mean_train_acc":   mean_train,
        "std_train_acc":    std_train,
        "all_test_accs":    final_test_accs,
        "all_train_accs":   final_train_accs,
    }
    out_path = (f"results/exp8/results"
                f"_{args.mode}_{args.config}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")



def get_args():
    parser = argparse.ArgumentParser(
        description="Exp 8 — Two Independent Spurious Features")
    parser.add_argument("--config",           type=str, default="A",
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_restarts",       type=int, default=10)
    parser.add_argument("--seed",             type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
