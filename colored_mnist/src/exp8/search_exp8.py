# search_exp8.py — Hyperparameter search for Experiment 8.
#
# Same design as Exp 7 but with TWO INDEPENDENT spurious features.
# Color and texture are assigned with separate independent Bernoulli flips.
#
# 4 configs matching Exp 5 and 7:
#   A: {0.1, 0.2} — low proximity,  low diversity
#   B: {0.7, 0.8} — high proximity, low diversity
#   C: {0.1, 0.5} — low proximity,  high diversity
#   D: {0.1, 0.8} — high proximity, high diversity
#
# Usage
#   python src/exp8/search_exp8.py --config A --mode irm --selection_method oracle --n_trials 50
#
# Output
#   results/exp8/search_results_{mode}_{config}_{selection_method}.json

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset_texture_independent import make_environment_texture_independent
from utils.dataset                      import load_mnist
from utils.selection                    import (load_environments,
                                                score_oracle,
                                                score_train_domain_val,
                                                score_leave_one_domain_out)
from utils.trainer                      import train_one_restart
from utils.search                       import sample_irm, ERM_FIXED
from utils.losses                       import mean_accuracy

TEST_E = 0.9

CONFIGS = {
    "A": {"e_values": [0.1, 0.2], "desc": "low proximity, low diversity"},
    "B": {"e_values": [0.7, 0.8], "desc": "high proximity, low diversity"},
    "C": {"e_values": [0.1, 0.5], "desc": "low proximity, high diversity"},
    "D": {"e_values": [0.1, 0.8], "desc": "high proximity, high diversity"},
}


def evaluate(hparams, data, device, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    flags   = SimpleNamespace(**hparams)
    results = train_one_restart(flags, data["train"], device, verbose=False)

    model = results["model"]
    model.eval()

    with torch.no_grad():
        test_logits = model(data["test"]["images"])
        test_acc    = mean_accuracy(
            test_logits, data["test"]["labels"]).item()

    if data["folds"] is not None:
        fold_val_accs = []
        for fold in data["folds"]:
            torch.manual_seed(seed)
            fold_res   = train_one_restart(
                flags, fold["train"], device, verbose=False)
            fold_model = fold_res["model"]
            fold_model.eval()
            with torch.no_grad():
                fv = fold_model(fold["val"]["images"])
                fold_val_accs.append(
                    mean_accuracy(fv, fold["val"]["labels"]).item())
        val_accs = fold_val_accs
    else:
        val_envs = data["val"] if isinstance(data["val"], list) \
            else [data["val"]]
        model.eval()
        val_accs = []
        with torch.no_grad():
            for ve in val_envs:
                vl = model(ve["images"])
                val_accs.append(
                    mean_accuracy(vl, ve["labels"]).item())

    return results["train_accs"], val_accs, test_acc

def run_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = CONFIGS[args.config]

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}  |  Config: {args.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Selection: {args.selection_method}  |  "
          f"Trials: {args.n_trials}")

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

        t0 = time.time()
        train_accs, val_accs, test_acc = evaluate(
            hparams, data, device, seed=args.seed + trial_idx)
        elapsed = time.time() - t0

        if args.selection_method == "leave_one_domain_out":
            score = score_leave_one_domain_out(val_accs)
        elif args.selection_method == "train_domain_val":
            score = score_train_domain_val(train_accs, val_accs, test_acc)
        else:
            score = score_oracle(train_accs, val_accs, test_acc)

        val_str = str([f"{v:.4f}" for v in val_accs])
        print(f"Trial {trial_idx:02d} | score: {score:.4f} | "
              f"val: {val_str} | test: {test_acc:.4f} | "
              f"elapsed: {elapsed:.1f}s")

        if score > best_score:
            best_score   = score
            best_hparams = hparams.copy()

    print(f"\nBest score: {best_score:.4f}")
    print(f"Best hparams: {best_hparams}")

    os.makedirs("results/exp8", exist_ok=True)
    out_path = (f"results/exp8/search_results"
                f"_{args.mode}_{args.config}"
                f"_{args.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump({"best": best_hparams}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search — Exp 8 Two Independent Spurious Features")
    parser.add_argument("--config",           type=str, default="A",
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--mode",             type=str, default="irm",
                        choices=["irm", "erm"])
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--n_trials",         type=int, default=50)
    parser.add_argument("--seed",             type=int, default=0)
    args = parser.parse_args()
    run_search(args)
