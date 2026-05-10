# main_dist_sweep.py
# Experiment 7 — Performance vs Distribution Distance
#
# Trains IRM and ERM on a single training environment with a swept e value
# (e = 0.1, 0.2, ..., 0.9) and evaluates on the fixed test environment
# (e = 0.9). Results are saved to JSON for downstream distance plotting.
#
# The key question: how does the distributional distance between the
# training environment and the test environment relate to IRM vs ERM
# test accuracy? Distance is quantified by MMD, Wasserstein, and PAD
# (computed separately by compute_distances.py).
#
# Usage
# -----
#   python main_dist_sweep.py --e_train 0.2 --mode irm --selection_method oracle
#
#   # Run all combinations
#   run_dist_sweep_experiments.bat

import argparse
import json
import numpy as np
import torch
from torchvision import datasets
from torch import optim

from main import make_environment, MLP, mean_nll, mean_accuracy, irm_penalty


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_E      = 0.9
E_VALUES    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VAL_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Colored MNIST — Distribution Distance Sweep (Experiment 7)")
    parser.add_argument("--e_train",               type=float, default=0.2,
                        choices=E_VALUES,
                        help="Training environment e value (0.1 to 0.9)")
    parser.add_argument("--mode",                  type=str,   default="irm",
                        choices=["irm", "erm"],
                        help="Training method")
    parser.add_argument("--selection_method",      type=str,   default="oracle",
                        choices=["oracle", "train_domain_val"],
                        help="Model selection strategy")
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr",                    type=float, default=0.001)
    parser.add_argument("--n_restarts",            type=int,   default=10)
    parser.add_argument("--penalty_anneal_iters",  type=int,   default=100)
    parser.add_argument("--penalty_weight",        type=float, default=10000.0)
    parser.add_argument("--steps",                 type=int,   default=501)
    parser.add_argument("--seed",                  type=int,   default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_environments(e_train, selection_method, device):
    """
    Load Colored MNIST with a single training environment at e=e_train
    and a fixed test environment at e=0.9.

    Returns a dict with keys:
        'train': single training environment
        'val':   validation environment (for selection)
        'test':  fixed test environment (never used for selection)
    """
    mnist       = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_pool  = (mnist.data[:50000], mnist.targets[:50000])
    test_pool   = (mnist.data[50000:], mnist.targets[50000:])

    # Shuffle train pool
    rng_state = np.random.get_state()
    np.random.shuffle(train_pool[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_pool[1].numpy())

    if selection_method == "train_domain_val":
        n_total = len(train_pool[0])
        n_val   = int(n_total * VAL_FRACTION)
        n_train = n_total - n_val
        train_env = make_environment(
            train_pool[0][:n_train],
            train_pool[1][:n_train],
            e_train, device)
        val_env = make_environment(
            train_pool[0][n_train:],
            train_pool[1][n_train:],
            e_train, device)
    else:
        # Oracle — use full train pool; val is unused (test used directly)
        train_env = make_environment(
            train_pool[0], train_pool[1], e_train, device)
        val_env   = make_environment(
            train_pool[0], train_pool[1], e_train, device)

    test_env = make_environment(
        test_pool[0], test_pool[1], TEST_E, device)

    return {
        "train": train_env,
        "val":   val_env,
        "test":  test_env,
    }


# ---------------------------------------------------------------------------
# Training loop (one restart)
# ---------------------------------------------------------------------------

def train_one_restart(flags, data, device, verbose=True):
    """
    Train on data['train'], select on data['val'], evaluate on data['test'].
    Returns dict with all accuracies.
    """
    model     = MLP(flags.hidden_dim, grayscale=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)

    train_env = data["train"]

    if verbose:
        cols = ["step", "train_nll", "train_acc", "train_penalty", "test_acc"]
        print("   ".join(c.ljust(13) for c in cols))

    for step in range(flags.steps):
        logits              = model(train_env["images"])
        train_env["nll"]    = mean_nll(logits, train_env["labels"])
        train_env["acc"]    = mean_accuracy(logits, train_env["labels"])
        train_env["penalty"]= irm_penalty(logits, train_env["labels"])

        weight_norm = torch.tensor(0.0, device=device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        penalty_weight = (
            flags.penalty_weight
            if (flags.mode == "irm" and step >= flags.penalty_anneal_iters)
            else 1.0
        )

        loss  = train_env["nll"].clone()
        loss += flags.l2_regularizer_weight * weight_norm

        if flags.mode == "irm":
            loss += penalty_weight * train_env["penalty"]
            if penalty_weight > 1.0:
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            def fmt(v):
                return np.array2string(
                    v, precision=5, floatmode="fixed").ljust(13)
            with torch.no_grad():
                test_logits = model(data["test"]["images"])
                test_acc    = mean_accuracy(
                    test_logits, data["test"]["labels"])
            print("   ".join([
                str(np.int32(step)).ljust(13),
                fmt(train_env["nll"].detach().cpu().numpy()),
                fmt(train_env["acc"].detach().cpu().numpy()),
                fmt(train_env["penalty"].detach().cpu().numpy()),
                fmt(test_acc.detach().cpu().numpy()),
            ]))

    # Final evaluation
    with torch.no_grad():
        train_logits        = model(train_env["images"])
        train_env["acc"]    = mean_accuracy(
            train_logits, train_env["labels"])

        val_logits          = model(data["val"]["images"])
        data["val"]["acc"]  = mean_accuracy(
            val_logits, data["val"]["labels"])

        test_logits         = model(data["test"]["images"])
        data["test"]["acc"] = mean_accuracy(
            test_logits, data["test"]["labels"])

    return {
        "train_acc": train_env["acc"].detach().cpu().item(),
        "val_acc":   data["val"]["acc"].detach().cpu().item(),
        "test_acc":  data["test"]["acc"].detach().cpu().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = get_args()

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Mode: {flags.mode}  |  "
          f"e_train: {flags.e_train}  |  "
          f"Selection: {flags.selection_method}\n")
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    data = load_environments(flags.e_train, flags.selection_method, device)

    final_test_accs  = []
    final_train_accs = []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        results = train_one_restart(flags, data, device, verbose=True)
        final_test_accs.append(results["test_acc"])
        final_train_accs.append(results["train_acc"])
        print(f"  Train acc: {results['train_acc']:.4f}")
        print(f"  Val   acc: {results['val_acc']:.4f}")
        print(f"  Test  acc: {results['test_acc']:.4f}")
        print(f"  Test so far: "
              f"{np.mean(final_test_accs):.4f} ± "
              f"{np.std(final_test_accs):.4f}")

    mean_test = float(np.mean(final_test_accs))
    std_test  = float(np.std(final_test_accs))

    print("\n=== Final results ===")
    print(f"Mode: {flags.mode}  |  "
          f"e_train: {flags.e_train}  |  "
          f"Selection: {flags.selection_method}")
    print(f"Test acc (mean ± std): {mean_test:.4f} ± {std_test:.4f}")

    # Save results to JSON for downstream plotting
    out = {
        "mode":             flags.mode,
        "e_train":          flags.e_train,
        "selection_method": flags.selection_method,
        "mean_test_acc":    mean_test,
        "std_test_acc":     std_test,
        "all_test_accs":    final_test_accs,
        "hidden_dim":       flags.hidden_dim,
        "lr":               flags.lr,
        "l2_regularizer_weight": flags.l2_regularizer_weight,
        "penalty_anneal_iters":  flags.penalty_anneal_iters,
        "penalty_weight":        flags.penalty_weight,
        "steps":                 flags.steps,
    }
    out_path = (f"results_dist_sweep"
                f"_{flags.mode}"
                f"_e{flags.e_train:.1f}"
                f"_{flags.selection_method}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()