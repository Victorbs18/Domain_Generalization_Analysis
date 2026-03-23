# main_prox_div.py
# Disentangles the effect of environment DIVERSITY vs PROXIMITY to the
# test distribution on IRM's ability to generalize under realistic
# model selection.
#
# 2x2 design — always n=2 training environments:
#
#   Config A: {0.1, 0.2} — Low diversity,  Low proximity
#   Config B: {0.7, 0.8} — Low diversity,  High proximity
#   Config C: {0.1, 0.5} — High diversity, Low proximity
#   Config D: {0.1, 0.8} — High diversity, High proximity
#
# Test environment always fixed at e=0.9.
# Three model selection methods supported:
#   oracle, train_domain_val, leave_one_domain_out
#
# If diversity is the key factor:
#   C and D should recover, A and B should fail
# If proximity is the key factor:
#   B and D should recover, A and C should fail
# If both matter:
#   Only D fully recovers

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import optim

from main import make_environment, MLP, mean_nll, mean_accuracy, irm_penalty


# ---------------------------------------------------------------------------
# Environment configurations — 2x2 design
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "A": {"e_values": [0.1, 0.2], "diversity": "low",  "proximity": "low"},
    "B": {"e_values": [0.7, 0.8], "diversity": "low",  "proximity": "high"},
    "C": {"e_values": [0.1, 0.5], "diversity": "high", "proximity": "low"},
    "D": {"e_values": [0.1, 0.8], "diversity": "high", "proximity": "high"},
}

TEST_E = 0.9


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Colored MNIST — Proximity vs Diversity 2x2 IRM")
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr",                    type=float, default=0.001)
    parser.add_argument("--n_restarts",            type=int,   default=10)
    parser.add_argument("--penalty_anneal_iters",  type=int,   default=100)
    parser.add_argument("--penalty_weight",        type=float, default=10000.0)
    parser.add_argument("--steps",                 type=int,   default=501)
    parser.add_argument("--grayscale_model",       action="store_true")
    parser.add_argument("--seed",                  type=int,   default=0)
    parser.add_argument("--config",                type=str,   default="A",
                        choices=["A", "B", "C", "D"],
                        help=(
                            "Environment config: "
                            "A={0.1,0.2} low div/low prox, "
                            "B={0.7,0.8} low div/high prox, "
                            "C={0.1,0.5} high div/low prox, "
                            "D={0.1,0.8} high div/high prox"
                        ))
    parser.add_argument("--selection_method",      type=str,   default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_environments(config, selection_method, device, val_fraction=0.2):
    """
    Load Colored MNIST with 2 training environments defined by config.

    Returns a dict with keys:
        'train': list of 2 training environments
        'val':   list of 2 validation environments (for selection)
        'test':  single test environment (never used for selection)
    """
    cfg      = ENV_CONFIGS[config]
    e_values = cfg["e_values"]

    mnist      = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_pool = (mnist.data[:50000], mnist.targets[:50000])
    test_pool  = (mnist.data[50000:], mnist.targets[50000:])

    # Shuffle train pool
    rng_state = np.random.get_state()
    np.random.shuffle(train_pool[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_pool[1].numpy())

    # Split into 2 equal halves
    n_total    = len(train_pool[0])
    split_size = n_total // 2

    train_envs = []
    val_envs   = []

    for i, e in enumerate(e_values):
        start = i * split_size
        end   = start + split_size if i < 1 else n_total

        env_images = train_pool[0][start:end]
        env_labels = train_pool[1][start:end]

        if selection_method == "train_domain_val":
            n_val   = int(len(env_images) * val_fraction)
            n_train = len(env_images) - n_val
            train_envs.append(make_environment(
                env_images[:n_train], env_labels[:n_train], e, device))
            val_envs.append(make_environment(
                env_images[n_train:], env_labels[n_train:], e, device))
        else:
            train_envs.append(make_environment(
                env_images, env_labels, e, device))
            val_envs.append(make_environment(
                env_images, env_labels, e, device))

    test_env = make_environment(
        test_pool[0], test_pool[1], TEST_E, device)

    return {
        "train": train_envs,
        "val":   val_envs,
        "test":  test_env,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_restart(flags, data, device, verbose=True):
    """
    Train on data['train'], evaluate on data['val'] and data['test'].
    Returns dict with all accuracies.
    """
    model      = MLP(flags.hidden_dim, flags.grayscale_model).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=flags.lr)
    train_envs = data["train"]

    if verbose:
        cols = ["step", "train_nll", "train_acc", "train_penalty", "test_acc"]
        print("   ".join(c.ljust(13) for c in cols))

    for step in range(flags.steps):
        for env in train_envs:
            logits         = model(env["images"])
            env["nll"]     = mean_nll(logits, env["labels"])
            env["acc"]     = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll     = torch.stack([e["nll"]     for e in train_envs]).mean()
        train_acc     = torch.stack([e["acc"]     for e in train_envs]).mean()
        train_penalty = torch.stack([e["penalty"] for e in train_envs]).mean()

        weight_norm = torch.tensor(0.0, device=device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        penalty_weight = (
            flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
        )
        loss  = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        loss += penalty_weight * train_penalty
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
                fmt(train_nll.detach().cpu().numpy()),
                fmt(train_acc.detach().cpu().numpy()),
                fmt(train_penalty.detach().cpu().numpy()),
                fmt(test_acc.detach().cpu().numpy()),
            ]))

    # Final evaluation
    with torch.no_grad():
        for env in train_envs + data["val"]:
            logits     = model(env["images"])
            env["acc"] = mean_accuracy(logits, env["labels"])
        test_logits         = model(data["test"]["images"])
        data["test"]["acc"] = mean_accuracy(
            test_logits, data["test"]["labels"])

    train_accs = [e["acc"].detach().cpu().item() for e in train_envs]
    val_accs   = [e["acc"].detach().cpu().item() for e in data["val"]]
    test_acc   = data["test"]["acc"].detach().cpu().item()

    return {
        "train_accs": train_accs,
        "val_accs":   val_accs,
        "test_acc":   test_acc,
        "mean_train": float(np.mean(train_accs)),
        "mean_val":   float(np.mean(val_accs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = get_args()

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ENV_CONFIGS[flags.config]
    print(f"Using device: {device}")
    print(f"Config: {flags.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Diversity: {cfg['diversity']}  |  "
          f"Proximity: {cfg['proximity']}")
    print(f"Selection method: {flags.selection_method}\n")

    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    data = load_environments(flags.config, flags.selection_method, device)

    final_test_accs = []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        results = train_one_restart(flags, data, device, verbose=True)
        final_test_accs.append(results["test_acc"])
        print(f"  Train accs: {[f'{a:.4f}' for a in results['train_accs']]}")
        print(f"  Val   accs: {[f'{a:.4f}' for a in results['val_accs']]}")
        print(f"  Test  acc:  {results['test_acc']:.4f}")
        print(f"  Test so far: "
              f"{np.mean(final_test_accs):.4f} ± "
              f"{np.std(final_test_accs):.4f}")

    print("\n=== Final results ===")
    print(f"Config: {flags.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Diversity: {cfg['diversity']}  |  "
          f"Proximity: {cfg['proximity']}")
    print(f"Selection: {flags.selection_method}")
    print(f"Test acc (mean ± std): "
          f"{np.mean(final_test_accs):.4f} ± "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()