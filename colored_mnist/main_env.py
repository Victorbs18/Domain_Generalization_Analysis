# main_env.py
# Extends the IRM Colored MNIST experiment to support variable numbers
# of training environments with increasing color-flip diversity.
#
# Environment e values (fixed anchor + increasing diversity):
#   n=1: {0.1}
#   n=2: {0.1, 0.2}
#   n=3: {0.1, 0.2, 0.3}
#   n=4: {0.1, 0.2, 0.3, 0.4}
#   n=5: {0.1, 0.2, 0.3, 0.4, 0.45}
#
# Test environment is always fixed at e=0.9.
# Three model selection methods supported:
#   oracle, train_domain_val, leave_one_domain_out

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

# Reuse helpers from original main.py
from main import make_environment, MLP, mean_nll, mean_accuracy, irm_penalty


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

ENV_E_VALUES = {
    1: [0.1],
    2: [0.1, 0.2],
    3: [0.1, 0.2, 0.3],
    4: [0.1, 0.2, 0.3, 0.4],
    5: [0.1, 0.2, 0.3, 0.4, 0.45],
}

TEST_E = 0.9


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Colored MNIST — Variable Environments IRM")
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr",                    type=float, default=0.001)
    parser.add_argument("--n_restarts",            type=int,   default=10)
    parser.add_argument("--penalty_anneal_iters",  type=int,   default=100)
    parser.add_argument("--penalty_weight",        type=float, default=10000.0)
    parser.add_argument("--steps",                 type=int,   default=501)
    parser.add_argument("--grayscale_model",       action="store_true")
    parser.add_argument("--seed",                  type=int,   default=0)
    parser.add_argument("--n_envs",                type=int,   default=2,
                        choices=[1, 2, 3, 4, 5],
                        help="Number of training environments")
    parser.add_argument("--selection_method",      type=str,   default="oracle",
                        choices=["oracle", "train_domain_val", "leave_one_domain_out"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_environments(n_envs, selection_method, device, val_fraction=0.2):
    """
    Load Colored MNIST with n_envs training environments.

    Returns a dict with keys:
        'train': list of training environments
        'val':   list of validation environments (for selection)
        'test':  single test environment (never used for selection)

    Selection method determines how val environments are constructed:
        oracle:               val = test env (e=0.9)
        train_domain_val:     val = held-out 20% of each train env
        leave_one_domain_out: val = full copy of each train env (left out)
    """
    e_values = ENV_E_VALUES[n_envs]

    mnist      = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_pool = (mnist.data[:50000], mnist.targets[:50000])
    test_pool  = (mnist.data[50000:], mnist.targets[50000:])

    # Shuffle train pool
    rng_state = np.random.get_state()
    np.random.shuffle(train_pool[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_pool[1].numpy())

    # Divide train pool into n_envs equal splits
    n_total    = len(train_pool[0])
    split_size = n_total // n_envs

    train_envs = []
    val_envs   = []

    for i, e in enumerate(e_values):
        start = i * split_size
        end   = start + split_size if i < n_envs - 1 else n_total

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
            # oracle or leave_one_domain_out — full split used for training
            train_envs.append(make_environment(env_images, env_labels, e, device))
            val_envs.append(make_environment(env_images, env_labels, e, device))

    # Test environment — always fixed at e=0.9, never used for selection
    test_env = make_environment(test_pool[0], test_pool[1], TEST_E, device)

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
            # Evaluate test acc for display only
            with torch.no_grad():
                test_logits = model(data["test"]["images"])
                test_acc    = mean_accuracy(test_logits, data["test"]["labels"])
            print("   ".join([
                str(np.int32(step)).ljust(13),
                fmt(train_nll.detach().cpu().numpy()),
                fmt(train_acc.detach().cpu().numpy()),
                fmt(train_penalty.detach().cpu().numpy()),
                fmt(test_acc.detach().cpu().numpy()),
            ]))

    # Final evaluation on all environments
    with torch.no_grad():
        for env in train_envs + data["val"]:
            logits     = model(env["images"])
            env["acc"] = mean_accuracy(logits, env["labels"])
        test_logits         = model(data["test"]["images"])
        data["test"]["acc"] = mean_accuracy(test_logits, data["test"]["labels"])

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
    print(f"Using device: {device}")
    print(f"N environments: {flags.n_envs}  |  "
          f"e values: {ENV_E_VALUES[flags.n_envs]}")
    print(f"Selection method: {flags.selection_method}\n")

    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    data = load_environments(flags.n_envs, flags.selection_method, device)

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
    print(f"N envs: {flags.n_envs}  |  Selection: {flags.selection_method}")
    print(f"Test acc (mean ± std): "
          f"{np.mean(final_test_accs):.4f} ± "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()