# main_realistic.py
# Extends main.py with realistic data splits for model selection.
# Original main.py is untouched — this file only adds new functionality.
#
# New data loading functions support three selection strategies:
#   - oracle:             use test env (e=0.9) for selection (original)
#   - train_domain_val:   hold out 20% of each train env for validation
#   - leave_one_domain_out: validate on the left-out train env

import numpy as np
import torch
from torchvision import datasets

# Import everything from original main.py so nothing is duplicated
from main import (
    make_environment,
    MLP,
    mean_nll,
    mean_accuracy,
    irm_penalty,
    train_one_restart,
)


# ---------------------------------------------------------------------------
# New data loading functions
# ---------------------------------------------------------------------------

def load_colored_mnist_train_domain_val(device, val_fraction=0.2):
    """
    Realistic model selection: hold out val_fraction of each train environment.

    Returns 5 environments:
        envs[0]: 80% of train split A (e=0.2) → training
        envs[1]: 80% of train split B (e=0.1) → training
        envs[2]: 20% of train split A (e=0.2) → validation for selection
        envs[3]: 20% of train split B (e=0.1) → validation for selection
        envs[4]: full val set         (e=0.9) → final test (never used for selection)
    """
    mnist = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_data = (mnist.data[:50000], mnist.targets[:50000])
    val_data   = (mnist.data[50000:], mnist.targets[50000:])

    # Shuffle train set (keep images <-> labels aligned)
    rng_state = np.random.get_state()
    np.random.shuffle(train_data[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_data[1].numpy())

    # Split A: even indices
    split_a_images = train_data[0][::2]
    split_a_labels = train_data[1][::2]
    n_a      = len(split_a_images)
    n_a_val  = int(n_a * val_fraction)
    n_a_train = n_a - n_a_val

    # Split B: odd indices
    split_b_images = train_data[0][1::2]
    split_b_labels = train_data[1][1::2]
    n_b      = len(split_b_images)
    n_b_val  = int(n_b * val_fraction)
    n_b_train = n_b - n_b_val

    envs = [
        # Training environments (80%)
        make_environment(split_a_images[:n_a_train], split_a_labels[:n_a_train], 0.2, device),
        make_environment(split_b_images[:n_b_train], split_b_labels[:n_b_train], 0.1, device),
        # Validation environments (20%) — used for selection
        make_environment(split_a_images[n_a_train:], split_a_labels[n_a_train:], 0.2, device),
        make_environment(split_b_images[n_b_train:], split_b_labels[n_b_train:], 0.1, device),
        # Test environment — never used for selection
        make_environment(val_data[0], val_data[1], 0.9, device),
    ]
    return envs


def load_colored_mnist_leave_one_domain_out(device):
    """
    Realistic model selection: leave-one-domain-out.
    Each train environment is used as validation for the other.

    Returns 5 environments:
        envs[0]: train split A (e=0.2) → training
        envs[1]: train split B (e=0.1) → training
        envs[2]: train split A (e=0.2) → validation (left out)
        envs[3]: train split B (e=0.1) → validation (left out)
        envs[4]: full val set  (e=0.9) → final test (never used for selection)

    Note: envs[2] and envs[3] are the same data as envs[0] and envs[1]
    but are kept separate to make the selection logic explicit and clean.
    """
    mnist = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_data = (mnist.data[:50000], mnist.targets[:50000])
    val_data   = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(train_data[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_data[1].numpy())

    envs = [
        # Training environments
        make_environment(train_data[0][::2],  train_data[1][::2],  0.2, device),
        make_environment(train_data[0][1::2], train_data[1][1::2], 0.1, device),
        # Left-out validation environments (same data, used only for selection)
        make_environment(train_data[0][::2],  train_data[1][::2],  0.2, device),
        make_environment(train_data[0][1::2], train_data[1][1::2], 0.1, device),
        # Test environment — never used for selection
        make_environment(val_data[0], val_data[1], 0.9, device),
    ]
    return envs


# ---------------------------------------------------------------------------
# Realistic training loop
# ---------------------------------------------------------------------------

def train_one_restart_realistic(flags, envs, device, verbose=False):
    """
    Training loop for realistic selection methods.
    Only trains on envs[0] and envs[1] — the validation envs are
    never seen during training, only used for selection in search_realistic.py.

    Returns (env0_acc, env1_acc, val0_acc, val1_acc, test_acc)
    """
    from torch import optim

    model     = MLP(flags.hidden_dim, flags.grayscale_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)

    # Only train on the first two environments
    train_envs = envs[:2]

    for step in range(flags.steps):
        for env in train_envs:
            logits         = model(env["images"])
            env["nll"]     = mean_nll(logits, env["labels"])
            env["acc"]     = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll     = torch.stack([train_envs[0]["nll"],     train_envs[1]["nll"]]).mean()
        train_penalty = torch.stack([train_envs[0]["penalty"], train_envs[1]["penalty"]]).mean()

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

    # Evaluate on all environments after training
    with torch.no_grad():
        for env in envs:
            logits      = model(env["images"])
            env["acc"]  = mean_accuracy(logits, env["labels"])

    env0_acc = envs[0]["acc"].detach().cpu().item()  # train env A
    env1_acc = envs[1]["acc"].detach().cpu().item()  # train env B
    val0_acc = envs[2]["acc"].detach().cpu().item()  # val env A
    val1_acc = envs[3]["acc"].detach().cpu().item()  # val env B
    test_acc = envs[4]["acc"].detach().cpu().item()  # test env

    return env0_acc, env1_acc, val0_acc, val1_acc, test_acc