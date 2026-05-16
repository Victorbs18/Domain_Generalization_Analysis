# utils/selection.py
# Model selection criteria for Colored MNIST experiments.
#
# Three selection strategies:
#   oracle:               uses test accuracy directly — unrealistic upper bound
#   train_domain_val:     uses held-out 20% of each training environment
#   leave_one_domain_out: trains n_envs models each leaving one env out
#                         n=1: degenerate — train and val on same environment
#                         n=2: each fold trains on 1 env — IRM degrades to ERM
#                         n>=3: each fold trains on n_envs-1 envs — IRM works properly

import numpy as np
from utils.dataset import make_environment


def load_environments(
    selection_method: str,
    train_pool: tuple,
    test_pool: tuple,
    e_values: list,
    device,
    val_fraction: float = 0.2,
) -> dict:
    """
    Build train, val and test environments for a given selection method.

    Args:
        selection_method: 'oracle', 'train_domain_val',
                          or 'leave_one_domain_out'
        train_pool:       tuple (images, labels) — 50,000 MNIST train samples
        test_pool:        tuple (images, labels) — 10,000 MNIST test samples
        e_values:         list of e values for training environments
        device:           torch device
        val_fraction:     fraction held out for train_domain_val (default 0.2)

    Returns:
        dict with keys:
            'train': list of all training environments
            'val':   list of validation environments
            'test':  single test environment (never used for selection)
            'folds': list of folds for leave_one_domain_out, each fold is:
                        'train': list of n_envs-1 training environments
                        'val':   single held-out validation environment
                     None for oracle and train_domain_val
    """
    n_envs     = len(e_values)
    n_total    = len(train_pool[0])
    split_size = n_total // n_envs

    # Build all full environments
    all_envs = []
    for i, e in enumerate(e_values):
        start = i * split_size
        end   = start + split_size if i < n_envs - 1 else n_total
        all_envs.append(make_environment(
            train_pool[0][start:end],
            train_pool[1][start:end],
            e, device))

    # Test environment — always fixed at e=0.9
    test_env = make_environment(
        test_pool[0], test_pool[1], 0.9, device)

    if selection_method == "train_domain_val":
        train_envs = []
        val_envs   = []
        for i, e in enumerate(e_values):
            start      = i * split_size
            end        = start + split_size if i < n_envs - 1 else n_total
            env_images = train_pool[0][start:end]
            env_labels = train_pool[1][start:end]
            n_val      = int(len(env_images) * val_fraction)
            n_train    = len(env_images) - n_val
            train_envs.append(make_environment(
                env_images[:n_train], env_labels[:n_train], e, device))
            val_envs.append(make_environment(
                env_images[n_train:], env_labels[n_train:], e, device))
        return {
            "train": train_envs,
            "val":   val_envs,
            "test":  test_env,
            "folds": None,
        }

    elif selection_method == "leave_one_domain_out":
        if n_envs == 1:
            # Degenerate case — train and validate on same environment
            folds = [{"train": all_envs, "val": all_envs[0]}]
        else:
            folds = []
            for i in range(n_envs):
                fold_train = [all_envs[j] for j in range(n_envs) if j != i]
                fold_val   = all_envs[i]
                folds.append({
                    "train": fold_train,
                    "val":   fold_val,
                })
        return {
            "train": all_envs,
            "val":   all_envs,
            "test":  test_env,
            "folds": folds,
        }

    else:
        # Oracle — no split needed
        return {
            "train": all_envs,
            "val":   all_envs,
            "test":  test_env,
            "folds": None,
        }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_oracle(train_accs: list, val_accs: list, test_acc: float) -> float:
    """
    Oracle: min(train_accs + [test_acc])
    Unrealistic — uses test accuracy directly.
    """
    return min(min(train_accs), test_acc)


def score_train_domain_val(train_accs: list, val_accs: list, test_acc: float) -> float:
    """
    Train domain val: min(val_accs)
    Val set is genuinely held out from training.
    """
    return min(val_accs)


def score_leave_one_domain_out(fold_val_accs: list) -> float:
    """
    Leave one domain out: min(val_acc from each fold)
    For n=1: degenerate — val = train data
    For n=2: each fold trains on 1 env — IRM degrades to ERM
    For n>=3: each fold trains on n_envs-1 envs — IRM works properly
    """
    return min(fold_val_accs)


SCORE_FNS = {
    "oracle":               score_oracle,
    "train_domain_val":     score_train_domain_val,
    "leave_one_domain_out": score_leave_one_domain_out,
}