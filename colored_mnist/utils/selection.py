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
from utils.dataset import make_environment as _default_make_env


def load_environments(
    selection_method: str,
    train_pool: tuple,
    test_pool: tuple,
    e_values: list,
    device,
    val_fraction: float = 0.2,
    make_env_fn=None,
    test_e: float = 0.9,
) -> dict:
    if make_env_fn is None:
        make_env_fn = _default_make_env

    n_envs     = len(e_values)
    n_total    = len(train_pool[0])
    split_size = n_total // n_envs

    all_envs = []
    for i, e in enumerate(e_values):
        start = i * split_size
        end   = start + split_size if i < n_envs - 1 else n_total
        all_envs.append(make_env_fn(
            train_pool[0][start:end],
            train_pool[1][start:end],
            e, device))

    test_env = make_env_fn(
        test_pool[0], test_pool[1], test_e, device)

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
            train_envs.append(make_env_fn(
                env_images[:n_train], env_labels[:n_train], e, device))
            val_envs.append(make_env_fn(
                env_images[n_train:], env_labels[n_train:], e, device))
        return {"train": train_envs, "val": val_envs, "test": test_env, "folds": None}

    elif selection_method == "leave_one_domain_out":
        if n_envs == 1:
            folds = [{"train": all_envs, "val": all_envs[0]}]
        else:
            folds = []
            for i in range(n_envs):
                fold_train = [all_envs[j] for j in range(n_envs) if j != i]
                folds.append({"train": fold_train, "val": all_envs[i]})
        return {"train": all_envs, "val": all_envs, "test": test_env, "folds": folds}

    else:  # oracle
        return {"train": all_envs, "val": all_envs, "test": test_env, "folds": None}


def score_oracle(train_accs, val_accs, test_acc):
    return min(min(train_accs), test_acc)

def score_train_domain_val(train_accs, val_accs, test_acc):
    return min(val_accs)

def score_leave_one_domain_out(fold_val_accs):
    return min(fold_val_accs)

SCORE_FNS = {
    "oracle":               score_oracle,
    "train_domain_val":     score_train_domain_val,
    "leave_one_domain_out": score_leave_one_domain_out,
}
