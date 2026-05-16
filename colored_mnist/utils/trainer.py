# utils/trainer.py
# Shared training loop for Colored MNIST experiments.

import numpy as np
import torch
from torch import optim
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.model   import MLP
from utils.losses  import mean_nll, mean_accuracy, irm_penalty


def train_one_restart(flags, train_envs, device, verbose=True):
    """
    Train one full restart on the given training environments.

    Args:
        flags:      argparse namespace with hyperparameters:
                        hidden_dim, l2_regularizer_weight, lr,
                        penalty_anneal_iters, penalty_weight,
                        steps, grayscale_model
        train_envs: list of training environments, each a dict with
                        'images' and 'labels'
        device:     torch device
        verbose:    print training progress every 100 steps

    Returns:
        dict with keys:
            'train_accs': list of final accuracies per training environment
            'mean_train': mean accuracy across training environments
    """
    grayscale = getattr(flags, 'grayscale_model', False)
    model     = MLP(flags.hidden_dim, grayscale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)

    if verbose:
        cols = ["step", "train_nll", "train_acc", "train_penalty"]
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
            flags.penalty_weight
            if step >= flags.penalty_anneal_iters
            else 1.0
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
            print("   ".join([
                str(np.int32(step)).ljust(13),
                fmt(train_nll.detach().cpu().numpy()),
                fmt(train_acc.detach().cpu().numpy()),
                fmt(train_penalty.detach().cpu().numpy()),
            ]))

    # Final evaluation
    with torch.no_grad():
        for env in train_envs:
            logits     = model(env["images"])
            env["acc"] = mean_accuracy(logits, env["labels"])

    train_accs = [e["acc"].detach().cpu().item() for e in train_envs]

    return {
        "model":       model,
        "train_accs":  train_accs,
        "mean_train":  float(np.mean(train_accs)),
    }