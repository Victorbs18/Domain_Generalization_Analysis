# utils/losses.py
# Loss functions and IRM penalty for Colored MNIST.

import torch
from torch import nn, autograd


def mean_nll(logits, y):
    """
    Binary cross-entropy loss.
    Takes raw logits and binary labels.
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    """
    Binary classification accuracy.
    Threshold at 0.
    """
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def irm_penalty(logits, y):
    """
    IRM invariance penalty.

    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss  = mean_nll(logits * scale, y)
    grad  = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)