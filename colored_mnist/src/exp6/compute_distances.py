# compute_distances.py — Experiment 6
#
# Computes MMD, Wasserstein (Sinkhorn), and Proxy A-Distance (PAD)
# between the e2 training environment and the fixed test environment
# (e=0.9). e1=0.1 is fixed across all configurations so only e2 varies.
#
# Usage
#   python src/exp6/compute_distances.py
#
# Output
#   results/exp6/distances.json

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset import make_environment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E1       = 0.1
E2_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEST_E   = 0.9


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def compute_mmd(X, Y, kernel_bandwidths=None):
    """
    Maximum Mean Discrepancy with RBF kernel.
    X, Y: (n, d) float tensors.
    Returns scalar distance >= 0.
    """
    # Always compute on CPU with fixed subsampling for reproducibility
    X = X.cpu()
    Y = Y.cpu()
    max_n = 2000
    g = torch.Generator()
    g.manual_seed(42)
    if len(X) > max_n:
        X = X[torch.randperm(len(X), generator=g)[:max_n]]
    g.manual_seed(42)
    if len(Y) > max_n:
        Y = Y[torch.randperm(len(Y), generator=g)[:max_n]]

    if kernel_bandwidths is None:
        XY    = torch.cat([X, Y], dim=0)
        dists = torch.pdist(XY)
        sigma = dists.median().item()
        sigma = max(sigma, 1e-6)
        kernel_bandwidths = [sigma / 2, sigma, sigma * 2]

    def rbf(A, B, sigma):
        diff = A.unsqueeze(1) - B.unsqueeze(0)   # (n, m, d)
        sq   = (diff ** 2).sum(-1)               # (n, m)
        return torch.exp(-sq / (2 * sigma ** 2))

    mmd = 0.0
    for sigma in kernel_bandwidths:
        Kxx  = rbf(X, X, sigma).mean()
        Kyy  = rbf(Y, Y, sigma).mean()
        Kxy  = rbf(X, Y, sigma).mean()
        mmd += (Kxx + Kyy - 2 * Kxy).item()
    return mmd / len(kernel_bandwidths)


def compute_wasserstein(X, Y, n_iters=100, reg=0.05):
    """
    Sinkhorn approximation of Wasserstein distance.
    X, Y: (n, d) float tensors. Subsampled to max_n for speed.
    """
    # Always compute on CPU with fixed subsampling for reproducibility
    X = X.cpu()
    Y = Y.cpu()
    max_n = 1000
    g = torch.Generator()
    g.manual_seed(42)
    if len(X) > max_n:
        X = X[torch.randperm(len(X), generator=g)[:max_n]]
    g.manual_seed(42)
    if len(Y) > max_n:
        Y = Y[torch.randperm(len(Y), generator=g)[:max_n]]

    n, m = len(X), len(Y)
    diff = X.unsqueeze(1) - Y.unsqueeze(0)      # (n, m, d)
    C    = (diff ** 2).sum(-1).sqrt()            # (n, m) cost matrix

    log_a = torch.full((n,), -np.log(n), device=X.device)
    log_b = torch.full((m,), -np.log(m), device=X.device)
    log_K = -C / reg

    log_u = torch.zeros(n, device=X.device)
    for _ in range(n_iters):
        log_v = log_b - torch.logsumexp(
            log_K + log_u.unsqueeze(1), dim=0)
        log_u = log_a - torch.logsumexp(
            log_K + log_v.unsqueeze(0), dim=1)

    log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    T     = log_T.exp()
    return (T * C).sum().item()


def compute_pad(X, Y, n_iters=500, lr=1e-3):
    """
    Proxy A-Distance = 2 * (1 - 2 * min_classifier_error).
    Trains a linear classifier to distinguish X from Y.
    X, Y: (n, d) float tensors.
    Range: [0, 2]. Higher = more separable = farther apart.
    """
    # Always compute on CPU with fixed subsampling for reproducibility
    X = X.cpu()
    Y = Y.cpu()
    max_n = 2000
    g = torch.Generator()
    g.manual_seed(42)
    if len(X) > max_n:
        X = X[torch.randperm(len(X), generator=g)[:max_n]]
    g.manual_seed(42)
    if len(Y) > max_n:
        Y = Y[torch.randperm(len(Y), generator=g)[:max_n]]

    device  = X.device
    data    = torch.cat([X, Y], dim=0)
    labels  = torch.cat([
        torch.zeros(len(X), device=device),
        torch.ones( len(Y), device=device),
    ]).long()

    perm   = torch.randperm(len(data))
    data   = data[perm]
    labels = labels[perm]

    clf     = nn.Linear(data.shape[1], 2).to(device)
    opt     = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_iters):
        opt.zero_grad()
        loss = loss_fn(clf(data), labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = clf(data).argmax(dim=1)
        error = (preds != labels).float().mean().item()

    return max(0.0, 2.0 * (1.0 - 2.0 * error))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mnist      = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_pool = (mnist.data[:50000], mnist.targets[:50000])
    test_pool  = (mnist.data[50000:],  mnist.targets[50000:])

    # Build test environment once
    test_env = make_environment(test_pool[0], test_pool[1], TEST_E, device)
    X_test   = test_env["images"].view(len(test_env["images"]), -1).float()

    results = []

    for e2 in E2_SWEEP:
        print(f"\ne1={E1:.1f}, e2={e2:.1f} -> test={TEST_E:.1f}")

        # Build e2 environment using second half of train pool
        half  = len(train_pool[0]) // 2
        env2  = make_environment(
            train_pool[0][half:], train_pool[1][half:], e2, device)
        X2    = env2["images"].view(len(env2["images"]), -1).float()

        # Distance is between e2 and test only (e1 is constant)
        print(f"  Computing MMD ...", end=" ", flush=True)
        mmd = compute_mmd(X2, X_test)
        print(f"{mmd:.6f}")

        print(f"  Computing Wasserstein ...", end=" ", flush=True)
        wass = compute_wasserstein(X2, X_test)
        print(f"{wass:.6f}")

        print(f"  Computing PAD ...", end=" ", flush=True)
        pad = compute_pad(X2, X_test)
        print(f"{pad:.6f}")

        results.append({
            "e2":          e2,
            "mmd":         mmd,
            "wasserstein": wass,
            "pad":         pad,
        })
        print(f"  MMD={mmd:.4f}  Wass={wass:.4f}  PAD={pad:.4f}")

    out = {
        "e1":       E1,
        "test_e":   TEST_E,
        "note":     "distances computed between e2 env and test env only; e1=0.1 is fixed",
        "distances": results,
    }
    os.makedirs("results/exp6", exist_ok=True)
    out_path = "results/exp6/distances.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDistances saved to {out_path}")


if __name__ == "__main__":
    main()
