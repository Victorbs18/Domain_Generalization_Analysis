# compute_distances.py — Experiment 7
#
# Computes three distribution distance metrics between each training
# environment (e = 0.1 to 0.9) and the fixed test environment (e = 0.9):
#
#   - MMD  (Maximum Mean Discrepancy) with RBF kernel + median heuristic
#   - Wasserstein distance           with PCA-50 + Sinkhorn approximation
#   - PAD  (Proxy A-Distance)        with logistic regression classifier
#
# Distances are computed on flattened image tensors (2 x 14 x 14 = 392 dims).
# Results saved to distances.json for use by plot_dist_sweep.py.
#
# Usage
# -----
#   python compute_distances.py --n_samples 1000

import argparse
import json
import numpy as np
import torch
from torchvision import datasets
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import ot

from main import make_environment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_E   = 0.9
E_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def get_flat_samples(e, all_images, all_labels, device, n_samples):
    """
    Generate n_samples from a Colored MNIST environment with parameter e.
    Returns flat numpy array of shape (n_samples, 392).
    Images are 2-channel 14x14 tensors flattened to 392 dims.
    """
    idx = np.random.choice(len(all_images), n_samples, replace=False)
    env = make_environment(
        all_images[idx],
        all_labels[idx],
        e,
        device,
    )
    # env["images"] shape: (n_samples, 2, 14, 14) -> flatten to (n_samples, 392)
    return env["images"].cpu().numpy().reshape(n_samples, -1)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def median_heuristic(X, Y, subsample=500):
    """Estimate RBF gamma via the median heuristic."""
    XY = np.vstack([X, Y])
    if len(XY) > subsample:
        idx = np.random.choice(len(XY), subsample, replace=False)
        XY  = XY[idx]
    dists     = np.sum((XY[:, None] - XY[None, :]) ** 2, axis=-1)
    median_sq = np.median(dists[dists > 0])
    return 1.0 / (2.0 * median_sq)


def compute_mmd(X, Y):
    """
    Biased MMD² estimator with RBF kernel.
    Gamma set via median heuristic.
    """
    gamma = median_heuristic(X, Y)
    XX    = rbf_kernel(X, X, gamma)
    YY    = rbf_kernel(Y, Y, gamma)
    XY    = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2.0 * XY.mean())


def compute_wasserstein(X, Y, n_components=50, reg=0.01):
    """
    Sinkhorn approximation of 2-Wasserstein distance.
    PCA applied first to reduce from 392 to n_components dims.
    """
    pca = PCA(n_components=n_components)
    pca.fit(np.vstack([X, Y]))
    X_r = pca.transform(X)
    Y_r = pca.transform(Y)

    a = np.ones(len(X_r)) / len(X_r)
    b = np.ones(len(Y_r)) / len(Y_r)
    M = ot.dist(X_r, Y_r, metric="sqeuclidean")
    return float(ot.sinkhorn2(a, b, M, reg=reg)[0])


def compute_pad(X_train, X_test):
    """
    Proxy A-Distance: 2(1 - 2ε) where ε is the error of a
    logistic regression classifier trained to distinguish
    X_train from X_test.
    PAD=0 means indistinguishable; PAD=2 means perfectly separable.
    """
    X = np.concatenate([X_train, X_test])
    y = np.array([0] * len(X_train) + [1] * len(X_test))
    clf    = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    error  = 1.0 - scores.mean()
    return float(2.0 * (1.0 - 2.0 * error))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute distribution distances — Experiment 7")
    parser.add_argument("--n_samples",    type=int, default=1000,
                        help="Samples per distribution")
    parser.add_argument("--n_components", type=int, default=50,
                        help="PCA components for Wasserstein")
    parser.add_argument("--sinkhorn_reg", type=float, default=0.01,
                        help="Sinkhorn regularization")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"Loading MNIST...")
    mnist      = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    all_images = mnist.data
    all_labels = mnist.targets

    print(f"Generating test distribution (e={TEST_E})...")
    X_test = get_flat_samples(
        TEST_E, all_images, all_labels, device, args.n_samples)

    results = []

    for e in E_VALUES:
        print(f"\nProcessing e={e:.1f}...")
        X_train = get_flat_samples(
            e, all_images, all_labels, device, args.n_samples)

        print(f"  Computing MMD...",        end=" ", flush=True)
        mmd  = compute_mmd(X_train, X_test)
        print(f"{mmd:.6f}")

        print(f"  Computing Wasserstein...", end=" ", flush=True)
        wass = compute_wasserstein(
            X_train, X_test,
            n_components=args.n_components,
            reg=args.sinkhorn_reg)
        print(f"{wass:.6f}")

        print(f"  Computing PAD...",        end=" ", flush=True)
        pad  = compute_pad(X_train, X_test)
        print(f"{pad:.6f}")

        results.append({
            "e_train":     e,
            "mmd":         mmd,
            "wasserstein": wass,
            "pad":         pad,
        })

    # Save
    out = {
        "test_e":     TEST_E,
        "n_samples":  args.n_samples,
        "n_components": args.n_components,
        "sinkhorn_reg": args.sinkhorn_reg,
        "distances":  results,
    }
    with open("distances.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("Distances summary:")
    print(f"{'e_train':>8}  {'MMD':>10}  {'Wasserstein':>12}  {'PAD':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['e_train']:>8.1f}  "
              f"{r['mmd']:>10.6f}  "
              f"{r['wasserstein']:>12.6f}  "
              f"{r['pad']:>8.6f}")

    print("\nDistances saved to distances.json")


if __name__ == "__main__":
    main()