# plot_dist_sweep.py — Experiment 7
#
# Loads distances.json (from compute_distances.py) and the per-e-value
# training results (from main_dist_sweep.py) and produces the final
# Performance vs Distribution Distance plot.
#
# Three subplots, one per distance metric (MMD, Wasserstein, PAD).
# Each subplot shows IRM and ERM test accuracy on the y-axis and
# the distance between the training environment and the test
# environment (e=0.9) on the x-axis.
#
# Usage
# -----
#   python plot_dist_sweep.py --selection_method oracle

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_accuracies(mode, selection_method):
    """
    Load test accuracies for all e_train values for a given mode
    and selection method. Returns two lists: means and stds.
    """
    means, stds = [], []
    for e in E_VALUES:
        path = (f"results_dist_sweep"
                f"_{mode}"
                f"_e{e:.1f}"
                f"_{selection_method}.json")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — using NaN")
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        with open(path) as f:
            data = json.load(f)
        means.append(data["mean_test_acc"])
        stds.append(data["std_test_acc"])
    return np.array(means), np.array(stds)


def load_distances():
    """Load distances.json produced by compute_distances.py."""
    with open("distances.json") as f:
        data = json.load(f)
    distances = {r["e_train"]: r for r in data["distances"]}
    return distances


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(selection_method):
    print(f"Loading distances...")
    distances = load_distances()

    print(f"Loading IRM results ({selection_method})...")
    irm_means, irm_stds = load_accuracies("irm", selection_method)

    print(f"Loading ERM results ({selection_method})...")
    erm_means, erm_stds = load_accuracies("erm", selection_method)

    metrics = [
        ("MMD",         "mmd",         "Maximum Mean Discrepancy"),
        ("Wasserstein", "wasserstein",  "Wasserstein Distance (Sinkhorn)"),
        ("PAD",         "pad",          "Proxy A-Distance"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (short_name, key, full_name) in zip(axes, metrics):
        dist_vals = np.array([distances[e][key] for e in E_VALUES])

        # Sort by distance for clean line plot
        order = np.argsort(dist_vals)
        d_sorted   = dist_vals[order]
        irm_sorted = irm_means[order]
        erm_sorted = erm_means[order]
        irm_std_s  = irm_stds[order]
        erm_std_s  = erm_stds[order]
        e_sorted   = np.array(E_VALUES)[order]

        # IRM
        ax.plot(d_sorted, irm_sorted, "o-",
                color="steelblue", label="IRM", linewidth=2, markersize=7)
        ax.fill_between(d_sorted,
                        irm_sorted - irm_std_s,
                        irm_sorted + irm_std_s,
                        alpha=0.2, color="steelblue")

        # ERM
        ax.plot(d_sorted, erm_sorted, "s--",
                color="darkorange", label="ERM", linewidth=2, markersize=7)
        ax.fill_between(d_sorted,
                        erm_sorted - erm_std_s,
                        erm_sorted + erm_std_s,
                        alpha=0.2, color="darkorange")

        # Annotate each point with its e value
        for d, irm_a, e in zip(d_sorted, irm_sorted, e_sorted):
            ax.annotate(f"e={e:.1f}",
                        xy=(d, irm_a),
                        xytext=(4, 6),
                        textcoords="offset points",
                        fontsize=7, color="steelblue")

        ax.set_xlabel(f"{full_name}", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.set_title(f"Performance vs {short_name}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.05)

    selection_label = selection_method.replace("_", " ").title()
    fig.suptitle(
        f"Experiment 7 — Performance vs Distribution Distance\n"
        f"(Colored MNIST, single train env, "
        f"test env fixed at e=0.9, selection: {selection_label})",
        fontsize=13)

    plt.tight_layout()
    out_path = f"experiment7_dist_sweep_{selection_method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot — Distribution Distance Sweep (Experiment 7)")
    parser.add_argument("--selection_method", type=str, default="oracle",
                        choices=["oracle", "train_domain_val"])
    args = parser.parse_args()
    make_plot(args.selection_method)