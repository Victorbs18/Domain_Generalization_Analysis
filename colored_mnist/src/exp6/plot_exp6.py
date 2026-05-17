# plot_exp6.py — Experiment 6
#
# Loads distances.json and per-(mode, e2, selection_method) result JSONs
# and produces a grid plot:
#   rows: selection methods (oracle, train_domain_val, leave_one_domain_out)
#   cols: distance metrics (MMD, Wasserstein, PAD)
#
# Each subplot shows IRM and ERM test accuracy (y) vs distance (x).
#
# Usage
#   python src/exp6/plot_exp6.py
#
# Output
#   results/exp6/exp6_performance_vs_distance.png

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E2_SWEEP   = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SELECTIONS = ["oracle", "train_domain_val", "leave_one_domain_out"]
METRICS    = [
    ("mmd",         "MMD"),
    ("wasserstein", "Wasserstein"),
    ("pad",         "PAD"),
]
SELECTION_LABELS = {
    "oracle":               "Oracle",
    "train_domain_val":     "Train Domain Val",
    "leave_one_domain_out": "Leave One Domain Out",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_distances(results_dir):
    path = os.path.join(results_dir, "distances.json")
    with open(path) as f:
        data = json.load(f)
    dist_by_e2 = {r["e2"]: r for r in data["distances"]}
    return dist_by_e2


def load_accuracies(results_dir, mode, selection_method):
    means, stds = [], []
    for e2 in E2_SWEEP:
        path = os.path.join(
            results_dir,
            f"results_{mode}_e{e2:.1f}_{selection_method}.json")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — using NaN")
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        with open(path) as f:
            d = json.load(f)
        means.append(d["mean_test_acc"])
        stds.append(d["std_test_acc"])
    return np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results_dir):
    dist_by_e2 = load_distances(results_dir)

    n_rows = len(SELECTIONS)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=False, sharey=False)

    for row_idx, selection in enumerate(SELECTIONS):
        irm_means, irm_stds = load_accuracies(
            results_dir, "irm", selection)
        erm_means, erm_stds = load_accuracies(
            results_dir, "erm", selection)

        for col_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]

            dist_vals = np.array(
                [dist_by_e2[e2][metric_key] for e2 in E2_SWEEP])

            # Sort by distance for clean line
            order      = np.argsort(dist_vals)
            d_sorted   = dist_vals[order]
            irm_sorted = irm_means[order]
            erm_sorted = erm_means[order]
            irm_std_s  = irm_stds[order]
            erm_std_s  = erm_stds[order]
            e_sorted   = np.array(E2_SWEEP)[order]

            # IRM
            ax.plot(d_sorted, irm_sorted, "o-",
                    color="steelblue", label="IRM",
                    linewidth=2, markersize=6)
            ax.fill_between(
                d_sorted,
                irm_sorted - irm_std_s,
                irm_sorted + irm_std_s,
                alpha=0.2, color="steelblue")

            # ERM
            ax.plot(d_sorted, erm_sorted, "s--",
                    color="darkorange", label="ERM",
                    linewidth=2, markersize=6)
            ax.fill_between(
                d_sorted,
                erm_sorted - erm_std_s,
                erm_sorted + erm_std_s,
                alpha=0.2, color="darkorange")

            # Annotate e2 values
            for d, irm_a, e2 in zip(d_sorted, irm_sorted, e_sorted):
                ax.annotate(
                    f"e={e2:.1f}",
                    xy=(d, irm_a),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=7, color="steelblue")

            ax.set_xlabel(metric_label, fontsize=11)
            ax.set_ylabel("Test Accuracy", fontsize=10)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(
                    f"{SELECTION_LABELS[selection]}\n\nTest Accuracy",
                    fontsize=10)

            # Column title on top row
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=12, fontweight="bold")

    fig.suptitle(
        "Experiment 6 — Performance vs Distribution Distance\n"
        "e1=0.1 fixed, e2 swept {0.2..0.9}, test env e=0.9",
        fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(
        results_dir, "exp6_performance_vs_distance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = "results/exp6"
    make_plot(results_dir)
