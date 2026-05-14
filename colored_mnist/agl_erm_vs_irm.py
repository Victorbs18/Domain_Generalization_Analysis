"""
ERM vs IRM Agreement-on-the-Line — Colored MNIST
=================================================
Drop into colored_mnist/ alongside main.py.

THE HONEST PIPELINE
-------------------
In a real deployment you don't know which hyperparameters work best OOD.
This script simulates that realistic scenario:

  Phase 1 — Run a grid search over the SAME hyperparameter ranges used
             in your original search.py for both ERM and IRM.
             Each (hp_config, seed) pair produces one trained model.

  Phase 2 — Fit the AGL line from ERM-ERM pairs using only ID agreement
             (no OOD labels needed).

  Phase 3 — For each IRM config, compute IRM-ERM agreement on unlabeled
             OOD data. Check if it falls below the AGL line.

  Phase 4 — DECISION:
             * All IRM-ERM pairs on the line  -> IRM adds no value,
               pick best ERM config by ID validation accuracy.
             * IRM-ERM pairs below the line   -> IRM diverges from ERM
               under shift. Use agreement deviation as the selection
               criterion to pick the best IRM config — no OOD labels.

  Phase 5 — Compare: does agreement-selected IRM match oracle-selected
             IRM? This is the validation of H1 and H2.

Hyperparameter ranges mirror your original search.py ranges exactly.

Usage:
    python agl_erm_vs_irm.py --env_config diverse --n_trials 20 --n_seeds 3 --device cpu
    python agl_erm_vs_irm.py --all_configs --n_trials 30 --n_seeds 3 --device cpu
"""

import argparse
import itertools
import json
import os
import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Hyperparameter ranges — mirrors your original search.py exactly
# =============================================================================

HP_RANGES = {
    "hidden_dim":           [32, 64, 80, 85, 88, 92, 95, 99, 100, 102, 103,
                             114, 117, 119, 138, 176, 178, 215, 237, 253,
                             254, 256, 258, 301, 309, 319, 340, 341, 423,
                             437, 486],
    "lr":                   [1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3],
    "l2_reg":               [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "steps":                [101, 201, 301, 401, 501],
    "penalty_anneal_iters": [50, 80, 100, 150, 200, 250],
    "penalty_weight":       [10, 100, 500, 1000, 5000, 10000,
                             50000, 100000, 500000, 700000],
}

ENV_CONFIGS = {
    "original":  [0.1, 0.2],
    "diverse":   [0.1, 0.5],
    "proximate": [0.7, 0.8],
}
E_TEST              = 0.9
DEVIATION_THRESHOLD = -0.10


# =============================================================================
# Data
# =============================================================================

def load_mnist_raw():
    from torchvision import datasets
    print("Loading MNIST (once)...")
    mnist        = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_images = mnist.data[:50000]
    train_labels = mnist.targets[:50000]
    val_images   = mnist.data[50000:]
    val_labels   = mnist.targets[50000:]
    print(f"  train: {train_images.shape}   val: {val_images.shape}")
    return train_images, train_labels, val_images, val_labels


def make_environment(images, labels, e, seed=0):
    rng    = np.random.RandomState(seed)
    labels = (labels < 5).float()
    labels = torch.logical_xor(labels, torch.from_numpy(
        rng.binomial(1, 0.25, size=labels.shape).astype(bool)
    )).float()
    colors = torch.logical_xor(labels.bool(), torch.from_numpy(
        rng.binomial(1, 1 - e, size=labels.shape).astype(bool)
    )).float()
    imgs = images.float() / 255.0
    imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)
    imgs[:, 0, :, :] *= (1 - colors).unsqueeze(1).unsqueeze(2)
    imgs[:, 1, :, :] *= colors.unsqueeze(1).unsqueeze(2)
    imgs[:, 2, :, :] *= 0.0
    return {"images": imgs.view(imgs.shape[0], -1), "labels": labels.unsqueeze(1)}


def build_envs(train_images, train_labels, e_values, seed):
    rng  = np.random.RandomState(seed)
    idx  = rng.permutation(len(train_images))
    imgs = train_images[idx]
    lbls = train_labels[idx]
    n    = len(imgs) // len(e_values)
    return [make_environment(imgs[i*n:(i+1)*n], lbls[i*n:(i+1)*n], e, seed=seed+i)
            for i, e in enumerate(e_values)]


def make_val_splits(envs, val_frac=0.2):
    """Hold out val_frac of each env for train-domain validation."""
    train_envs, val_envs = [], []
    for env in envs:
        n     = len(env["images"])
        split = int((1 - val_frac) * n)
        train_envs.append({"images": env["images"][:split],
                            "labels": env["labels"][:split]})
        val_envs.append(  {"images": env["images"][split:],
                            "labels": env["labels"][split:]})
    return train_envs, val_envs


# =============================================================================
# Model & training
# =============================================================================

class MLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 28 * 28, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x)


def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss  = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad  = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def train_model(envs, hp, method, device, seed):
    torch.manual_seed(seed)
    model     = MLP(hp["hidden_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp["lr"], weight_decay=hp["l2_reg"])
    loss_fn   = nn.BCEWithLogitsLoss()
    pw        = hp.get("penalty_weight", 0.0)
    pa        = hp.get("penalty_anneal_iters", 0)

    for step in range(hp["steps"]):
        model.train()
        total   = torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)
        for env in envs:
            x, y   = env["images"].to(device), env["labels"].to(device)
            logits  = model(x)
            total  += loss_fn(logits, y)
            if method == "irm":
                penalty += irm_penalty(logits, y)
        total   /= len(envs)
        penalty /= len(envs)
        if method == "irm" and pw > 0:
            w    = pw if step >= pa else 1.0
            loss = (total + w * penalty) / (w if w > 1 else 1)
        else:
            loss = total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def get_preds(model, data, device):
    model.eval()
    return (model(data["images"].to(device)) > 0).float().cpu()

def compute_acc(model, data, device):
    return (get_preds(model, data, device) == data["labels"].cpu()).float().mean().item()

def compute_agr(p1, p2):
    return (p1 == p2).float().mean().item()

def worst_env_val_acc(model, val_envs, device):
    """Your existing selection criterion: min val acc across environments."""
    return min(compute_acc(model, v, device) for v in val_envs)


# =============================================================================
# AGL helpers
# =============================================================================

def probit(p, eps=1e-6):
    return float(stats.norm.ppf(np.clip(p, eps, 1 - eps)))

def inv_probit(z):
    return float(stats.norm.cdf(z))

def fit_line(points):
    x = np.array([probit(p["id_agr"])  for p in points])
    y = np.array([probit(p["ood_agr"]) for p in points])
    s, i, r, _, _ = stats.linregress(x, y)
    return s, i, r ** 2

def deviation(id_agr, ood_agr, slope, intercept):
    """Signed deviation in probit space. Negative = below the line."""
    return probit(ood_agr) - (slope * probit(id_agr) + intercept)


# =============================================================================
# HP sampling — mirrors your search.py random search logic
# =============================================================================

def sample_erm_hp(rng):
    return {
        "hidden_dim": int(rng.choice(HP_RANGES["hidden_dim"])),
        "lr":         float(rng.choice(HP_RANGES["lr"])),
        "l2_reg":     float(rng.choice(HP_RANGES["l2_reg"])),
        "steps":      int(rng.choice(HP_RANGES["steps"])),
    }

def sample_irm_hp(rng):
    hp = sample_erm_hp(rng)
    hp["penalty_weight"]       = float(rng.choice(HP_RANGES["penalty_weight"]))
    hp["penalty_anneal_iters"] = int(rng.choice(HP_RANGES["penalty_anneal_iters"]))
    return hp


# =============================================================================
# Main experiment
# =============================================================================

def run(env_config, n_trials, n_seeds, device, output_dir, mnist_raw):
    os.makedirs(output_dir, exist_ok=True)
    e_values = ENV_CONFIGS[env_config]
    train_images, train_labels, val_images, val_labels = mnist_raw
    rng = np.random.RandomState(42)

    # Fixed reference environments for agreement (no labels used for OOD)
    ref_envs = build_envs(train_images, train_labels, e_values, seed=0)
    ref_test = make_environment(val_images, val_labels, E_TEST, seed=99)
    id_ref   = ref_envs[0]

    print(f"\n{'='*65}")
    print(f"  ERM vs IRM AGL Diagnostic  |  {env_config}")
    print(f"  e_train={e_values}  e_test={E_TEST}")
    print(f"  n_trials={n_trials} random HP configs  x  n_seeds={n_seeds} per config")
    print(f"  HP ranges: same as your original search.py")
    print(f"{'='*65}\n")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 1 — Random HP search: train ERM and IRM across n_trials configs
    # ------------------------------------------------------------------
    # For each trial:
    #   - Sample one ERM HP config and one IRM HP config from the ranges
    #   - Train n_seeds models per config (different data shuffles)
    #   - Record ID val accuracy (for non-oracle selection) and OOD accuracy
    #     (for oracle comparison only — not used in the AGL selection)

    print(f"[Phase 1] Training {n_trials} HP configs x {n_seeds} seeds...\n")

    erm_configs = []
    irm_configs = []

    for trial in range(n_trials):
        erm_hp = sample_erm_hp(rng)
        irm_hp = sample_irm_hp(rng)

        erm_entry = {"hp": erm_hp, "models": []}
        irm_entry = {"hp": irm_hp, "models": []}

        for seed in range(n_seeds):
            envs             = build_envs(train_images, train_labels, e_values, seed=seed)
            test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
            train_envs, val_envs = make_val_splits(envs)

            # ERM
            m_erm      = train_model(train_envs, erm_hp, "erm", device, seed)
            erm_val_a  = worst_env_val_acc(m_erm, val_envs, device)
            erm_ood_a  = compute_acc(m_erm, test_env, device)
            erm_entry["models"].append({
                "model": m_erm, "id_val_acc": erm_val_a,
                "ood_acc": erm_ood_a, "seed": seed
            })

            # IRM
            m_irm      = train_model(train_envs, irm_hp, "irm", device, seed)
            irm_val_a  = worst_env_val_acc(m_irm, val_envs, device)
            irm_ood_a  = compute_acc(m_irm, test_env, device)
            irm_entry["models"].append({
                "model": m_irm, "id_val_acc": irm_val_a,
                "ood_acc": irm_ood_a, "seed": seed
            })

        # Aggregate stats per config
        erm_entry["mean_id_val_acc"] = float(np.mean([m["id_val_acc"] for m in erm_entry["models"]]))
        erm_entry["mean_ood_acc"]    = float(np.mean([m["ood_acc"]    for m in erm_entry["models"]]))
        irm_entry["mean_id_val_acc"] = float(np.mean([m["id_val_acc"] for m in irm_entry["models"]]))
        irm_entry["mean_ood_acc"]    = float(np.mean([m["ood_acc"]    for m in irm_entry["models"]]))

        erm_configs.append(erm_entry)
        irm_configs.append(irm_entry)

        print(f"  trial={trial:02d} | "
              f"ERM  val={erm_entry['mean_id_val_acc']:.3f} ood={erm_entry['mean_ood_acc']:.3f} | "
              f"IRM  val={irm_entry['mean_id_val_acc']:.3f} ood={irm_entry['mean_ood_acc']:.3f} "
              f"λ={irm_hp['penalty_weight']:.0f}")

    print(f"\n  Training done in {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Phase 2 — Fit AGL line from all ERM-ERM pairs
    # ------------------------------------------------------------------

    print(f"\n[Phase 2] Fitting AGL line from ERM-ERM pairs...")

    erm_all = [m for cfg in erm_configs for m in cfg["models"]]
    erm_erm_points = []
    for i, j in itertools.combinations(range(len(erm_all)), 2):
        mi, mj = erm_all[i]["model"], erm_all[j]["model"]
        erm_erm_points.append({
            "id_agr":  compute_agr(get_preds(mi, id_ref,  device),
                                   get_preds(mj, id_ref,  device)),
            "ood_agr": compute_agr(get_preds(mi, ref_test, device),
                                   get_preds(mj, ref_test, device)),
        })

    slope, intercept, r2 = fit_line(erm_erm_points)
    print(f"  ERM-ERM pairs: {len(erm_erm_points)}")
    print(f"  AGL line: slope={slope:.3f}  intercept={intercept:.3f}  R²={r2:.3f}")

    # ------------------------------------------------------------------
    # Phase 3 — Compute IRM-ERM agreement per config, check deviation
    # ------------------------------------------------------------------

    print(f"\n[Phase 3] Computing IRM-ERM agreement per HP config...")

    for trial in range(n_trials):
        for seed in range(n_seeds):
            irm_m  = irm_configs[trial]["models"][seed]["model"]
            erm_m  = erm_configs[trial]["models"][seed]["model"]
            id_agr  = compute_agr(get_preds(irm_m, id_ref,  device),
                                  get_preds(erm_m, id_ref,  device))
            ood_agr = compute_agr(get_preds(irm_m, ref_test, device),
                                  get_preds(erm_m, ref_test, device))
            dev     = deviation(id_agr, ood_agr, slope, intercept)
            irm_configs[trial]["models"][seed].update({
                "id_agr": id_agr, "ood_agr": ood_agr, "deviation": dev
            })

        ms = irm_configs[trial]["models"]
        irm_configs[trial]["mean_deviation"]  = float(np.mean([m["deviation"] for m in ms]))
        irm_configs[trial]["n_below"]         = sum(1 for m in ms
                                                    if m["deviation"] < DEVIATION_THRESHOLD)

    # ------------------------------------------------------------------
    # Phase 4 — Decision
    # ------------------------------------------------------------------

    print(f"\n[Phase 4] Decision...")

    majority     = n_seeds // 2 + 1
    n_diverging  = sum(1 for c in irm_configs if c["n_below"] >= majority)
    frac_diverg  = n_diverging / n_trials

    if frac_diverg >= 0.3:
        decision      = "DIVERGE"
        decision_text = (f"{n_diverging}/{n_trials} IRM configs diverge from ERM under shift. "
                         f"IRM is doing real work. Use agreement deviation to select best config.")
    else:
        decision      = "AGREE"
        decision_text = (f"Only {n_diverging}/{n_trials} IRM configs diverge. "
                         f"IRM adds little value. Stick with best ERM (selected by ID val acc).")

    print(f"\n  *** DECISION: {decision} ***")
    print(f"  {decision_text}")

    # ------------------------------------------------------------------
    # Phase 5 — HP selection comparison (the key result)
    # ------------------------------------------------------------------

    print(f"\n[Phase 5] Comparing selection strategies...")

    # Oracle: best by OOD accuracy (requires OOD labels — not available in practice)
    oracle_irm = max(irm_configs, key=lambda c: c["mean_ood_acc"])
    oracle_erm = max(erm_configs, key=lambda c: c["mean_ood_acc"])

    # Standard non-oracle: best by ID val accuracy
    val_irm = max(irm_configs, key=lambda c: c["mean_id_val_acc"])
    val_erm = max(erm_configs, key=lambda c: c["mean_id_val_acc"])

    # Agreement-based: among diverging configs, pick best by ID val acc
    diverging = [c for c in irm_configs if c["n_below"] >= majority]
    if diverging:
        agr_irm = max(diverging, key=lambda c: c["mean_id_val_acc"])
    else:
        # No diverging configs — fallback to val_erm
        agr_irm = val_erm

    print(f"\n  {'Strategy':<35} {'OOD acc':>9}  {'λ (if IRM)':>12}")
    print(f"  {'─'*60}")
    print(f"  {'ERM — oracle (OOD labels)':<35} {oracle_erm['mean_ood_acc']:>9.3f}  {'—':>12}")
    print(f"  {'ERM — ID val selection':<35} {val_erm['mean_ood_acc']:>9.3f}  {'—':>12}")
    print(f"  {'IRM — oracle (OOD labels)':<35} {oracle_irm['mean_ood_acc']:>9.3f}  "
          f"{oracle_irm['hp']['penalty_weight']:>12.0f}")
    print(f"  {'IRM — ID val selection':<35} {val_irm['mean_ood_acc']:>9.3f}  "
          f"{val_irm['hp']['penalty_weight']:>12.0f}")
    agr_ood = agr_irm["mean_ood_acc"] if decision == "DIVERGE" else val_erm["mean_ood_acc"]
    agr_lam = f"{agr_irm['hp'].get('penalty_weight', 0):.0f}" if decision == "DIVERGE" else "— (ERM)"
    print(f"  {'Agreement selection (H1+H2)':<35} {agr_ood:>9.3f}  {agr_lam:>12}")

    gap_agr_vs_oracle = oracle_irm["mean_ood_acc"] - agr_ood
    gap_val_vs_oracle = oracle_irm["mean_ood_acc"] - val_irm["mean_ood_acc"]
    print(f"\n  Gap: oracle IRM vs agreement-selection : {gap_agr_vs_oracle:+.3f}")
    print(f"  Gap: oracle IRM vs ID-val-selection    : {gap_val_vs_oracle:+.3f}")
    print(f"  (Smaller gap = better label-free selection)")

    # Per-trial detail table
    print(f"\n  Per-trial breakdown (* = agr-selected   O = oracle IRM):")
    print(f"  {'t':>3} {'λ':>8} {'IRM val':>8} {'IRM ood':>8} "
          f"{'ERM ood':>8} {'adv':>7} {'mean dev':>9} {'n_below':>8}")
    print(f"  {'─'*70}")
    for t, (ic, ec) in enumerate(zip(irm_configs, erm_configs)):
        adv  = ic["mean_ood_acc"] - ec["mean_ood_acc"]
        flag = ""
        if ic is agr_irm:    flag += " *"
        if ic is oracle_irm: flag += " O"
        print(f"  {t:>3} {ic['hp']['penalty_weight']:>8.0f} "
              f"{ic['mean_id_val_acc']:>8.3f} {ic['mean_ood_acc']:>8.3f} "
              f"{ec['mean_ood_acc']:>8.3f} {adv:>+7.3f} "
              f"{ic['mean_deviation']:>+9.3f} "
              f"{ic['n_below']:>5}/{n_seeds}{flag}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    print(f"\n[Plotting...]")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: AGL scatter
    ax = axes[0]
    eid  = [p["id_agr"]  for p in erm_erm_points]
    eood = [p["ood_agr"] for p in erm_erm_points]
    ax.scatter(eid, eood, color="gray", alpha=0.3, s=18,
               label=f"ERM-ERM ({len(erm_erm_points)})", zorder=2)
    xs = np.linspace(min(eid)-0.02, max(eid)+0.02, 100)
    ax.plot(xs, [inv_probit(slope*probit(x)+intercept) for x in xs],
            "gray", linewidth=1.5, linestyle="--",
            label=f"AGL line R²={r2:.2f}", zorder=2)
    for t, ic in enumerate(irm_configs):
        is_div = ic["n_below"] >= majority
        col    = "#D85A30" if is_div else "#185FA5"
        mkr    = "v"       if is_div else "^"
        for m in ic["models"]:
            ax.scatter(m["id_agr"], m["ood_agr"],
                       color=col, marker=mkr, s=45, alpha=0.7, zorder=3)
    ax.legend(handles=[
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#D85A30",
               markersize=8, label="IRM-ERM below line"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#185FA5",
               markersize=8, label="IRM-ERM on line"),
    ] + ax.get_legend_handles_labels()[0], fontsize=8)
    ax.set_xlabel("ID agreement"); ax.set_ylabel("OOD agreement")
    ax.set_title(f"AGL scatter — {env_config}\nDecision: {decision}", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Middle: Deviation vs IRM advantage
    ax = axes[1]
    devs = [c["mean_deviation"] for c in irm_configs]
    advs = [c["mean_ood_acc"] - erm_configs[i]["mean_ood_acc"]
            for i, c in enumerate(irm_configs)]
    cols = ["#D85A30" if c["n_below"] >= majority else "#185FA5"
            for c in irm_configs]
    ax.scatter(devs, advs, c=cols, s=60, alpha=0.85, zorder=3)
    ax.axvline(DEVIATION_THRESHOLD, color="gray", linestyle=":",
               linewidth=1, label=f"threshold={DEVIATION_THRESHOLD}")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    if len(devs) > 2:
        m_f, b_f = np.polyfit(devs, advs, 1)
        xf       = np.linspace(min(devs)-0.05, max(devs)+0.05, 100)
        corr, pv = stats.pearsonr(devs, advs)
        ax.plot(xf, m_f*xf + b_f, color="gray", linewidth=1.5,
                alpha=0.6, label=f"r={corr:.2f}  p={pv:.3f}")
    ax.set_xlabel("Mean deviation (probit)\n← below   |   above →")
    ax.set_ylabel("IRM OOD acc − ERM OOD acc")
    ax.set_title("Does deviation predict IRM advantage?", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: Selection strategy bar chart (the main result)
    ax = axes[2]
    strategies = ["ERM\nOracle", "ERM\nID val",
                  "IRM\nOracle", "IRM\nID val", "IRM\nAgreement"]
    ood_vals   = [oracle_erm["mean_ood_acc"], val_erm["mean_ood_acc"],
                  oracle_irm["mean_ood_acc"], val_irm["mean_ood_acc"],
                  agr_ood]
    bar_cols   = ["#888780", "#888780", "#185FA5", "#1D9E75", "#D85A30"]
    hatches    = ["//", "", "//", "", ""]
    bars = ax.bar(strategies, ood_vals, color=bar_cols,
                  hatch=hatches, alpha=0.85, width=0.55)
    for bar, v in zip(bars, ood_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(oracle_irm["mean_ood_acc"], color="#185FA5",
               linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean OOD accuracy")
    ax.set_title("HP selection strategy comparison\n(// = requires OOD labels)", fontsize=10)
    ax.set_ylim(0, min(1.0, max(ood_vals) + 0.1))
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"ERM vs IRM AGL Diagnostic — {env_config}  "
        f"e_train={ENV_CONFIGS[env_config]}  e_test={E_TEST}  "
        f"n_trials={n_trials}  n_seeds={n_seeds}\n"
        f"AGL R²={r2:.3f}  |  Decision: {decision}  |  "
        f"Agreement-selection gap vs oracle IRM: {gap_agr_vs_oracle:+.3f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"agl_erm_vs_irm_{env_config}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {plot_path}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    def strip(cfg):
        out = {k: v for k, v in cfg.items() if k != "models"}
        out["models"] = [{k2: v2 for k2, v2 in m.items() if k2 != "model"}
                         for m in cfg["models"]]
        return out

    result = {
        "env_config": env_config, "e_train": e_values, "e_test": E_TEST,
        "n_trials": n_trials, "n_seeds": n_seeds,
        "agl_line": {"slope": slope, "intercept": intercept, "r2": r2},
        "decision": decision, "decision_text": decision_text,
        "selection_comparison": {
            "erm_oracle":        oracle_erm["mean_ood_acc"],
            "erm_id_val":        val_erm["mean_ood_acc"],
            "irm_oracle":        oracle_irm["mean_ood_acc"],
            "irm_id_val":        val_irm["mean_ood_acc"],
            "irm_agreement":     agr_ood,
            "gap_agr_vs_oracle": gap_agr_vs_oracle,
            "gap_val_vs_oracle": gap_val_vs_oracle,
        },
        "erm_configs": [strip(c) for c in erm_configs],
        "irm_configs": [strip(c) for c in irm_configs],
        "erm_erm_pairs": erm_erm_points,
    }
    json_path = os.path.join(output_dir, f"agl_erm_vs_irm_{env_config}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON → {json_path}\n")
    return result


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config",  type=str, default="diverse",
                        choices=["original", "diverse", "proximate"])
    parser.add_argument("--all_configs", action="store_true",
                        help="Run all three env configs sequentially")
    parser.add_argument("--n_trials",    type=int, default=20,
                        help="Random HP configs to sample (like n_trials in search.py)")
    parser.add_argument("--n_seeds",     type=int, default=3,
                        help="Seeds per HP config (more = better AGL line)")
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--output_dir",  type=str, default="agl_results")
    args = parser.parse_args()

    mnist_raw = load_mnist_raw()
    configs   = (["original", "diverse", "proximate"]
                 if args.all_configs else [args.env_config])

    for env_cfg in configs:
        run(env_cfg, args.n_trials, args.n_seeds,
            args.device, args.output_dir, mnist_raw)

    print("Done.")