"""
Agreement-on-the-Line Diagnostic for IRM vs ERM
=================================================

Runs the full AGL diagnostic:
  1. Trains multiple ERM and IRM models with different data subsets and lambda values
  2. Computes pairwise ID and OOD agreement for ERM-ERM and IRM-ERM pairs
  3. Fits the AGL line from ERM-ERM pairs
  4. Checks whether IRM-ERM pairs fall below the line
  5. Saves results and plots

Usage:
    python agl_diagnostic.py --env_config diverse --n_seeds 3 --device cpu

Env configs (from your Experiment 5):
    original   -> e_train = [0.1, 0.2]   (low diversity, low proximity)
    diverse    -> e_train = [0.1, 0.5]   (high diversity, low proximity)
    proximate  -> e_train = [0.7, 0.8]   (low diversity, high proximity)
"""

import argparse
import itertools
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colored MNIST data — MNIST loaded ONCE, passed as raw tensors everywhere
# ---------------------------------------------------------------------------

def load_mnist_raw():
    """Load raw MNIST tensors once. Returns (train_images, train_labels, val_images, val_labels)."""
    from torchvision import datasets
    print("Loading MNIST (once)...")
    mnist = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_images = mnist.data[:50000]   # (50000, 28, 28)
    train_labels = mnist.targets[:50000]
    val_images   = mnist.data[50000:]   # (10000, 28, 28)
    val_labels   = mnist.targets[50000:]
    print(f"  train: {train_images.shape}  val: {val_images.shape}")
    return train_images, train_labels, val_images, val_labels


def make_environment(images, labels, e, seed=0):
    """Build a colored environment from raw image/label tensors."""
    rng = np.random.RandomState(seed)
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
    return {
        "images": imgs.view(imgs.shape[0], -1),
        "labels": labels.unsqueeze(1),
    }


def build_envs(train_images, train_labels, e_values, data_fraction, seed):
    """Subsample train data and build colored environments — no disk I/O."""
    rng = np.random.RandomState(seed)
    n_total = len(train_images)
    if data_fraction < 1.0:
        n = int(n_total * data_fraction)
        idx = rng.choice(n_total, n, replace=False)
        imgs = train_images[idx]
        lbls = train_labels[idx]
    else:
        imgs, lbls = train_images, train_labels

    n_per_env = len(imgs) // len(e_values)
    envs = []
    for i, e in enumerate(e_values):
        start = i * n_per_env
        envs.append(make_environment(imgs[start:start + n_per_env],
                                     lbls[start:start + n_per_env],
                                     e, seed=seed + i))
    return envs


# ---------------------------------------------------------------------------
# MLP (matches your architecture)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def irm_penalty(logits, labels):
    """IRMv1 penalty: gradient norm of a fixed scalar classifier."""
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def train_model(
    envs: List[Dict],
    method: str = "erm",
    hidden_dim: int = 256,
    lr: float = 1e-3,
    l2_reg: float = 1e-3,
    steps: int = 501,
    penalty_weight: float = 1e4,
    penalty_anneal_iters: int = 100,
    device: str = "cpu",
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    model = MLP(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(steps):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        penalty_total = torch.tensor(0.0, device=device)

        for env in envs:
            x = env["images"].to(device)
            y = env["labels"].to(device)
            logits = model(x)
            env_loss = loss_fn(logits, y)
            total_loss += env_loss
            if method == "irm":
                penalty_total += irm_penalty(logits, y)

        total_loss /= len(envs)
        penalty_total /= len(envs)

        if method == "irm":
            w = penalty_weight if step >= penalty_anneal_iters else 1.0
            loss = total_loss + w * penalty_total
            # Normalize like original paper
            if w > 1:
                loss /= w
        else:
            loss = total_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model: nn.Module, data: Dict, device: str) -> torch.Tensor:
    model.eval()
    x = data["images"].to(device)
    logits = model(x)
    return (logits > 0).float().cpu()


def compute_agreement(preds1: torch.Tensor, preds2: torch.Tensor) -> float:
    return (preds1 == preds2).float().mean().item()


def compute_accuracy(model: nn.Module, data: Dict, device: str) -> float:
    preds = get_predictions(model, data, device)
    labels = data["labels"].cpu()
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Probit transform (for AGL line fitting)
# ---------------------------------------------------------------------------

def probit(p: float, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(stats.norm.ppf(p))


def inv_probit(z: float) -> float:
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    env_config: str = "diverse"
    e_test: float = 0.9
    n_seeds: int = 3
    lambda_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    data_fractions: List[float] = field(default_factory=lambda: [0.7, 0.85, 1.0])
    hidden_dim: int = 256
    lr: float = 1e-3
    l2_reg: float = 1e-3
    steps: int = 501
    penalty_anneal_iters: int = 100
    device: str = "cpu"
    output_dir: str = "agl_results"

    @property
    def e_train(self) -> List[float]:
        configs = {
            "original":  [0.1, 0.2],
            "diverse":   [0.1, 0.5],
            "proximate": [0.7, 0.8],
        }
        return configs[self.env_config]


def run_experiment(cfg: ExperimentConfig, mnist_raw=None):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AGL Diagnostic — env_config={cfg.env_config}, "
          f"e_train={cfg.e_train}, e_test={cfg.e_test}")
    print(f"n_seeds={cfg.n_seeds}  fractions={cfg.data_fractions}  "
          f"lambdas={cfg.lambda_values}  steps={cfg.steps}")
    print(f"Total models: {cfg.n_seeds * len(cfg.data_fractions)} ERM + "
          f"{cfg.n_seeds * len(cfg.data_fractions) * len(cfg.lambda_values)} IRM")
    print(f"{'='*60}\n")

    # Load MNIST once for the whole experiment
    if mnist_raw is None:
        mnist_raw = load_mnist_raw()
    train_images, train_labels, val_images, val_labels = mnist_raw

    # Fixed reference environments for agreement computation (no subsetting)
    ref_envs = [make_environment(
                    train_images[i * (len(train_images) // len(cfg.e_train)):
                                 (i+1) * (len(train_images) // len(cfg.e_train))],
                    train_labels[i * (len(train_labels) // len(cfg.e_train)):
                                 (i+1) * (len(train_labels) // len(cfg.e_train))],
                    e, seed=i)
                for i, e in enumerate(cfg.e_train)]
    ref_test = make_environment(val_images, val_labels, cfg.e_test, seed=99)
    id_ref = ref_envs[0]  # fixed ID reference for agreement

    # -----------------------------------------------------------------------
    # Step 1: Train all models
    # -----------------------------------------------------------------------

    erm_models = []
    irm_models = []

    combo_id = 0
    for seed_idx in range(cfg.n_seeds):
        for frac in cfg.data_fractions:
            seed = seed_idx * 100 + int(frac * 100)

            # Build environments from already-loaded MNIST tensors
            envs = build_envs(train_images, train_labels,
                              cfg.e_train, frac, seed)
            test_env = make_environment(val_images, val_labels,
                                        cfg.e_test, seed=seed + 99)

            # ERM
            print(f"[{combo_id:03d}] ERM   seed={seed} frac={frac:.2f}", end="  ")
            erm_model = train_model(
                envs, method="erm",
                hidden_dim=cfg.hidden_dim, lr=cfg.lr, l2_reg=cfg.l2_reg,
                steps=cfg.steps, device=cfg.device, seed=seed
            )
            id_acc  = float(np.mean([compute_accuracy(erm_model, e, cfg.device) for e in envs]))
            ood_acc = compute_accuracy(erm_model, test_env, cfg.device)
            erm_models.append({"model": erm_model, "id_acc": id_acc,
                                "ood_acc": ood_acc, "seed": seed, "frac": frac})
            print(f"id={id_acc:.3f}  ood={ood_acc:.3f}")

            # IRM with each lambda
            for lam in cfg.lambda_values:
                print(f"[{combo_id:03d}] IRM   seed={seed} frac={frac:.2f} λ={lam}", end="  ")
                irm_model = train_model(
                    envs, method="irm",
                    hidden_dim=cfg.hidden_dim, lr=cfg.lr, l2_reg=cfg.l2_reg,
                    steps=cfg.steps, penalty_weight=lam,
                    penalty_anneal_iters=cfg.penalty_anneal_iters,
                    device=cfg.device, seed=seed
                )
                id_acc_irm  = float(np.mean([compute_accuracy(irm_model, e, cfg.device) for e in envs]))
                ood_acc_irm = compute_accuracy(irm_model, test_env, cfg.device)
                irm_models.append({
                    "model": irm_model, "id_acc": id_acc_irm,
                    "ood_acc": ood_acc_irm, "lambda": lam,
                    "seed": seed, "frac": frac,
                    "erm_ref_idx": len(erm_models) - 1
                })
                print(f"id={id_acc_irm:.3f}  ood={ood_acc_irm:.3f}")

            combo_id += 1

    print(f"\nTrained {len(erm_models)} ERM + {len(irm_models)} IRM models\n")

    # -----------------------------------------------------------------------
    # Step 2: Compute pairwise agreement
    # (id_ref and ref_test are already set above from the single MNIST load)
    # -----------------------------------------------------------------------

    erm_erm_points = []   # (id_agr, ood_agr)
    irm_erm_points = []   # (id_agr, ood_agr, lambda)

    print("Computing ERM-ERM agreement pairs...")
    erm_pairs = list(itertools.combinations(range(len(erm_models)), 2))
    for i, j in erm_pairs:
        preds_i_id  = get_predictions(erm_models[i]["model"], id_ref, cfg.device)
        preds_j_id  = get_predictions(erm_models[j]["model"], id_ref, cfg.device)
        preds_i_ood = get_predictions(erm_models[i]["model"], ref_test, cfg.device)
        preds_j_ood = get_predictions(erm_models[j]["model"], ref_test, cfg.device)

        id_agr  = compute_agreement(preds_i_id, preds_j_id)
        ood_agr = compute_agreement(preds_i_ood, preds_j_ood)
        erm_erm_points.append({"id_agr": id_agr, "ood_agr": ood_agr})

    print(f"  {len(erm_erm_points)} ERM-ERM pairs computed")

    print("Computing IRM-ERM agreement pairs...")
    for irm_entry in irm_models:
        irm_model = irm_entry["model"]
        erm_model = erm_models[irm_entry["erm_ref_idx"]]["model"]

        preds_irm_id  = get_predictions(irm_model, id_ref, cfg.device)
        preds_erm_id  = get_predictions(erm_model, id_ref, cfg.device)
        preds_irm_ood = get_predictions(irm_model, ref_test, cfg.device)
        preds_erm_ood = get_predictions(erm_model, ref_test, cfg.device)

        id_agr  = compute_agreement(preds_irm_id, preds_erm_id)
        ood_agr = compute_agreement(preds_irm_ood, preds_erm_ood)
        irm_erm_points.append({
            "id_agr": id_agr, "ood_agr": ood_agr,
            "lambda": irm_entry["lambda"],
            "irm_ood_acc": irm_entry["ood_acc"],
            "erm_ood_acc": erm_models[irm_entry["erm_ref_idx"]]["ood_acc"],
        })

    print(f"  {len(irm_erm_points)} IRM-ERM pairs computed")

    # -----------------------------------------------------------------------
    # Step 3: Fit AGL line from ERM-ERM pairs (in probit space)
    # -----------------------------------------------------------------------

    erm_id_probit  = np.array([probit(p["id_agr"])  for p in erm_erm_points])
    erm_ood_probit = np.array([probit(p["ood_agr"]) for p in erm_erm_points])

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        erm_id_probit, erm_ood_probit
    )
    r2 = r_value ** 2

    print(f"\nAGL line (ERM-ERM, probit space):")
    print(f"  slope={slope:.3f}  intercept={intercept:.3f}  R²={r2:.3f}")

    # -----------------------------------------------------------------------
    # Step 4: Measure IRM-ERM deviation from line
    # -----------------------------------------------------------------------

    results = []
    for pt in irm_erm_points:
        id_p  = probit(pt["id_agr"])
        ood_p = probit(pt["ood_agr"])
        predicted_ood_p = slope * id_p + intercept
        deviation = ood_p - predicted_ood_p  # negative = below line
        results.append({
            "lambda": pt["lambda"],
            "id_agr": pt["id_agr"],
            "ood_agr": pt["ood_agr"],
            "deviation": deviation,
            "irm_ood_acc": pt["irm_ood_acc"],
            "erm_ood_acc": pt["erm_ood_acc"],
            "irm_advantage": pt["irm_ood_acc"] - pt["erm_ood_acc"],
        })

    # -----------------------------------------------------------------------
    # Step 5: Print summary table
    # -----------------------------------------------------------------------

    print(f"\n{'─'*70}")
    print(f"{'λ':>8} {'ID agr':>8} {'OOD agr':>9} {'Deviation':>11} "
          f"{'IRM acc':>9} {'ERM acc':>9} {'Advantage':>10}")
    print(f"{'─'*70}")
    for r in sorted(results, key=lambda x: x["lambda"]):
        flag = " ← BELOW" if r["deviation"] < -0.05 else ""
        print(f"{r['lambda']:>8.1f} {r['id_agr']:>8.3f} {r['ood_agr']:>9.3f} "
              f"{r['deviation']:>+11.3f} {r['irm_ood_acc']:>9.3f} "
              f"{r['erm_ood_acc']:>9.3f} {r['irm_advantage']:>+10.3f}{flag}")
    print(f"{'─'*70}")

    # Correlation: does deviation predict IRM advantage?
    deviations  = [r["deviation"]    for r in results]
    advantages  = [r["irm_advantage"] for r in results]
    if len(deviations) > 2:
        corr, pval = stats.pearsonr(deviations, advantages)
        print(f"\nCorrelation(deviation, IRM advantage): r={corr:.3f}  p={pval:.4f}")
        print("(Negative r supports H1: more below the line → bigger IRM advantage)")

    # -----------------------------------------------------------------------
    # Step 6: Plot
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: AGL scatter in raw agreement space ---
    ax = axes[0]
    erm_id  = [p["id_agr"]  for p in erm_erm_points]
    erm_ood = [p["ood_agr"] for p in erm_erm_points]
    ax.scatter(erm_id, erm_ood, color="#185FA5", alpha=0.6, s=40,
               label="ERM-ERM pairs", zorder=3)

    # AGL line in raw space (approximate)
    x_range = np.linspace(min(erm_id) - 0.02, max(erm_id) + 0.02, 100)
    y_line  = [inv_probit(slope * probit(x) + intercept) for x in x_range]
    ax.plot(x_range, y_line, color="#185FA5", linewidth=1.5,
            linestyle="--", label="AGL line (ERM-ERM)", zorder=2)

    # IRM-ERM points colored by lambda
    lambda_vals = sorted(set(r["lambda"] for r in results))
    colors_irm  = ["#D85A30", "#1D9E75", "#7F77DD"]
    for lam, col in zip(lambda_vals, colors_irm):
        pts = [r for r in results if r["lambda"] == lam]
        ax.scatter(
            [p["id_agr"] for p in pts],
            [p["ood_agr"] for p in pts],
            color=col, alpha=0.8, s=60, marker="^",
            label=f"IRM-ERM λ={lam}", zorder=4
        )

    ax.set_xlabel("ID agreement", fontsize=12)
    ax.set_ylabel("OOD agreement", fontsize=12)
    ax.set_title(f"Agreement-on-the-Line\n"
                 f"env={cfg.env_config}  R²={r2:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Deviation vs IRM advantage scatter ---
    ax2 = axes[1]
    for lam, col in zip(lambda_vals, colors_irm):
        pts = [r for r in results if r["lambda"] == lam]
        ax2.scatter(
            [p["deviation"] for p in pts],
            [p["irm_advantage"] for p in pts],
            color=col, alpha=0.8, s=60, marker="^",
            label=f"λ={lam}", zorder=3
        )

    ax2.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax2.axhline(0, color="gray", linewidth=1, linestyle="--")

    if len(deviations) > 2:
        m, b = np.polyfit(deviations, advantages, 1)
        xfit = np.linspace(min(deviations) - 0.05, max(deviations) + 0.05, 100)
        ax2.plot(xfit, m * xfit + b, color="#888780", linewidth=1.5,
                 linestyle="-", alpha=0.7, label=f"fit  r={corr:.2f}")

    ax2.set_xlabel("Deviation from AGL line (probit)\n← below line   |   above line →",
                   fontsize=11)
    ax2.set_ylabel("IRM accuracy − ERM accuracy\n(IRM advantage)", fontsize=11)
    ax2.set_title(f"Does deviation predict IRM utility?\n"
                  f"env={cfg.env_config}", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(cfg.output_dir, f"agl_{cfg.env_config}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    # -----------------------------------------------------------------------
    # Step 7: Save raw results
    # -----------------------------------------------------------------------

    output = {
        "config": asdict(cfg),
        "agl_line": {"slope": slope, "intercept": intercept, "r2": r2},
        "erm_erm_pairs": erm_erm_points,
        "irm_erm_results": results,
    }
    if len(deviations) > 2:
        output["deviation_advantage_correlation"] = {"r": corr, "p": pval}

    json_path = os.path.join(cfg.output_dir, f"agl_{cfg.env_config}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {json_path}\n")

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", type=str, default="diverse",
                        choices=["original", "diverse", "proximate"],
                        help="Which environment configuration from Experiment 5")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random seeds per (method, fraction) combo")
    parser.add_argument("--steps", type=int, default=501,
                        help="Training steps per model")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-3)
    parser.add_argument("--penalty_anneal_iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="agl_results")
    parser.add_argument("--all_configs", action="store_true",
                        help="Run all three env configs sequentially")
    args = parser.parse_args()

    # Load MNIST once — shared across all configs if --all_configs
    mnist_raw = load_mnist_raw()

    if args.all_configs:
        for env_cfg in ["original", "diverse", "proximate"]:
            cfg = ExperimentConfig(
                env_config=env_cfg,
                n_seeds=args.n_seeds,
                steps=args.steps,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                l2_reg=args.l2_reg,
                penalty_anneal_iters=args.penalty_anneal_iters,
                device=args.device,
                output_dir=args.output_dir,
            )
            run_experiment(cfg, mnist_raw=mnist_raw)
    else:
        cfg = ExperimentConfig(
            env_config=args.env_config,
            n_seeds=args.n_seeds,
            steps=args.steps,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            l2_reg=args.l2_reg,
            penalty_anneal_iters=args.penalty_anneal_iters,
            device=args.device,
            output_dir=args.output_dir,
        )
        run_experiment(cfg, mnist_raw=mnist_raw)