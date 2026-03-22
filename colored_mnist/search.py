"""
search.py — Random hyperparameter search for IRM on Colored MNIST.

Replicates the search ranges described in the original paper:

    hidden_dim            = int(2 ** U(6, 9))
    l2_regularizer_weight = 10 ** U(-5, -2)
    lr                    = 10 ** U(-3.5, -2.5)
    penalty_anneal_iters  = randint(50, 250)
    penalty_weight        = 10 ** U(2, 6)
    steps                 in {101, 201, 301, 401, 501}

Selection criterion (same as paper):
    maximize  min(train_env0_acc, train_env1_acc, test_acc)

Usage
-----
    # IRM search
    python search.py --mode irm --n_trials 50

    # Grayscale oracle search
    python search.py --mode grayscale --n_trials 50

    # Plain ERM — no search needed
    python search.py --mode erm
"""

import argparse
import json
import time
import numpy as np
import torch
from types import SimpleNamespace

from main import load_colored_mnist, MLP, mean_nll, mean_accuracy, irm_penalty


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

STEPS_GRID = [101, 201, 301, 401, 501]


def sample_irm(rng: np.random.Generator) -> dict:
    return {
        "hidden_dim":            int(2 ** rng.uniform(6, 9)),
        "l2_regularizer_weight": float(10 ** rng.uniform(-5, -2)),
        "lr":                    float(10 ** rng.uniform(-3.5, -2.5)),
        "penalty_anneal_iters":  int(rng.integers(50, 250)),
        "penalty_weight":        float(10 ** rng.uniform(2, 6)),
        "steps":                 int(rng.choice(STEPS_GRID)),
        "grayscale_model":       False,
    }


def sample_grayscale(rng: np.random.Generator) -> dict:
    return {
        "hidden_dim":            int(2 ** rng.uniform(6, 9)),
        "l2_regularizer_weight": float(10 ** rng.uniform(-5, -2)),
        "lr":                    float(10 ** rng.uniform(-3.5, -2.5)),
        "penalty_anneal_iters":  0,
        "penalty_weight":        0.0,
        "steps":                 int(rng.choice(STEPS_GRID)),
        "grayscale_model":       True,
    }


ERM_FIXED = {
    "hidden_dim":            256,
    "l2_regularizer_weight": 0.001,
    "lr":                    0.001,
    "penalty_anneal_iters":  0,
    "penalty_weight":        0.0,
    "steps":                 501,
    "grayscale_model":       False,
}


# ---------------------------------------------------------------------------
# Selection criterion
# ---------------------------------------------------------------------------

def selection_score(env0_acc: float, env1_acc: float, test_acc: float) -> float:
    """Paper criterion: maximise min(train_env0_acc, train_env1_acc, test_acc)."""
    return min(env0_acc, env1_acc, test_acc)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_hparams(hparams: dict, device: torch.device, seed: int = 0):
    """
    Train once and return per-environment accuracies.
    Returns (env0_acc, env1_acc, test_acc).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    flags = SimpleNamespace(**hparams)
    envs  = load_colored_mnist(device)

    model     = MLP(flags.hidden_dim, flags.grayscale_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=flags.lr)

    for step in range(flags.steps):
        for env in envs:
            logits         = model(env["images"])
            env["nll"]     = mean_nll(logits, env["labels"])
            env["acc"]     = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll     = torch.stack([envs[0]["nll"],     envs[1]["nll"]]).mean()
        train_penalty = torch.stack([envs[0]["penalty"], envs[1]["penalty"]]).mean()

        weight_norm = torch.tensor(0.0, device=device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        penalty_weight = (
            flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
        )
        loss  = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env0_acc = envs[0]["acc"].detach().cpu().item()
    env1_acc = envs[1]["acc"].detach().cpu().item()
    test_acc = envs[2]["acc"].detach().cpu().item()
    return env0_acc, env1_acc, test_acc


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  Trials: {args.n_trials}  |  Seed: {args.seed}\n")

    rng = np.random.default_rng(args.seed)

    best_score   = -np.inf
    best_hparams = None
    results      = []

    if args.mode == "erm":
        candidates = [ERM_FIXED]
    else:
        sampler    = sample_irm if args.mode == "irm" else sample_grayscale
        candidates = [sampler(rng) for _ in range(args.n_trials)]

    for trial_idx, hparams in enumerate(candidates):
        t0 = time.time()
        e0, e1, te = evaluate_hparams(hparams, device, seed=args.seed + trial_idx)
        score      = selection_score(e0, e1, te)
        elapsed    = time.time() - t0

        result = {
            "trial":    trial_idx,
            "score":    score,
            "env0_acc": e0,
            "env1_acc": e1,
            "test_acc": te,
            "elapsed":  elapsed,
            **hparams,
        }
        results.append(result)

        marker = " ← best" if score > best_score else ""
        print(
            f"[{trial_idx:3d}]  score={score:.4f}  "
            f"e0={e0:.4f}  e1={e1:.4f}  te={te:.4f}  "
            f"({elapsed:.1f}s){marker}"
        )

        if score > best_score:
            best_score   = score
            best_hparams = hparams

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Best score ({args.mode}): {best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in sorted(best_hparams.items()):
        print(f"    {k}: {v}")

    # Save to JSON
    out_path = f"search_results_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump({"best": best_hparams, "all_trials": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print equivalent CLI command
    cmd = (
        f"python main.py \\\n"
        f"  --hidden_dim={best_hparams['hidden_dim']} \\\n"
        f"  --l2_regularizer_weight={best_hparams['l2_regularizer_weight']} \\\n"
        f"  --lr={best_hparams['lr']} \\\n"
        f"  --penalty_anneal_iters={best_hparams['penalty_anneal_iters']} \\\n"
        f"  --penalty_weight={best_hparams['penalty_weight']} \\\n"
        f"  --steps={best_hparams['steps']}"
    )
    if best_hparams.get("grayscale_model"):
        cmd += " \\\n  --grayscale_model"
    print("\nBest command:\n")
    print(cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for IRM")
    parser.add_argument("--mode",     choices=["irm", "grayscale", "erm"], default="irm")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run_search(args)
