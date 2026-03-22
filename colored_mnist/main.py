# Reproduction of Invariant Risk Minimization (IRM)
# Arjovsky et al., 2019 — https://arxiv.org/abs/1907.02893
#
# Colored MNIST experiment.
# Hyperparameters are NOT copied from the paper; they are found via search.py.

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Colored MNIST — IRM reproduction")
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr",                    type=float, default=0.001)
    parser.add_argument("--n_restarts",            type=int,   default=10)
    parser.add_argument("--penalty_anneal_iters",  type=int,   default=100)
    parser.add_argument("--penalty_weight",        type=float, default=10000.0)
    parser.add_argument("--steps",                 type=int,   default=501)
    parser.add_argument("--grayscale_model",       action="store_true",
                        help="Use grayscale model (oracle / invariant-by-construction)")
    parser.add_argument("--seed",                  type=int,   default=0,
                        help="Global random seed for reproducibility")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_environment(images, labels, e, device):
    """
    Build one Colored-MNIST environment.

    Args:
        images:  uint8 tensor [N, 28, 28]
        labels:  long tensor [N]
        e:       float — probability of flipping the color assignment
        device:  torch device

    Returns:
        dict with keys 'images' [N, 2, 14, 14] and 'labels' [N, 1]

    Environment construction:
        - Binary label: digit < 5 → 1, else 0; flipped with prob 0.25
        - Color assigned based on label, then flipped with prob e
        - e=0.2 / e=0.1 → color strongly correlated with label (train)
        - e=0.9 → color anti-correlated with label (test)
    """
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # both inputs are in {0, 1}

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]

    # Binary label: digit < 5 → 1; flip with prob 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))

    # Color based on label, flipped with env-specific prob e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

    # Stack into 2-channel image and zero out the "wrong" channel
    images = torch.stack([images, images], dim=1)
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0

    return {
        "images": (images.float() / 255.0).to(device),
        "labels": labels[:, None].to(device),
    }


def load_colored_mnist(device):
    """
    Download MNIST and return three environments:
        env[0]: train split A, e=0.2  (strong spurious correlation)
        env[1]: train split B, e=0.1  (stronger spurious correlation)
        env[2]: val / test,    e=0.9  (color is anti-correlated → hard)
    """
    mnist = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_data = (mnist.data[:50000], mnist.targets[:50000])
    val_data   = (mnist.data[50000:], mnist.targets[50000:])

    # Shuffle train set (keep images <-> labels aligned)
    rng_state = np.random.get_state()
    np.random.shuffle(train_data[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_data[1].numpy())

    envs = [
        make_environment(train_data[0][::2],  train_data[1][::2],  0.2, device),
        make_environment(train_data[0][1::2], train_data[1][1::2], 0.1, device),
        make_environment(val_data[0],         val_data[1],         0.9, device),
    ]
    return envs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple 3-layer MLP for Colored MNIST."""

    def __init__(self, hidden_dim: int, grayscale: bool):
        super().__init__()
        in_dim = 14 * 14 if grayscale else 2 * 14 * 14
        lin1 = nn.Linear(in_dim,     hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main     = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        self.grayscale = grayscale

    def forward(self, x):
        if self.grayscale:
            # Merge the two channels by summing — model cannot see color
            out = x.view(x.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = x.view(x.shape[0], 2 * 14 * 14)
        return self._main(out)


# ---------------------------------------------------------------------------
# Loss / penalty helpers
# ---------------------------------------------------------------------------

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def irm_penalty(logits, y):
    """
    IRM penalty: ||∇_{scale=1} R_e(w ∘ scale)||²
    Forces the gradient of the loss w.r.t. a dummy scalar to be zero,
    which encourages an invariant predictor across environments.
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss  = mean_nll(logits * scale, y)
    grad  = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# ---------------------------------------------------------------------------
# Training loop (one restart)
# ---------------------------------------------------------------------------

def train_one_restart(flags, envs, device, verbose=True):
    """
    Run one full training run.
    Returns (final_train_acc, final_test_acc).
    """
    model     = MLP(flags.hidden_dim, flags.grayscale_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)

    def _header():
        cols = ["step", "train_nll", "train_acc", "train_penalty", "test_acc"]
        print("   ".join(c.ljust(13) for c in cols))

    def _row(*values):
        def fmt(v):
            if isinstance(v, str):
                return v.ljust(13)
            return np.array2string(v, precision=5, floatmode="fixed").ljust(13)
        print("   ".join(fmt(v) for v in values))

    if verbose:
        _header()

    for step in range(flags.steps):
        for env in envs:
            logits         = model(env["images"])
            env["nll"]     = mean_nll(logits, env["labels"])
            env["acc"]     = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll     = torch.stack([envs[0]["nll"],     envs[1]["nll"]]).mean()
        train_acc     = torch.stack([envs[0]["acc"],     envs[1]["acc"]]).mean()
        train_penalty = torch.stack([envs[0]["penalty"], envs[1]["penalty"]]).mean()

        # L2 regularisation
        weight_norm = torch.tensor(0.0, device=device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        # Penalty annealing: start with weight=1, then jump to penalty_weight
        penalty_weight = (
            flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
        )
        loss  = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            _row(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                envs[2]["acc"].detach().cpu().numpy(),
            )

    final_train_acc = train_acc.detach().cpu().item()
    final_test_acc  = envs[2]["acc"].detach().cpu().item()
    return final_train_acc, final_test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = get_args()

    # Reproducibility
    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    envs = load_colored_mnist(device)

    final_train_accs, final_test_accs = [], []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        tr_acc, te_acc = train_one_restart(flags, envs, device, verbose=True)
        final_train_accs.append(tr_acc)
        final_test_accs.append(te_acc)
        print(f"  Train acc so far: {np.mean(final_train_accs):.4f} ± {np.std(final_train_accs):.4f}")
        print(f"  Test  acc so far: {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")

    print("\n=== Final results ===")
    print(f"Train acc (mean ± std): {np.mean(final_train_accs):.4f} ± {np.std(final_train_accs):.4f}")
    print(f"Test  acc (mean ± std): {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()
