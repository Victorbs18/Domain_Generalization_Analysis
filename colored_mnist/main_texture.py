# main_texture.py
# Experiment 6 — Two correlated spurious variables (color + texture)
#
# Extends Colored MNIST with a second spurious feature: stripe texture
# applied to background pixels. Both color and texture share the same
# correlation parameter e, so they vary together across environments.
#
# Causal feature:     digit shape (invariant across environments)
# Spurious feature 1: color (red=0, green=1, corr=e with label)
# Spurious feature 2: background stripe pattern (corr=e with label)
#
# Test environment: e=0.9 (both color and texture are misleading)
#
# Research question: Does IRM's ability to identify the invariant feature
# (shape) degrade when two spurious correlations are present simultaneously
# and vary together across environments?
#
# Expected results vs original single-spurious experiment:
#   ERM:          similar failure (~17%) — still exploits both shortcuts
#   IRM oracle:   moderate degradation (~55-62%) — harder optimization
#   IRM realistic: still collapses (~10%) — degenerate configs even more
#                  attractive with two spurious signals

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import optim

from main import MLP, mean_nll, mean_accuracy, irm_penalty


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "original": {
        "e_values": [0.1, 0.2],
        "description": "Original 2-env setup, both spurious features"
    },
    "diverse": {
        "e_values": [0.1, 0.5],
        "description": "High diversity, both spurious features"
    },
    "proximate": {
        "e_values": [0.7, 0.8],
        "description": "High proximity, both spurious features"
    },
}

TEST_E = 0.9


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------

def make_stripe_mask(image_shape, pattern):
    """
    Create a binary stripe mask for background pixels.

    pattern=0: even rows = 1, odd rows = 0  (stripes type A)
    pattern=1: odd rows = 1, even rows = 0  (stripes type B)

    Args:
        image_shape: (H, W) of the image
        pattern: 0 or 1 — which stripe orientation to use

    Returns:
        mask: (H, W) binary tensor
    """
    H, W = image_shape
    mask = torch.zeros(H, W)
    if pattern == 0:
        mask[0::2, :] = 1.0  # even rows
    else:
        mask[1::2, :] = 1.0  # odd rows
    return mask


# ---------------------------------------------------------------------------
# Environment creation with two spurious features
# ---------------------------------------------------------------------------

def make_environment_texture(images, labels, e, device):
    """
    Create a Colored MNIST environment with TWO spurious features:
      1. Color  — digit pixels colored red/green correlated with label
      2. Texture — background pixels show stripe pattern correlated with label

    Both spurious features use the same correlation parameter e.

    A low e (e.g. 0.1) means both color and texture are ANTI-correlated
    with the label — the model must rely on shape to do well.

    A high e (e.g. 0.9) means both color and texture are strongly
    correlated with the label — easy shortcuts for the model to exploit.

    Args:
        images:  (N, 28, 28) uint8 MNIST images
        labels:  (N,) int labels 0-9
        e:       float, spurious correlation strength for BOTH features
        device:  torch device

    Returns:
        dict with 'images' (N, 2, 28, 28) and 'labels' (N,) on device
    """
    # --- Step 1: Binarize labels (0-4 → 0, 5-9 → 1) ---
    labels = (labels < 5).float()

    # --- Step 2: Flip labels with 25% probability (inject noise) ---
    labels = torch.logical_xor(
        labels.bool(),
        torch.bernoulli(0.25 * torch.ones(*labels.shape)).bool()
    ).float()

    # --- Step 3: Assign color based on label, flipped with prob e ---
    # color_label=1 → green digit, color_label=0 → red digit
    color_label = torch.logical_xor(
        labels.bool(),
        torch.bernoulli(e * torch.ones(*labels.shape)).bool()
    ).float()

    # --- Step 4: Assign texture pattern based on label, flipped with prob e ---
    # texture_label=1 → stripe pattern B, texture_label=0 → stripe pattern A
    texture_label = torch.logical_xor(
        labels.bool(),
        torch.bernoulli(e * torch.ones(*labels.shape)).bool()
    ).float()

    # --- Step 5: Color the digit pixels ---
    # Compute background mask BEFORE normalization (uint8, exact zeros)
    background_mask = (images == 0).float().to(device)  # (N, 28, 28)

    # Now normalize to [0, 1]
    images = images.float().to(device) / 255.0

    # Red channel: digit pixels colored red if color_label=0
    # Green channel: digit pixels colored green if color_label=1
    color_label = color_label.to(device)
    red_channel   = images * (1 - color_label).view(-1, 1, 1)
    green_channel = images * color_label.view(-1, 1, 1)

    # --- Step 6: Apply stripe texture to background pixels ---
    H, W = images.shape[1], images.shape[2]
    stripe_A = make_stripe_mask((H, W), pattern=0).to(device)
    stripe_B = make_stripe_mask((H, W), pattern=1).to(device)

    texture_label = texture_label.to(device)
    texture_pattern = torch.where(
        texture_label.bool().view(-1, 1, 1),
        stripe_B.unsqueeze(0).expand(len(images), -1, -1),
        stripe_A.unsqueeze(0).expand(len(images), -1, -1)
    )  # (N, 28, 28)

    # Apply texture only to background pixels
    texture_overlay = texture_pattern * background_mask * 0.5

    # Add texture to both channels equally (texture is grayscale)
    red_channel   = (red_channel   + texture_overlay).clamp(0, 1)
    green_channel = (green_channel + texture_overlay).clamp(0, 1)

    # --- Step 7: Stack into 2-channel image and downsample to 14x14 ---
    # Original make_environment downsamples to 14x14 to match MLP input size
    images_out = torch.stack([red_channel, green_channel], dim=1)
    # shape: (N, 2, 28, 28)

    # Downsample to 14x14 using average pooling — same as original
    images_out = torch.nn.functional.avg_pool2d(images_out, kernel_size=2)
    # shape: (N, 2, 14, 14)

    return {
        "images": images_out.to(device),
        "labels": labels.to(device)
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Colored MNIST — Two correlated spurious features")
    parser.add_argument("--hidden_dim",            type=int,   default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr",                    type=float, default=0.001)
    parser.add_argument("--n_restarts",            type=int,   default=10)
    parser.add_argument("--penalty_anneal_iters",  type=int,   default=100)
    parser.add_argument("--penalty_weight",        type=float, default=10000.0)
    parser.add_argument("--steps",                 type=int,   default=501)
    parser.add_argument("--grayscale_model",       action="store_true")
    parser.add_argument("--seed",                  type=int,   default=0)
    parser.add_argument("--config",                type=str,   default="original",
                        choices=["original", "diverse", "proximate"])
    parser.add_argument("--selection_method",      type=str,   default="oracle",
                        choices=["oracle", "train_domain_val",
                                 "leave_one_domain_out"])
    parser.add_argument("--mode",                  type=str,   default="irm",
                        choices=["irm", "erm"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_environments(config, selection_method, device, val_fraction=0.2):
    """
    Load Colored MNIST with texture, two training environments.
    """
    cfg      = ENV_CONFIGS[config]
    e_values = cfg["e_values"]

    mnist      = datasets.MNIST("~/datasets/mnist", train=True,  download=True)
    mnist_test = datasets.MNIST("~/datasets/mnist", train=False, download=True)

    train_pool = (mnist.data[:50000], mnist.targets[:50000])
    test_pool  = (mnist_test.data,    mnist_test.targets)

    n_total    = len(train_pool[0])
    split_size = n_total // len(e_values)

    train_envs = []
    val_envs   = []

    for i, e in enumerate(e_values):
        start = i * split_size
        end   = start + split_size if i < len(e_values) - 1 else n_total

        env_images = train_pool[0][start:end]
        env_labels = train_pool[1][start:end]

        if selection_method == "train_domain_val":
            n_val   = int(len(env_images) * val_fraction)
            n_train = len(env_images) - n_val
            train_envs.append(make_environment_texture(
                env_images[:n_train], env_labels[:n_train], e, device))
            val_envs.append(make_environment_texture(
                env_images[n_train:], env_labels[n_train:], e, device))
        else:
            train_envs.append(make_environment_texture(
                env_images, env_labels, e, device))
            val_envs.append(make_environment_texture(
                env_images, env_labels, e, device))

    test_env = make_environment_texture(
        test_pool[0], test_pool[1], TEST_E, device)

    return {
        "train": train_envs,
        "val":   val_envs,
        "test":  test_env,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_restart(flags, data, device, verbose=True):
    """
    Train IRM or ERM on data['train'], evaluate on data['val'] and data['test'].
    """
    model     = MLP(flags.hidden_dim, flags.grayscale_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)
    train_envs = data["train"]

    if verbose:
        cols = ["step", "train_nll", "train_acc", "train_penalty", "test_acc"]
        print("   ".join(c.ljust(13) for c in cols))

    for step in range(flags.steps):
        for env in train_envs:
            logits         = model(env["images"]).squeeze()
            env["nll"]     = mean_nll(logits, env["labels"])
            env["acc"]     = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll     = torch.stack([e["nll"]     for e in train_envs]).mean()
        train_acc     = torch.stack([e["acc"]     for e in train_envs]).mean()
        train_penalty = torch.stack([e["penalty"] for e in train_envs]).mean()

        weight_norm = torch.tensor(0.0, device=device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        if flags.mode == "irm":
            penalty_weight = (
                flags.penalty_weight
                if step >= flags.penalty_anneal_iters else 1.0
            )
            loss  = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= penalty_weight
        else:
            # ERM — no penalty
            loss  = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            def fmt(v):
                return np.array2string(
                    v, precision=5, floatmode="fixed").ljust(13)
            with torch.no_grad():
                test_logits = model(data["test"]["images"])
                test_acc    = mean_accuracy(
                    test_logits, data["test"]["labels"])
            print("   ".join([
                str(np.int32(step)).ljust(13),
                fmt(train_nll.detach().cpu().numpy()),
                fmt(train_acc.detach().cpu().numpy()),
                fmt(train_penalty.detach().cpu().numpy()),
                fmt(test_acc.detach().cpu().numpy()),
            ]))

    # Final evaluation
    with torch.no_grad():
        for env in train_envs + data["val"]:
            logits     = model(env["images"]).squeeze()
            env["acc"] = mean_accuracy(logits, env["labels"])
        test_logits         = model(data["test"]["images"]).squeeze()
        data["test"]["acc"] = mean_accuracy(
            test_logits, data["test"]["labels"])

    train_accs = [e["acc"].detach().cpu().item() for e in train_envs]
    val_accs   = [e["acc"].detach().cpu().item() for e in data["val"]]
    test_acc   = data["test"]["acc"].detach().cpu().item()

    return {
        "train_accs": train_accs,
        "val_accs":   val_accs,
        "test_acc":   test_acc,
        "mean_train": float(np.mean(train_accs)),
        "mean_val":   float(np.mean(val_accs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = get_args()

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ENV_CONFIGS[flags.config]
    print(f"Using device: {device}")
    print(f"Config: {flags.config}  |  "
          f"e values: {cfg['e_values']}  |  "
          f"Description: {cfg['description']}")
    print(f"Mode: {flags.mode}  |  "
          f"Selection method: {flags.selection_method}\n")

    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print(f"\t{k}: {v}")
    print()

    data = load_environments(flags.config, flags.selection_method, device)

    final_test_accs = []

    for restart in range(flags.n_restarts):
        print(f"\n--- Restart {restart} ---")
        results = train_one_restart(flags, data, device, verbose=True)
        final_test_accs.append(results["test_acc"])
        print(f"  Train accs: {[f'{a:.4f}' for a in results['train_accs']]}")
        print(f"  Val   accs: {[f'{a:.4f}' for a in results['val_accs']]}")
        print(f"  Test  acc:  {results['test_acc']:.4f}")
        print(f"  Test so far: "
              f"{np.mean(final_test_accs):.4f} ± "
              f"{np.std(final_test_accs):.4f}")

    print("\n=== Final results ===")
    print(f"Config: {flags.config}  |  e values: {cfg['e_values']}")
    print(f"Mode: {flags.mode}  |  Selection: {flags.selection_method}")
    print(f"Test acc (mean ± std): "
          f"{np.mean(final_test_accs):.4f} ± "
          f"{np.std(final_test_accs):.4f}")


if __name__ == "__main__":
    main()