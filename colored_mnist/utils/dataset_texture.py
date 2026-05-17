# utils/dataset_texture.py
# Environment builder for Colored MNIST with two correlated spurious features.
#
# Spurious feature 1: color: digit pixels colored red/green, corr=e with label
# Spurious feature 2: texture: background stripe pattern, corr=e with label
#
# Both spurious features use the same correlation parameter e, so they always vary together across environments.
#
# Causal feature: digit shape 

import torch


def make_environment_texture(images, labels, e, device):
    """
    Build one Colored MNIST environment with color + texture spurious features.

    Args:
        images:  uint8 tensor [N, 28, 28]
        labels:  long tensor [N]
        e:       float — correlation strength for BOTH color and texture
                 e=0.1 → both 90% correlated with label
                 e=0.9 → both 90% anti-correlated with label (test)
        device:  torch device

    Returns:
        dict with keys:
            'images' — float tensor [N, 2, 14, 14] normalized to [0, 1]
            'labels' — float tensor [N, 1]
    """
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()

    images = images.reshape((-1, 28, 28))[:, ::2, ::2]   # (N, 14, 14)
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))

    # Assign color based on label, flip with prob e 
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

    # Assign texture pattern based on label, flip with prob e 
    textures = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Normalize
    images_float = images.float() / 255.0   # (N, 14, 14)

    background_mask = (images == 0).float()   # (N, 14, 14)
    images_2ch = torch.stack([images_float, images_float], dim=1)  # (N, 2, 14, 14)
    images_2ch[torch.arange(len(images_2ch)), (1 - colors).long(), :, :] *= 0

    # Apply stripe texture to background pixels 
    H, W = 14, 14
    stripe_A = torch.zeros(H, W)
    stripe_A[0::2, :] = 1.0   # even rows

    stripe_B = torch.zeros(H, W)
    stripe_B[1::2, :] = 1.0   # odd rows

    pattern = torch.where(
        textures.bool().view(-1, 1, 1),
        stripe_B.unsqueeze(0).expand(len(images_float), -1, -1),
        stripe_A.unsqueeze(0).expand(len(images_float), -1, -1),
    )   # (N, 14, 14)

    # Apply texture only to background pixels with intensity 0.5
    texture_overlay = pattern * background_mask * 0.5   # (N, 14, 14)
    # Add texture to BOTH channels equally 
    images_2ch[:, 0, :, :] = (images_2ch[:, 0, :, :] + texture_overlay).clamp(0, 1)
    images_2ch[:, 1, :, :] = (images_2ch[:, 1, :, :] + texture_overlay).clamp(0, 1)

    return {
        "images": images_2ch.to(device),
        "labels": labels[:, None].to(device),
    }
