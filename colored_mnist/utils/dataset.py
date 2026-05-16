# utils/dataset.py
# Core environment builder for Colored MNIST.

import torch
import numpy as np
from torchvision import datasets


def make_environment(images, labels, e, device):
    """
    Build one Colored MNIST environment.

    Args:
        images:  uint8 tensor [N, 28, 28]
        labels:  long tensor [N]
        e:       float — probability of flipping the color assignment
                 e=0.1 → color 90% correlated with label (train)
                 e=0.2 → color 80% correlated with label (train)
                 e=0.5 → color random, no correlation
                 e=0.9 → color 90% anti-correlated (test)
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

    # 2x subsample: 28x28 -> 14x14
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]

    # Binary label: digit < 5 -> 1, else 0
    # Flip with prob 0.25 to add label noise (theoretical ceiling ~75%)
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))

    # Assign color based on label, flip with prob e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

    # Build 2-channel image: channel 0 = red, channel 1 = green
    # Zero out the channel not corresponding to the assigned color
    images = torch.stack([images, images], dim=1)
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0

    return {
        "images": (images.float() / 255.0).to(device),
        "labels": labels[:, None].to(device),
    }



def load_mnist():
    """
    Load MNIST and return train and test pools.
    Shuffles train pool keeping images and labels aligned.

    Returns:
        train_pool: tuple (images, labels): 50,000 samples
        test_pool:  tuple (images, labels): 10,000 samples
    """
    mnist      = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_pool = (mnist.data[:50000], mnist.targets[:50000])
    test_pool  = (mnist.data[50000:],  mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(train_pool[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(train_pool[1].numpy())

    return train_pool, test_pool