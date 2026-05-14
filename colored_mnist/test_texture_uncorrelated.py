import torch
from torchvision import datasets
from main_texture_uncorrelated import make_environment_texture

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
images, labels = mnist.data[:1000], mnist.targets[:1000]
device = torch.device('cpu')

# Test e=0.1 environment
env_low = make_environment_texture(images, labels, e=0.1, device=device)
# Test e=0.9 environment  
env_high = make_environment_texture(images, labels, e=0.9, device=device)

print("=== Sanity check: independent spurious features ===")
print(f"e=0.1 env: color weakly correlated, texture strongly correlated (1-e=0.9)")
print(f"e=0.9 env: color strongly correlated, texture weakly correlated (1-e=0.1)")
print(f"Image shape: {env_low['images'].shape}")
print(f"Label distribution e=0.1: {env_low['labels'].mean().item():.3f}")
print(f"Label distribution e=0.9: {env_high['labels'].mean().item():.3f}")
print("SUCCESS")