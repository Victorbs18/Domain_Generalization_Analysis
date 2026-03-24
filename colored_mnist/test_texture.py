import torch
from torchvision import datasets
from main_texture import make_environment_texture

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
images, labels = mnist.data[:100], mnist.targets[:100]
device = torch.device('cpu')

env = make_environment_texture(images, labels, e=0.1, device=device)
print('Image shape:', env['images'].shape)
print('Labels shape:', env['labels'].shape)
print('Image min/max:', env['images'].min().item(), env['images'].max().item())
print('Label distribution:', env['labels'].mean().item())

# Check that background pixels have texture
# Background pixels in original image are 0
original = images[0].float() / 255.0
background_mask = (original == 0)
textured_img = env['images'][0]  # shape (2, 28, 28)

# Sum both channels to get total intensity
total = textured_img[0] + textured_img[1]
bg_values = total[background_mask]
print('Background pixel values (should be > 0 for texture):', bg_values.unique()[:5])
print('Digit pixel values (should be > 0 only on digit):', total[~background_mask].mean().item())
print('SUCCESS')