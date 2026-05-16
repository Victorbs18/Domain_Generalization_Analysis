# utils/model.py
# MLP model for Colored MNIST.

from torch import nn


class MLP(nn.Module):
    """
    3-layer MLP for Colored MNIST binary classification.

    Input:
        - Color model:     2 channels x 14 x 14 = 392 features
        - Grayscale model: sums both channels    = 196 features
    Output:
        - 1 logit (binary classification)
    """

    def __init__(self, hidden_dim: int, grayscale: bool = False):
        super().__init__()
        in_dim = 14 * 14 if grayscale else 2 * 14 * 14
        lin1   = nn.Linear(in_dim,     hidden_dim)
        lin2   = nn.Linear(hidden_dim, hidden_dim)
        lin3   = nn.Linear(hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main     = nn.Sequential(lin1, nn.ReLU(True),
                                       lin2, nn.ReLU(True), lin3)
        self.grayscale = grayscale

    def forward(self, x):
        if self.grayscale:
            # Sum both channels
            out = x.view(x.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = x.view(x.shape[0], 2 * 14 * 14)
        return self._main(out)