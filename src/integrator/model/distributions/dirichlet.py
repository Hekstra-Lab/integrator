from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class DirichletDistribution(torch.nn.Module):
    """

    Attributes:
        n_pixels:
        eps:
    """

    def __init__(
        self,
        in_features: int = 64,
        sbox_shape: tuple[int, ...] = (3, 21, 21),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_pixels = prod(sbox_shape)
        if in_features is not None:
            self.alpha_layer = nn.Linear(in_features, self.n_pixels)
        self.eps = eps

    def forward(self, x, group_labels=None):
        x = self.alpha_layer(x)

        x = F.softplus(x) + self.eps
        q = Dirichlet(x)
        return q
