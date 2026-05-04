from math import prod

import torch.nn as nn
from torch.distributions import Dirichlet

from .utils import get_positive_constraint


class DirichletDistribution(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        sbox_shape: tuple[int, ...] = (3, 21, 21),
        eps: float = 1e-6,
        positive_constraint: str = "softplus",
    ):
        super().__init__()
        self.n_pixels = prod(sbox_shape)
        self.alpha_layer = nn.Linear(in_features, self.n_pixels)
        self.eps = eps
        self._constrain = get_positive_constraint(positive_constraint)

    def forward(self, x, mc_samples=None, group_labels=None, **kwargs):
        alpha = self._constrain(self.alpha_layer(x)) + self.eps
        return Dirichlet(alpha)
