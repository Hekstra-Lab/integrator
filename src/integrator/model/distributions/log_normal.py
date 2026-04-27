import torch.nn as nn
from torch import Tensor
from torch.distributions import LogNormal

from .utils import get_positive_constraint


class LogNormalDistribution(nn.Module):
    """LogNormal distribution surrogate.
    `loc` (mu in log-space) is unconstrained.
    `scale` (sigma in log-space) is constrained positive.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-3,
        positive_constraint: str = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self._constrain = get_positive_constraint(positive_constraint)
        self.linear_loc = nn.Linear(in_features, 1)
        self.linear_scale = nn.Linear(in_features, 1)

    def forward(self, x: Tensor, x_: Tensor) -> LogNormal:
        loc = self.linear_loc(x).squeeze()
        scale = self._constrain(self.linear_scale(x_)).squeeze() + self.eps
        return LogNormal(loc=loc, scale=scale)
