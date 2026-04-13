import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import LogNormal


class LogNormalDistribution(nn.Module):
    """LogNormal distribution surrogate.
    `loc` (mu in log-space) is unconstrained.
    `scale` (sigma in log-space) is constrained positive via softplus.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-3,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_loc = nn.Linear(in_features, 1)
            self.linear_scale = nn.Linear(in_features, 1)
        else:
            self.fc = nn.Linear(in_features, 2)

    def forward(
        self,
        x: Tensor,
        x_: Tensor | None = None,
    ) -> LogNormal:
        if self.separate_inputs:
            raw_loc = self.linear_loc(x)
            raw_scale = self.linear_scale(x_ if x_ is not None else x)
        else:
            raw_loc, raw_scale = self.fc(x).chunk(2, dim=-1)

        loc = raw_loc.squeeze()
        scale = F.softplus(raw_scale).squeeze() + self.eps
        return LogNormal(loc=loc, scale=scale)
