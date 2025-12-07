from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Constrain, Linear


class LogNormalDistribution(nn.Module):
    """
    LogNormal distribution with parameters predicted by a linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        eps: float = 1e-12,
        beta: int = 1,
        bias: bool = False,
        constraint: Literal["exp", "softplus"] | None = "softplus",
    ):
        super().__init__()

        self.fc = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )

    def forward(
        self,
        x: Tensor,
    ) -> LogNormal:
        raw_loc, raw_scale = self.fc(x).chunk(2, dim=-1)
        loc = torch.tanh(raw_loc) * 12
        scale = F.softplus(raw_scale)
        lognormal = LogNormal(loc=loc.squeeze(), scale=scale.squeeze())

        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    lognormal = LogNormalDistribution(in_features=64, constraint="softplus")
