import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Linear


class LogNormalDistribution(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.fc = Linear(
            in_features=in_features,
            out_features=2,
            bias=False,
        )
        self.eps = eps

    def forward(
        self,
        x: Tensor,
    ) -> LogNormal:
        raw_loc, raw_scale = self.fc(x).chunk(2, dim=-1)
        loc = torch.exp(raw_loc) + self.eps
        scale = F.softplus(raw_scale) + self.eps
        lognormal = LogNormal(loc=loc.squeeze(), scale=scale.squeeze())

        return lognormal
