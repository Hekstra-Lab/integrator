from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.half_normal import HalfNormal

from integrator.layers import Constrain, Linear


class HalfNormalDistribution(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-12,
        beta: int = 1,
        constraint: Literal["exp", "softplus"] | None = "softplus",
    ):
        super().__init__()
        self.fc = Linear(
            in_features=in_features,
            out_features=1,
        )

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )

    def forward(
        self,
        x: Tensor,
    ) -> HalfNormal:
        scale = self.fc(x)
        scale = self.constrain_fn(scale)
        return HalfNormal(scale=scale)


if __name__ == "__main__":
    # Example usage

    in_features = 64
    half_normal_dist = HalfNormalDistribution(in_features, out_features=1)

    representation = torch.randn(10, in_features)
    qbg = half_normal_dist(representation)
