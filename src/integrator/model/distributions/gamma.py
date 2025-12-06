from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator.layers import Constrain


class GammaDistribution(nn.Module):
    fc: nn.Module
    """`Linear` layer to map input tensors to distribution parameters"""

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        eps: float = 1e-4,
        beta: int = 1,
        constraint: Literal["exp", "softplus"] | None = "softplus",
    ):
        """
        Args:
            in_features: Dimension of input Tensor
            out_features: Dimension of the networks parameter Tensor
        """
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Gamma:
        """

        Args:
            x: Input batch of shoeboxes
        Returns:
            `torch.distributions.Gamma`

        """
        params = self.fc(x)
        concentration = self.constrain_fn(params[..., 0])
        rate = self.constrain_fn(params[..., 1])
        return Gamma(concentration.flatten(), rate.flatten())


if __name__ == "__main__":
    # Example usage
    in_features = 64
    gamma_dist = GammaDistribution(in_features)
    representation = torch.randn(10, in_features)  # Example input
    metarep = torch.randn(
        10, in_features * 2
    )  # Example metadata representation

    # use without metadata
    qbg = gamma_dist(representation)

    # use with metadata
    qbg = gamma_dist(representation)
