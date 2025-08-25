from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class GammaDistribution(BaseDistribution[Gamma]):
    fc: nn.Module
    """Linear layer to map input tensor to distribution parameters"""

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
    ):
        """

        Args:
            in_features:
            out_features:
        """
        super().__init__(in_features=in_features)

        self.fc = Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def distribution(
        self,
        concentration: Tensor,
        rate: Tensor,
    ) -> Gamma:
        """

        Args:
            concentration:
            rate:

        Returns:

        """
        concentration = self.constraint(concentration)
        rate = self.constraint(rate)
        return Gamma(concentration.flatten(), rate.flatten())

    def forward(
        self,
        x: Tensor,
    ) -> Gamma:
        """

        Args:
            x: Input batch of shoeboxes
        Returns:

        """
        params = self.fc(x)
        gamma = self.distribution(params[..., 0], params[..., 1])

        return gamma


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
