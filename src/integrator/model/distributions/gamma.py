import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class GammaDistribution(BaseDistribution[Gamma]):
    fc: nn.Module
    """`Linear` layer to map input tensors to distribution parameters"""

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        """
        Args:
            in_features: Dimension of input Tensor
            out_features: Dimension of the networks parameter Tensor
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            beta=beta,
        )

        self.fc = Linear(
            in_features=in_features,
            out_features=out_features,
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
        concentration = self._constrain_fn(params[..., 0])
        rate = self._constrain_fn(params[..., 1])
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
