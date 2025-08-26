from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Dirichlet

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class DirichletDistribution(BaseDistribution[Dirichlet]):
    """
    Dirichlet distribution with parameters predicted by a linear layer.
    """

    input_shape: tuple[int, ...]
    """Shape of an input shoebox as ``(C, H, W)`` or ``(H, W)``."""

    def __init__(
        self,
        in_features: int = 64,
        out_features: tuple[int, ...] = (3, 21, 21),
        **kwargs,
    ):
        """
        Args:
            in_features: Input feature dimension.
            out_features: ``(C, H, W)`` or ``(H, W)`` used to calculate dimension of the Diritchlet concentration paramter.
        """
        super().__init__(
            eps=1e-6,
            in_features=in_features,
            out_features=out_features,
            **kwargs,
        )

        self.num_components = 0

        if len(out_features) == 3:
            self.num_components = (
                out_features[0] * out_features[1] * out_features[2]
            )
        elif len(out_features) == 2:
            self.num_components = out_features[0] * out_features[1]

        if self.out_features is not None:
            self.alpha_layer = Linear(
                self.in_features,
                self.num_components,
            )

    def forward(
        self,
        x: Tensor,
    ) -> Dirichlet:
        """
        Return a `torch.distributions.Dirichlet` from an input shoebox

        Args:
            x: Input batch of shoeboxes

        Returns:
            qp: A `torch.distributions.Dirichlet(x)`

        """
        x = self.alpha_layer(x)
        x = F.softplus(x) + self.eps
        qp = Dirichlet(x)

        return qp


if __name__ == "__main__":
    x = torch.rand(10, 64)
    dirichlet = DirichletDistribution(in_features=64, input_shape=(21, 21))

    # DirichletDistribution
    qp = dirichlet(x)

    # 3D Case
    dirichlet = DirichletDistribution(in_features=64, input_shape=(3, 21, 21))
    qp = dirichlet(x)

    # DirichletDistribution
    dirichlet = DirichletDistribution(in_features=64, input_shape=(21, 21))
    q = dirichlet(x)
