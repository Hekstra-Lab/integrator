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
        input_shape: tuple[int, ...] = (3, 21, 21),
    ):
        """
        Args:
            in_features: Input feature dimension.
            input_shape: ``(C, H, W)`` or ``(H, W)`` used to derive ``num_components``.
        """
        super().__init__(
            eps=1e-6,
            in_features=in_features,
        )

        self.input_shape = input_shape

        if len(input_shape) == 3:
            self.num_components = (
                input_shape[0] * input_shape[1] * input_shape[2]
            )
        elif len(input_shape) == 2:
            self.num_components = input_shape[0] * input_shape[1]

        if self.in_features is not None:
            self.alpha_layer = Linear(in_features, self.num_components)

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

    qp = dirichlet(x)

    # 3D Case

    dirichlet = DirichletDistribution(in_features=64, input_shape=(3, 21, 21))
    qp = dirichlet(x)

    dirichlet = DirichletDistribution(in_features=64, input_shape=(21, 21))
    q = dirichlet(x)
