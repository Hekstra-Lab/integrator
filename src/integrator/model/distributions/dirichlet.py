import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Dirichlet

from integrator.layers import Constrain, Linear


class DirichletDistribution(nn.Module):
    """
    Dirichlet distribution with parameters predicted by a linear layer.
    """

    input_shape: tuple[int, ...]
    """Shape of an input shoebox as ``(C, H, W)`` or ``(H, W)``."""

    def __init__(
        self,
        in_features: int = 64,
        out_features: tuple[int, ...] = (3, 21, 21),
        constraint: Literal["exp", "softplus"] | None = "softplus",
        eps: float = 0.01,
        beta: int = 1,
    ):
        """
        Args:
            in_features: Input feature dimension.
            out_features: ``(C, H, W)`` or ``(H, W)`` used to calculate dimension of the Diritchlet concentration paramter.
        """
        super().__init__()

        if len(out_features) == 3:
            self.num_components = (
                out_features[0] * out_features[1] * out_features[2]
            )
        elif len(out_features) == 2:
            self.num_components = out_features[0] * out_features[1]

        if out_features is not None:
            self.alpha_layer = Linear(
                in_features,
                self.num_components,
            )

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )
        self.beta = beta
        self.eps = eps
        self.min_log_alpha = math.log(1e-3)
        self.max_log_alpha = math.log(1e3)

    def forward(self, x: Tensor) -> Dirichlet:
        """
        Return a `torch.distributions.Dirichlet` from an input shoebox

        Args:
            x: Input batch of shoeboxes

        Returns:
            qp: A `torch.distributions.Dirichlet(x)`

        """
        # x = self.alpha_layer(x)
        # x = self.constrain_fn(x) + self.eps

        if torch.isnan(x).any():
            raise RuntimeError("NaNs in Dirichlet input x")

        log_alpha = self.alpha_layer(x)

        if torch.isnan(log_alpha).any():
            raise RuntimeError("NaNs right after Dirichlet fc")

        log_alpha = torch.clamp(
            log_alpha, self.min_log_alpha, self.max_log_alpha
        )
        alpha = torch.exp(log_alpha)

        if torch.isnan(alpha).any() or (alpha <= 0).any():
            raise RuntimeError(
                "NaNs or nonpositive alpha before constructing Dirichlet"
            )
        # keep log_alpha within [-4, 4]
        # log_alpha = torch.tanh(log_alpha) * 4.0
        # alpha = torch.exp(log_alpha)
        qp = Dirichlet(alpha)

        return qp


if __name__ == "__main__":
    pass
