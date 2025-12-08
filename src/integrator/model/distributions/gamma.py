from typing import Literal

import torch
import torch.nn as nn
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

        # bounds
        self.k_min = 0.1
        self.k_max = 1000000.0
        self.r_min = 0.1
        self.r_max = 5.0

    def smooth_bound(self, x, a, b):
        return a + (b - a) * (torch.atan(x) / torch.pi + 0.5)

    def smooth_bound_square(self, x, a, b):
        t = (2 / torch.pi) * torch.atan(x)  # (-1,1)
        t = 0.5 * (t + 1.0)  # (0,1)
        return a + (b - a) * (t**2)  # square for large-range stability

    # def forward(
    #     self,
    #     x: Tensor,
    # ) -> Gamma:
    #     """
    #
    #     Args:
    #         x: Input batch of shoeboxes
    #     Returns:
    #         `torch.distributions.Gamma`
    #
    #     """
    #     params = self.fc(x)
    #     concentration = self.constrain_fn(params[..., 0])
    #     rate = self.constrain_fn(params[..., 1])
    #     return Gamma(concentration.flatten(), rate.flatten())
    #

    def forward(self, x) -> Gamma:
        raw_k, raw_r = self.fc(x).chunk(2, dim=-1)
        print("mean raw k", raw_k.mean())
        print("min raw k", raw_k.min())
        print("max raw k", raw_k.max())
        print("mean raw r", raw_r.mean())
        print("min raw r", raw_r.min())
        print("max raw r", raw_r.max())

        # # shape = slow-saturating large-range mapping
        # k = self.smooth_bound_square(raw_k, self.k_min, self.k_max)
        #
        # # rate = simpler mapping because range is small
        # r = self.smooth_bound(raw_r, self.r_min, self.r_max)

        k = torch.nn.functional.softplus(raw_k) + 0.1
        r = torch.nn.functional.softplus(raw_r) + 0.1

        print("qbg,", self.fc.bias)

        print("mean k", k.mean())
        print("min k", k.min())
        print("max k", k.max())
        print("mean r", r.mean())
        print("min r", r.min())
        print("max r", r.max())

        return Gamma(concentration=k.flatten(), rate=r.flatten())


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
