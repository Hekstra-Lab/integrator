import math
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
        estimand: Literal["background", "intensity"],
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
        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 6e5  # mean in [~0, 600k]
            self.r_min, self.r_max = 0.5, 10.0  # Fano in [0.1, 2.0]
        elif estimand == "background":
            self.mu_min, selfmu_max = 1e-3, 100.0  # mean in [~0, 100]
            self.r_min, self.r_max = 0.5, 10.0

        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_r_min = math.log(self.r_min)
        self.log_r_max = math.log(self.r_max)

    def smooth_bound(self, x, a, b):
        return a + (b - a) * (torch.atan(x) / torch.pi + 0.5)

    def smooth_bound_square(self, x, a, b):
        t = (2 / torch.pi) * torch.atan(x)  # (-1,1)
        t = 0.5 * (t + 1.0)  # (0,1)
        return a + (b - a) * (t**2)  # square for large-range stability

    # def forward(self, x) -> Gamma:
    #     raw_k, raw_r = self.fc(x).chunk(2, dim=-1)
    #     print("mean raw k", raw_k.mean())
    #     print("min raw k", raw_k.min())
    #     print("max raw k", raw_k.max())
    #     print("mean raw r", raw_r.mean())
    #     print("min raw r", raw_r.min())
    #     print("max raw r", raw_r.max())
    #
    #     k = torch.nn.functional.softplus(raw_k) + 0.01
    #     r = torch.nn.functional.softplus(raw_r) + 0.01
    #
    #     print("qbg,", self.fc.bias)
    #
    #     print("mean k", k.mean())
    #     print("min k", k.min())
    #     print("max k", k.max())
    #     print("mean r", r.mean())
    #     print("min r", r.min())
    #     print("max r", r.max())
    #
    #     return Gamma(concentration=k.flatten(), rate=r.flatten())

    def forward(self, x) -> Gamma:
        raw_m, raw_r = self.fc(x).chunk(2, dim=-1)

        # bound log-mean and log-rate
        log_mu = self.smooth_bound_square(
            raw_m, self.log_mu_min, self.log_mu_max
        )
        log_r = self.smooth_bound_square(raw_r, self.log_r_min, self.log_r_max)

        mu = torch.exp(log_mu)  # mean
        r = torch.exp(log_r)  # rate

        # shape
        k = mu * r

        # OPTIONAL: debug prints
        # print("mu stats:", mu.min(), mu.max(), mu.mean())
        # print("r stats:",  r.min(),  r.max(),  r.mean())
        # print("fano stats:", (1.0 / r).min(), (1.0 / r).max())

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
