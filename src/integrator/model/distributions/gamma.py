import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from integrator.layers import Constrain


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, 2, bias=True)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


class GammaDistribution(nn.Module):
    fc: nn.Module
    """`Linear` layer to map input tensors to distribution parameters"""

    def __init__(
        self,
        estimand: Literal["background", "intensity"],
        in_features: int,
        out_features: int = 2,
        eps: float = 1e-2,
        beta: int = 1,
        constraint: Literal["exp", "softplus"] | None = "softplus",
    ):
        """
        Args:
            in_features: Dimension of input Tensor
            out_features: Dimension of the networks parameter Tensor
        """
        super().__init__()

        # self.fc = Linear(
        #     in_features=in_features,
        #     out_features=out_features,
        #     bias=True,
        # )

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )
        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 6e5  # mean in [~0, 600k]
            self.r_min, self.r_max = 0.2, 50.0  # Fano in [0.1, 2.0]
            self.estimand = estimand
        elif estimand == "background":
            self.mu_min, self.mu_max = 1e-3, 100.0  # mean in [~0, 100]
            self.r_min, self.r_max = 0.2, 10.0
            self.estimand = estimand

        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_r_min = math.log(self.r_min)
        self.log_r_max = math.log(self.r_max)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.mlp = MLP(in_dim=in_features)

    def forward(self, x) -> Gamma:
        raw_mu, raw_r = self.mlp(x).chunk(2, dim=-1)

        print(f"\n{self.estimand} stats:")
        print("mean raw_mu", raw_mu.mean())
        print("min raw_mu", raw_mu.min())
        print("max raw_mu", raw_mu.max())
        print("mean raw r", raw_r.mean())
        print("min raw r", raw_r.min())
        print("max raw r", raw_r.max())

        mu = F.softplus(raw_mu) + 0.001
        r = F.softplus(raw_r) + 0.0001
        k = mu * r

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
