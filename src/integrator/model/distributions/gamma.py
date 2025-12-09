import torch
import torch.nn as nn
from torch.distributions import Gamma

from integrator.layers import Linear


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.fc1 = Linear(
            in_dim,
            hidden,
            bias=False,
        )
        self.act = nn.SiLU()
        self.fc2 = Linear(
            hidden,
            2,
            bias=False,
        )

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


class GammaDistribution(nn.Module):
    def __init__(
        self,
        estimand: str,  # "intensity" or "background"
        in_features: int,
        hidden_features: int = 64,  # for your MLP
        out_features: int = 2,
        eps_mu: float = 1e-8,
        eps_fano: float = 1e-8,
        # Fano scheduling hyperparameters:
        F_small: float = 100.0,  # max Fano at tiny μ
        F_large: float = 4.0,  # max Fano at large μ
        mu_transition: float = 100.0,  # μ scale where F_small → F_large
        constraint: str = "softplus",
    ):
        super().__init__()

        self.estimand = estimand

        # Tiny MLP used as head
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(
                hidden_features, out_features
            ),  # outputs raw_log_mu, raw_fano
        )

        # hyperparameters for Fano schedule
        self.F_small = F_small
        self.F_large = F_large
        self.mu_transition = mu_transition

        # numerical epsilons
        self.eps_mu = eps_mu
        self.eps_fano = eps_fano

    def forward(self, x) -> Gamma:
        raw_log_mu, raw_fano = self.mlp(x).chunk(2, dim=-1)

        mu = torch.exp(raw_log_mu) + self.eps_mu

        Fano_max = self.F_large + (self.F_small - self.F_large) * torch.exp(
            -mu / self.mu_transition
        )

        s = torch.sigmoid(raw_fano)
        Fano = s * Fano_max + self.eps_fano

        r = 1.0 / Fano
        k = mu * r

        # remove event dim
        k = k.squeeze(-1)
        r = r.squeeze(-1)

        return Gamma(concentration=k, rate=r)


# class GammaDistribution(nn.Module):
#     fc: nn.Module
#     """`Linear` layer to map input tensors to distribution parameters"""
#
#     def __init__(
#         self,
#         estimand: Literal["background", "intensity"],
#         in_features: int,
#         out_features: int = 2,
#         eps: float = 1e-2,
#         beta: int = 1,
#         constraint: Literal["exp", "softplus"] | None = "softplus",
#     ):
#         """
#         Args:
#             in_features: Dimension of input Tensor
#             out_features: Dimension of the networks parameter Tensor
#         """
#         super().__init__()
#
#         # self.fc = Linear(
#         #     in_features=in_features,
#         #     out_features=out_features,
#         #     bias=True,
#         # )
#
#         self.constrain_fn = Constrain(
#             constraint_fn=constraint,
#             eps=eps,
#             beta=beta,
#         )
#         if estimand == "intensity":
#             self.mu_min, self.mu_max = 1e-3, 6e5  # mean in [~0, 600k]
#             self.r_min, self.r_max = 0.2, 50.0  # Fano in [0.1, 2.0]
#             self.estimand = estimand
#         elif estimand == "background":
#             self.mu_min, self.mu_max = 1e-3, 100.0  # mean in [~0, 100]
#             self.r_min, self.r_max = 0.2, 10.0
#             self.estimand = estimand
#
#         self.log_mu_min = math.log(self.mu_min)
#         self.log_mu_max = math.log(self.mu_max)
#         self.log_r_min = math.log(self.r_min)
#         self.log_r_max = math.log(self.r_max)
#         self.register_buffer("eps", torch.tensor(eps))
#         self.register_buffer("beta", torch.tensor(beta))
#         self.mlp = MLP(in_dim=in_features)
#
#     def forward(self, x) -> Gamma:
#         raw_k, raw_r = self.mlp(x).chunk(2, dim=-1)
#
#         print(f"\n{self.estimand} stats:")
#         print("mean raw_k", raw_k.mean())
#         print("min raw_k", raw_k.min())
#         print("max raw_k", raw_k.max())
#         print("mean raw r", raw_r.mean())
#         print("min raw r", raw_r.min())
#         print("max raw r", raw_r.max())
#
#         k = F.softplus(raw_k) + 0.001
#         r = F.softplus(raw_r) + 0.0001
#
#         print("mean k", k.mean())
#         print("min k", k.min())
#         print("max k", k.max())
#         print("mean r", r.mean())
#         print("min r", r.min())
#         print("max r", r.max())
#
#         return Gamma(concentration=k.flatten(), rate=r.flatten())
#
#
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
