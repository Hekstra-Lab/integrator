import math
from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Gamma

from integrator.layers import Linear


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_features: int = 2,
    ):
        super().__init__()
        self.fc1 = Linear(
            in_dim,
            hidden,
            bias=False,
        )
        self.act = nn.SiLU()
        self.fc2 = Linear(
            hidden,
            out_features,
            bias=False,
        )

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


class GammaDistribution(nn.Module):
    """
    Gamma posterior parameterized by (mean, fano_per_image).

    - Network predicts mean μ(x)
    - Fano φ comes from an embedding indexed by image_id
    """

    def __init__(
        self,
        estimand: Literal["background", "intensity"],
        in_features: int,
        n_images: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 1e6
            self.fano_min, self.fano_max = 0.2, 2.0
        else:
            self.mu_min, self.mu_max = 1e-3, 100.0
            self.fano_min, self.fano_max = 0.2, 5.0

        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_fano_min = math.log(self.fano_min)
        self.log_fano_max = math.log(self.fano_max)

        self.mlp = MLP(in_dim=in_features, out_features=1)

        self.log_phi_table = nn.Embedding(
            num_embeddings=n_images,
            embedding_dim=1,
            sparse=False,
        )
        nn.init.constant_(self.log_phi_table.weight, 0.0)  # start φ=1

        self.eps = eps

    def _bound(self, raw, log_min, log_max):
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))

    def forward(self, x, img_ids):
        """
        x:      (batch, features)
        img_ids:(batch,) integer indices 0...n_images-1
        """
        raw_mu = self.mlp(x)
        mu = self._bound(raw_mu, self.log_mu_min, self.log_mu_max)  # (B,1)

        log_phi_img = self.log_phi_table(img_ids[:, 2].int())

        phi = torch.exp(log_phi_img)
        phi = torch.clamp(phi, self.fano_min, self.fano_max)

        beta = 1.0 / (phi + self.eps)
        alpha = mu * beta

        dist = Gamma(concentration=alpha.squeeze(-1), rate=beta.squeeze(-1))
        return dist


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
#         self.mlp = MLP(in_dim=in_features, out_features=out_features)
#
#     def forward(self, x):
#         raw_k, raw_r = self.mlp(x).chunk(2, dim=-1)
#
#         k = F.softplus(raw_k) + 0.0001
#         r = F.softplus(raw_r) + 0.0001
#
#         fano = 1 / r
#
#         lambda_corr = 1.0
#
#         log_fano = torch.log(1.0 / r + 1e-8)
#         log_mean = torch.log((k / r) + 1e-8)
#
#         log_fano_c = log_fano - log_fano.mean()
#         log_mean_c = log_mean - log_mean.mean()
#
#         corr = (log_fano_c * log_mean_c).mean()
#
#         shape_penalty = lambda_corr * corr**2
#
#         return (
#             Gamma(concentration=k.flatten(), rate=r.flatten()),
#             fano,
#             shape_penalty,
#         )
#         # return Gamma(concentration=k.flatten(), rate=r.flatten()), r.flatten()
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
