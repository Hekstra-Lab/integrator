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


def mean_pool_by_image(emb: torch.Tensor, img_ids: torch.Tensor):
    device = emb.device
    B, F = emb.shape

    # 1. Get unique image IDs and mapping
    pooled_ids, per_ref_idx = torch.unique(img_ids, return_inverse=True)
    n_img = pooled_ids.size(0)

    # 2. Sum per image using index_add_
    sums = torch.zeros(n_img, F, device=device)
    counts = torch.zeros(n_img, 1, device=device)

    sums.index_add_(0, per_ref_idx, emb)  # sum embeddings per image
    ones = torch.ones(B, 1, device=device)
    counts.index_add_(0, per_ref_idx, ones)  # count per image

    pooled = sums / counts.clamp_min(1.0)  # mean embedding per image
    return pooled, pooled_ids, per_ref_idx


class GammaDistribution(nn.Module):
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

        # self.log_phi_table = nn.Parameter(torch.zeros(n_images))  # (n_images,)

        self.linear_alpha = torch.nn.Linear(in_features, 1)
        self.linear_beta = torch.nn.Linear(in_features, 1)

    def _bound(self, raw, log_min, log_max):
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))

    def forward(self, x, img_ids):
        """
        x: (batch, features)
        img_ids:(batch,) integer indices 0...n_images-1
        """

        raw_alpha = self.linear_alpha(x)
        pooled, pooled_ids, per_ref_idx = mean_pool_by_image(
            x, img_ids[:, 2].long()
        )

        raw_r = self.linear_beta(pooled)  # (n_img,1)
        rate_image = torch.nn.functional.softplus(raw_r)  # (n_img,1)

        # broadcast back:
        rate = rate_image[per_ref_idx]  # (B,1)

        # raw_r, _, _ = mean_pool_by_image(x, img_ids)
        # raw_r = self.linear_beta(raw_r)

        # mu = self._bound(raw_mu, self.log_mu_min, self.log_mu_max)  # (B,1)
        alpha = torch.nn.functional.softplus(raw_alpha) + 0.0001
        # rate = torch.nn.functional.softplus(raw_r) + 0.0001

        # log_phi_img = self.log_phi_table[img_ids[:, 2].long()]
        # phi = torch.exp(log_phi_img).unsqueeze(-1)
        # phi = torch.clamp(phi, self.fano_min, self.fano_max)

        # beta = 1.0 / (phi + self.eps)
        # alpha = mu * beta

        # dist = Gamma(concentration=alpha.flatten(), rate=beta.flatten())
        dist = Gamma(concentration=alpha.flatten(), rate=rate.flatten())
        return dist


if __name__ == "__main__":
    import torch

    from integrator.model.distributions import (
        DirichletDistribution,
        FoldedNormalDistribution,
    )
    from integrator.model.loss import LossConfig
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        load_config,
    )
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[-1]
    cfg = load_config(cfg)

    integrator = create_integrator(cfg)
    data = create_data_loader(cfg)

    losscfg = LossConfig(pprf=None, pi=None, pbg=None, shape=(1, 21, 21))

    # hyperparameters
    mc_samples = 100
    shape = (1, 21, 21)

    # distributions
    qbg_ = FoldedNormalDistribution(in_features=64)
    qi_ = FoldedNormalDistribution(in_features=64)
    qp_ = DirichletDistribution(in_features=64, out_features=(1, 21, 21))

    # load a batch
    counts, sbox, mask, meta = next(iter(data.train_dataloader()))

    gamma_dist = GammaDistribution(
        in_features=in_features, n_images=1000, estimand="intensity"
    )

    x_intensity = torch.randn(100, 64)
    img_ids = torch.randint(0, 10, (100,))

    qi = gamma_dist(x_intensity, img_ids)
