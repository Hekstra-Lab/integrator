import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _get_gamma_params(
    x: torch.Tensor,
    parameterization: str,
    linear_k: torch.nn.Linear,
    linear_r: torch.nn.Linear | None = None,
    eps: float = 1e-6,
):
    if parameterization == "a":
        assert linear_r is not None
        k = F.softplus(linear_k(x)) + eps
        r = F.softplus(linear_r(x)) + eps
        return k, r

    if parameterization == "b":
        k, r = F.softplus(linear_k(x)) + eps
        return k, r  # assuming this is intentional

    if parameterization == "c":
        assert linear_r is not None
        mu = F.softplus(linear_k(x)) + eps
        fano = F.softplus(linear_r(x)) + eps
        r = 1 / (fano + eps)
        k = r * mu
        return k, r

    if parameterization == "d":
        assert linear_r is not None
        k = F.softplus(linear_k(x)) + eps
        fano = F.softplus(linear_r(x)) + eps
        r = 1 / (fano + eps)
        return k, r

    raise ValueError(f"Unknown parameterization: {parameterization}")


def _x_to_params(x, parameterization, linear_k, linear_r=None):
    return _get_gamma_params(
        x=x,
        linear_k=linear_k,
        linear_r=linear_r,
        parameterization=parameterization,
    )


# %%
class GammaDistributionRepamA(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        in_features: int = 64,
    ):
        super().__init__()

        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)

        # buffers
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        raw_k = self.linear_k(x)
        k = F.softplus(raw_k) + self.eps

        raw_r = self.linear_r(x_)
        r = F.softplus(raw_r) + self.eps

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamB(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

        # Linear layers
        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
    ):
        raw_mu = self.linear_mu(x)
        mu = F.softplus(raw_mu) + self.eps

        raw_fano = self.linear_fano(x_)
        fano = F.softplus(raw_fano) + self.eps

        r = 1 / (fano + self.eps)
        k = mu * r

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamC(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        # Linear layers
        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_phi = nn.Linear(in_features, 1)

    def forward(
        self,
        x,
        x_,
    ):
        raw_mu = self.linear_mu(x)
        mu = F.softplus(raw_mu) + self.eps

        raw_phi = self.linear_phi(x_)
        phi = F.softplus(raw_phi) + self.eps

        k = 1 / (phi + self.eps)
        r = 1 / (phi * mu + self.eps)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamD(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

    def forward(
        self,
        x,
        x_,
    ):
        raw_k = self.linear_k(x)
        k = F.softplus(raw_k) + self.eps

        raw_fano = self.linear_fano(x_)
        fano = F.softplus(raw_fano) + self.eps

        r = 1 / (fano + self.eps)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistribution(nn.Module):
    def __init__(
        self,
        in_features: int,
        estimand: Literal["background", "intensity"],
        parameterization: str = "a",
        eps: float = 1e-6,
    ):
        super().__init__()

        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 1e6
            self.fano_min, self.fano_max = 0.2, 2.0
        else:
            self.mu_min, self.mu_max = 1e-3, 100.0
            self.fano_min, self.fano_max = 0.2, 5.0

        self.name = "Gamma"
        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_fano_min = math.log(self.fano_min)
        self.log_fano_max = math.log(self.fano_max)

        if parameterization == "b":
            self.linear_k = torch.nn.Linear(in_features, 2)
            pass

        else:
            self.linear_k = torch.nn.Linear(in_features, 1)
            self.linear_r = torch.nn.Linear(in_features, 1)
            pass

        self.linear_k = torch.nn.Linear(in_features, 1)
        self.ln_beta = nn.LayerNorm(in_features)
        self.linear_r = torch.nn.Linear(in_features, 1)

        self.rmin = 0.01
        self.rmax = 5.0
        self.parameterization = parameterization

    def _bound(self, raw, log_min, log_max):
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))

    # def forward(self, x, xim, im_idx):
    def forward(self, x, x_=None):
        """
        x: (batch, features)
        img_ids:(batch,) integer indices 0...n_images-1
        """

        raw_mu = self.linear_k(x)
        mu = torch.nn.functional.softplus(raw_mu) + 1e-6

        raw_fano = self.linear_r(x)
        fano = torch.nn.functional.softplus(raw_fano) + 1e-6
        rate = 1 / (fano + 1e-6)
        alpha = mu * rate

        q = Gamma(concentration=alpha.flatten(), rate=rate.flatten())

        return q


# %%
class GammaDistribution(nn.Module):
    def __init__(
        self,
        in_features: int,
        estimand: Literal["background", "intensity"],
        parameterization: str = "a",
        eps: float = 1e-6,
    ):
        super().__init__()

        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 1e6
            self.fano_min, self.fano_max = 0.2, 2.0
        else:
            self.mu_min, self.mu_max = 1e-3, 100.0
            self.fano_min, self.fano_max = 0.2, 5.0

        self.name = "Gamma"
        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_fano_min = math.log(self.fano_min)
        self.log_fano_max = math.log(self.fano_max)

        if parameterization == "b":
            self.linear_k = torch.nn.Linear(in_features, 2)
            pass

        else:
            self.linear_k = torch.nn.Linear(in_features, 1)
            self.linear_r = torch.nn.Linear(in_features, 1)
            pass

        # managed to amortize it
        # self.log_phi_table = nn.Parameter(torch.zeros(n_images))  # (n_images,)

        self.linear_k = torch.nn.Linear(in_features, 1)
        self.ln_beta = nn.LayerNorm(in_features)
        self.linear_r = torch.nn.Linear(in_features, 1)

        self.rmin = 0.01
        self.rmax = 5.0
        self.parameterization = parameterization

    def _bound(self, raw, log_min, log_max):
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))

    # def forward(self, x, xim, im_idx):
    def forward(self, x, x_=None):
        """
        x: (batch, features)
        img_ids:(batch,) integer indices 0...n_images-1
        """

        raw_mu = self.linear_k(x)
        mu = torch.nn.functional.softplus(raw_mu) + 1e-6

        raw_fano = self.linear_r(x)
        fano = torch.nn.functional.softplus(raw_fano) + 1e-6
        rate = 1 / (fano + 1e-6)
        alpha = mu * rate

        q = Gamma(concentration=alpha.flatten(), rate=rate.flatten())

        return q


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
