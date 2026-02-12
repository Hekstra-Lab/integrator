import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from integrator.layers import Linear

# min k value
_K_MIN = 1e-2


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
        k = F.softplus(linear_k(x)).clamp(min=_K_MIN)
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
        k = (r * mu).clamp(min=_K_MIN)
        return k, r

    if parameterization == "d":
        assert linear_r is not None
        k = F.softplus(linear_k(x)).clamp(min=_K_MIN)
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

        if x_ is not None:
            raw_r = self.linear_r(x_)
        else:
            raw_r = self.linear_r(x)

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
        x_: torch.Tensor | None = None,
    ):
        raw_mu = self.linear_mu(x)
        mu = F.softplus(raw_mu) + self.eps

        if x_ is not None:
            raw_fano = self.linear_fano(x_)
        else:
            raw_fano = self.linear_fano(x)

        fano = F.softplus(raw_fano) + self.eps

        r = 1 / (fano + self.eps)
        k = (mu * r) + self.eps

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

        k = (1 / (phi + self.eps)).clamp(min=_K_MIN)
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
        k = F.softplus(raw_k).clamp(min=_K_MIN)

        raw_fano = self.linear_fano(x_)
        fano = F.softplus(raw_fano) + self.eps

        r = 1 / (fano + self.eps)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistribution(nn.Module):
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
        k = (mu * r).clamp(min=_K_MIN)

        return Gamma(concentration=k.flatten(), rate=r.flatten())
