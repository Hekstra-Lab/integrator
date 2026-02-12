import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from integrator.layers import Linear

# min k value
_K_MIN = 1e-2


def _bound_k(raw_k: torch.Tensor, k_max: float | None, eps: float) -> torch.Tensor:
    """Convert raw linear output to bounded concentration parameter.

    When ``k_max`` is set, uses sigmoid to bound k in (eps, k_max).
    Otherwise uses softplus (unbounded).
    """
    if k_max is not None:
        return k_max * torch.sigmoid(raw_k) + eps
    return F.softplus(raw_k) + eps


def _init_k_bias(linear: nn.Linear, k_max: float | None, k_init: float = 1.0):
    """Initialize linear layer bias so that k starts near ``k_init``.

    For sigmoid: bias = logit(k_init / k_max).
    For softplus: bias = softplus_inv(k_init) ≈ k_init for k_init >= 1.
    """
    if linear.bias is None:
        return
    with torch.no_grad():
        if k_max is not None:
            # sigmoid(bias) = k_init / k_max  =>  bias = log(k_init / (k_max - k_init))
            ratio = max(k_init / k_max, 1e-6)
            ratio = min(ratio, 1.0 - 1e-6)
            linear.bias.fill_(math.log(ratio / (1.0 - ratio)))
        else:
            # softplus(bias) ≈ k_init  =>  bias = log(exp(k_init) - 1)
            linear.bias.fill_(math.log(math.expm1(k_init)))


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
    k_max: float | None = None,
):
    if parameterization == "a":
        assert linear_r is not None
        k = _bound_k(linear_k(x), k_max, eps)
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
        if k_max is not None:
            k = k.clamp(max=k_max)
        return k, r

    if parameterization == "d":
        assert linear_r is not None
        k = _bound_k(linear_k(x), k_max, eps)
        fano = F.softplus(linear_r(x)) + eps
        r = 1 / (fano + eps)
        return k, r

    raise ValueError(f"Unknown parameterization: {parameterization}")


def _x_to_params(x, parameterization, linear_k, linear_r=None, k_max=None):
    return _get_gamma_params(
        x=x,
        linear_k=linear_k,
        linear_r=linear_r,
        parameterization=parameterization,
        k_max=k_max,
    )


# %%
class GammaDistributionRepamA(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        in_features: int = 64,
        k_max: float | None = None,
    ):
        super().__init__()

        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)

        # buffers
        self.eps = eps
        self.k_max = k_max

        _init_k_bias(self.linear_k, k_max)

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        raw_k = self.linear_k(x)
        k = _bound_k(raw_k, self.k_max, self.eps)

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
        k_max: float | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max

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

        if self.k_max is not None:
            k = k.clamp(max=self.k_max)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamC(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
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
        if self.k_max is not None:
            k = k.clamp(max=self.k_max)
        r = 1 / (phi * mu + self.eps)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamD(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

        _init_k_bias(self.linear_k, k_max)

    def forward(
        self,
        x,
        x_,
    ):
        raw_k = self.linear_k(x)
        k = _bound_k(raw_k, self.k_max, self.eps)

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
        k_max: float | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max

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

        if self.k_max is not None:
            k = k.clamp(max=self.k_max)

        return Gamma(concentration=k.flatten(), rate=r.flatten())
