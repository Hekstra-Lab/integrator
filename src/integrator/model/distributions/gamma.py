import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma
from torch.types import _size

from integrator.layers import Linear


class FanoGamma(Gamma):
    def __init__(
        self,
        concentration: torch.Tensor | float,
        fano: torch.Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.fano, _ = torch.broadcast_tensors(
            torch.as_tensor(fano), torch.as_tensor(concentration)
        )
        super().__init__(
            concentration, 1.0 / self.fano, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FanoGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.fano = self.fano.expand(batch_shape)
        super(FanoGamma, new).__init__(
            self.concentration.expand(batch_shape),
            self.rate.expand(batch_shape),
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        value = torch._standard_gamma(
            self.concentration.expand(shape)
        ) * self.fano.expand(shape)
        value.detach().clamp_(min=torch.finfo(value.dtype).tiny)
        return value


def _bound_k(
    raw_k: torch.Tensor, k_max: float | None, k_min: float
) -> torch.Tensor:
    """Convert raw linear output to bounded concentration parameter.

    When ``k_max`` is set, uses sigmoid to bound k in [k_min, k_max).
    Otherwise uses softplus + k_min (unbounded above).
    """
    if k_max is not None:
        return k_min + (k_max - k_min) * torch.sigmoid(raw_k)
    return F.softplus(raw_k) + k_min


def _init_k_bias(
    linear: nn.Linear,
    k_max: float | None,
    k_init: float = 1.0,
    k_min: float = 0.1,
):
    """Initialize linear layer bias so that k starts near ``k_init``.

    For sigmoid: sigmoid(bias) = (k_init - k_min) / (k_max - k_min).
    For softplus: bias = softplus_inv(k_init - k_min).
    """
    if linear.bias is None:
        return
    with torch.no_grad():
        if k_max is not None:
            # sigmoid(bias) = (k_init - k_min) / (k_max - k_min)
            ratio = (k_init - k_min) / (k_max - k_min)
            ratio = max(ratio, 1e-6)
            ratio = min(ratio, 1.0 - 1e-6)
            linear.bias.fill_(math.log(ratio / (1.0 - ratio)))
        else:
            # softplus(bias) = k_init - k_min  =>  bias = log(exp(k_init - k_min) - 1)
            linear.bias.fill_(math.log(math.expm1(k_init - k_min)))


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
    k_min: float = 0.1,
):
    if parameterization == "a":
        assert linear_r is not None
        k = _bound_k(linear_k(x), k_max, k_min)
        r = F.softplus(linear_r(x)) + eps
        return k, r

    if parameterization == "b":
        k, r = F.softplus(linear_k(x)) + eps
        return k, r  # assuming this is intentional

    if parameterization == "c":
        assert linear_r is not None
        mu = F.softplus(linear_k(x)) + eps
        fano = F.softplus(linear_r(x)) + eps
        r = 1.0 / fano
        k = (r * mu).clamp(min=k_min)
        if k_max is not None:
            k = k.clamp(max=k_max)
        return k, r

    if parameterization == "d":
        assert linear_r is not None
        k = _bound_k(linear_k(x), k_max, k_min)
        fano = F.softplus(linear_r(x)) + eps
        r = 1.0 / fano
        return k, r

    raise ValueError(f"Unknown parameterization: {parameterization}")


def _x_to_params(
    x, parameterization, linear_k, linear_r=None, k_max=None, k_min=0.1
):
    return _get_gamma_params(
        x=x,
        linear_k=linear_k,
        linear_r=linear_r,
        parameterization=parameterization,
        k_max=k_max,
        k_min=k_min,
    )


# %%
class GammaDistributionRepamA(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        in_features: int = 64,
        k_max: float | None = None,
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)

        # buffers
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min

        _init_k_bias(self.linear_k, k_max, k_min=k_min)

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        raw_k = self.linear_k(x)
        k = _bound_k(raw_k, self.k_max, self.k_min)

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
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min

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

        r = 1.0 / fano
        k = (mu * r).clamp(min=self.k_min)

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
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min
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

        k = (1.0 / phi).clamp(min=self.k_min)
        if self.k_max is not None:
            k = k.clamp(max=self.k_max)
        r = 1.0 / (phi * mu)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamD(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min
        # Linear layers
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

        _init_k_bias(self.linear_k, k_max, k_min=k_min)

    def forward(
        self,
        x,
        x_,
    ):
        raw_k = self.linear_k(x)
        k = _bound_k(raw_k, self.k_max, self.k_min)

        raw_fano = self.linear_fano(x_)
        fano = F.softplus(raw_fano) + self.eps

        r = 1.0 / fano

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistribution(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min

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

        r = 1.0 / fano
        k = (mu * r).clamp(min=self.k_min)

        if self.k_max is not None:
            k = k.clamp(max=self.k_max)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
# ─── FanoGamma variants ──────────────────────────────────────────────────────
# These use FanoGamma (multiply by fano in rsample) instead of
# Gamma (divide by rate=1/fano), giving a shorter autograd chain.
# Functionally identical to the originals — same samples, same gradients.


class FanoGammaRepamB(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min
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

        k = (mu / fano).clamp(min=self.k_min)

        if self.k_max is not None:
            k = k.clamp(max=self.k_max)

        return FanoGamma(concentration=k.flatten(), fano=fano.flatten())


class FanoGammaRepamD(nn.Module):
    def __init__(
        self,
        in_features: int,
        eps: float = 1e-6,
        k_max: float | None = None,
        k_min: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_max = k_max
        self.k_min = k_min
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

        _init_k_bias(self.linear_k, k_max, k_min=k_min)

    def forward(
        self,
        x,
        x_,
    ):
        raw_k = self.linear_k(x)
        k = _bound_k(raw_k, self.k_max, self.k_min)

        raw_fano = self.linear_fano(x_)
        fano = F.softplus(raw_fano) + self.eps

        return FanoGamma(concentration=k.flatten(), fano=fano.flatten())
