import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma


def _bound_k(raw_k: torch.Tensor, k_min: float) -> torch.Tensor:
    """Convert raw linear output to positive concentration: softplus + k_min."""
    return F.softplus(raw_k) + k_min


def _init_k_bias(
    linear: nn.Linear,
    k_init: float = 1.0,
    k_min: float = 0.1,
):
    """Initialize linear layer bias so that k starts near `k_init`."""
    if linear.bias is None:
        return
    with torch.no_grad():
        linear.bias.fill_(math.log(math.expm1(k_init - k_min)))


# %%
class GammaDistributionRepamA(nn.Module):
    """Gamma(k, r): k via softplus+k_min, r via softplus."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_k = nn.Linear(in_features, 1)
            self.linear_r = nn.Linear(in_features, 1)
            _init_k_bias(self.linear_k, k_min=k_min)
        else:
            self.fc = nn.Linear(in_features, 2)
            # Initialize the k-bias (first output unit)
            if self.fc.bias is not None:
                with torch.no_grad():
                    self.fc.bias[0] = math.log(math.expm1(1.0 - k_min))

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        if self.separate_inputs:
            raw_k = self.linear_k(x)
            raw_r = self.linear_r(x_ if x_ is not None else x)
        else:
            raw_k, raw_r = self.fc(x).chunk(2, dim=-1)

        k = _bound_k(raw_k, self.k_min)
        r = F.softplus(raw_r) + self.eps

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamB(nn.Module):
    """Gamma via (mu, fano): k = mu/fano, r = 1/fano."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_mu = nn.Linear(in_features, 1)
            self.linear_fano = nn.Linear(in_features, 1)
        else:
            self.fc = nn.Linear(in_features, 2)

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        if self.separate_inputs:
            raw_mu = self.linear_mu(x)
            raw_fano = self.linear_fano(x_ if x_ is not None else x)
        else:
            raw_mu, raw_fano = self.fc(x).chunk(2, dim=-1)

        mu = F.softplus(raw_mu) + self.eps
        fano = F.softplus(raw_fano) + self.eps

        r = 1.0 / fano
        k = mu * r

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamC(nn.Module):
    """Gamma via (mu, phi): k = 1/phi, r = 1/(phi*mu)."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_mu = nn.Linear(in_features, 1)
            self.linear_phi = nn.Linear(in_features, 1)
        else:
            self.fc = nn.Linear(in_features, 2)

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        if self.separate_inputs:
            raw_mu = self.linear_mu(x)
            raw_phi = self.linear_phi(x_ if x_ is not None else x)
        else:
            raw_mu, raw_phi = self.fc(x).chunk(2, dim=-1)

        mu = F.softplus(raw_mu) + self.eps
        phi = F.softplus(raw_phi) + self.eps

        k = 1.0 / phi
        r = 1.0 / (phi * mu)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamD(nn.Module):
    """Gamma(k, fano): k via softplus+k_min, r = 1/fano."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_k = nn.Linear(in_features, 1)
            self.linear_fano = nn.Linear(in_features, 1)
            _init_k_bias(self.linear_k, k_min=k_min)
        else:
            self.fc = nn.Linear(in_features, 2)
            if self.fc.bias is not None:
                with torch.no_grad():
                    self.fc.bias[0] = math.log(math.expm1(1.0 - k_min))

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        if self.separate_inputs:
            raw_k = self.linear_k(x)
            raw_fano = self.linear_fano(x_ if x_ is not None else x)
        else:
            raw_k, raw_fano = self.fc(x).chunk(2, dim=-1)

        k = _bound_k(raw_k, self.k_min)
        fano = F.softplus(raw_fano) + self.eps

        r = 1.0 / fano

        return Gamma(concentration=k.flatten(), rate=r.flatten())
