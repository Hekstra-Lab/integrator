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


class GammaDistributionLogMean(nn.Module):
    """Gamma via (log_mean, log_fano). Derives k = mean/fano, r = 1/fano.

    Output parameterization lives in log-space so a single linear projection
    can span many orders of magnitude in the Gamma mean without softplus
    saturation or asymmetric bounds. Intended for cases where qi and qbg
    share an encoder but target very different scales (e.g. qi mean ~200K,
    qbg mean ~10).

    Pass `mean_init` (natural units) to initialize at a target scale; the
    class computes the right (log_mean_init, log_fano_init) so the initial
    k stays inside [k_min, k_max] while the initial Gamma mean matches.
    """

    def __init__(
        self,
        in_features: int = 64,
        k_min: float = 0.1,
        k_max: float = 500.0,
        log_mean_min: float = -10.0,
        log_mean_max: float = 15.0,
        log_fano_min: float = -3.0,
        log_fano_max: float = 8.0,
        mean_init: float | None = None,
        log_mean_init: float = 0.0,
        log_fano_init: float = 0.0,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.log_mean_min = log_mean_min
        self.log_mean_max = log_mean_max
        self.log_fano_min = log_fano_min
        self.log_fano_max = log_fano_max
        self.separate_inputs = separate_inputs

        if mean_init is not None:
            log_mean_init = math.log(mean_init)
            # If the target mean exceeds k_max, k would clamp unless fano
            # is initialized large enough to pull k = mean/fano below k_max.
            if mean_init > k_max:
                log_fano_init = max(log_fano_init, math.log(mean_init / k_max))

        if separate_inputs:
            self.linear_lm = nn.Linear(in_features, 1)
            self.linear_lf = nn.Linear(in_features, 1)
            with torch.no_grad():
                if self.linear_lm.bias is not None:
                    self.linear_lm.bias.fill_(log_mean_init)
                if self.linear_lf.bias is not None:
                    self.linear_lf.bias.fill_(log_fano_init)
        else:
            self.fc = nn.Linear(in_features, 2)
            with torch.no_grad():
                if self.fc.bias is not None:
                    self.fc.bias[0] = log_mean_init
                    self.fc.bias[1] = log_fano_init

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ):
        if self.separate_inputs:
            log_mean = self.linear_lm(x)
            log_fano = self.linear_lf(x_ if x_ is not None else x)
        else:
            log_mean, log_fano = self.fc(x).chunk(2, dim=-1)

        log_mean = log_mean.clamp(self.log_mean_min, self.log_mean_max)
        log_fano = log_fano.clamp(self.log_fano_min, self.log_fano_max)

        fano = torch.exp(log_fano)
        r = 1.0 / fano
        k = torch.exp(log_mean - log_fano).clamp(self.k_min, self.k_max)

        return Gamma(concentration=k.flatten(), rate=r.flatten())
