import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from .utils import get_positive_constraint


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


def _softplus_inverse_shifted(target: float, shift: float) -> float:
    """Inverse-softplus for `softplus(raw) + shift ≈ target`."""
    delta = max(target - shift, 1e-6)
    if delta > 30.0:
        return delta
    return math.log(math.expm1(delta))


# %%
class GammaDistributionRepamA(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.01,
        positive_constraint: str = "softplus",
        k_init: float = 1.0,
        r_init: float | None = None,
        zero_head_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self._constrain = get_positive_constraint(positive_constraint)
        self._constraint_name = positive_constraint

        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self._init_k_head_bias(self.linear_k, k_init)
        self._init_r_head_bias(self.linear_r, r_init)

        if zero_head_weights:
            with torch.no_grad():
                self.linear_k.weight.zero_()
                self.linear_r.weight.zero_()

    def _inverse_bias(self, target: float, floor: float) -> float:
        """Bias so that `act(bias) + floor ≈ target` at init."""
        delta = max(target - floor, 1e-12)
        if self._constraint_name == "log":
            return math.log(delta)
        if delta > 30.0:
            return float(delta)
        return math.log(math.expm1(delta))

    def _init_k_head_bias(self, linear: nn.Linear, target: float) -> None:
        if linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._inverse_bias(target, self.k_min))

    def _init_r_head_bias(
        self, linear: nn.Linear, target: float | None
    ) -> None:
        if target is None or linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._inverse_bias(target, self.eps))

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        k = self._constrain(self.linear_k(x)) + self.k_min
        r = self._constrain(self.linear_r(x_)) + self.eps
        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamB(nn.Module):
    """Gamma via (mu, fano): k = mu/fano, r = 1/fano."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        mean_init: float | None = None,
        fano_init: float = 1.0,
        mu_positive_constraint: str = "softplus",
        floor_k_min: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.floor_k_min = (
            float(floor_k_min) if floor_k_min is not None else None
        )
        self._mu_constrain = get_positive_constraint(mu_positive_constraint)
        self._mu_constraint_name = mu_positive_constraint

        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)
        self._init_mu_head(self.linear_mu, mean_init)
        # self._init_fano_head(self.linear_fano, fano_init)

        if mean_init is not None:
            with torch.no_grad():
                self.linear_mu.weight.zero_()
                self.linear_fano.weight.zero_()

    def _mu_bias(self, target: float) -> float:
        """Bias value so the mu head evaluates to `target` at init."""
        if self._mu_constraint_name == "log":
            return math.log(max(target, 1e-12))
        return _softplus_inverse_shifted(target, self.eps)

    def _init_mu_head(self, linear: nn.Linear, target: float | None) -> None:
        if target is None or linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._mu_bias(target))

    def _init_fano_head(self, linear: nn.Linear, target: float | None) -> None:
        if target is None or linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(_softplus_inverse_shifted(target, self.eps))

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        mu = self._mu_constrain(self.linear_mu(x))
        if self._mu_constraint_name == "softplus":
            mu = mu + self.eps
        fano = F.softplus(self.linear_fano(x_)) + self.eps

        r = 1.0 / fano
        k = (mu * r).clamp(min=self.k_min)

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamE(nn.Module):
    """Squared Nakagami parameterization: Gamma(m, m / Omega)."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        m_min: float = 0.1,
        positive_constraint: str = "softplus",
        mean_init: float | None = None,
        m_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.m_min = m_min
        self._constrain = get_positive_constraint(positive_constraint)

        self.linear_m = nn.Linear(in_features, 1)
        self.linear_omega = nn.Linear(in_features, 1)
        _init_k_bias(self.linear_m, k_init=m_init, k_min=m_min)
        self._init_omega_head(self.linear_omega, mean_init)

    @staticmethod
    def _omega_bias(target: float, shift: float) -> float:
        return _softplus_inverse_shifted(target, shift)

    def _init_omega_head(
        self, linear: nn.Linear, target: float | None
    ) -> None:
        if target is None or linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._omega_bias(target, self.eps))

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        m = self._constrain(self.linear_m(x)) + self.m_min
        omega = self._constrain(self.linear_omega(x_)) + self.eps
        r = m / omega
        return Gamma(concentration=m.flatten(), rate=r.flatten())
