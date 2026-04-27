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


def _softplus_inverse_shifted(target: float, shift: float) -> float:
    """Inverse-softplus for `softplus(raw) + shift ≈ target`.

    Used to set a linear layer's bias so softplus(bias) + shift evaluates
    to a desired target at init. For ``delta > 30`` ``softplus(y) ≈ y`` to
    many decimals, so we short-circuit to avoid ``expm1`` overflow at
    large target magnitudes (e.g. target = 200_000 for qi intensity
    scales).
    """
    delta = max(target - shift, 1e-6)
    if delta > 30.0:
        return delta
    return math.log(math.expm1(delta))


# %%
class GammaDistributionRepamA(nn.Module):
    """Gamma(k, r) directly parameterized — both heads independent.

    Two activation modes via ``parameterization``:
      * ``"softplus"`` (default — backward-compatible):
            k = softplus(raw_k) + k_min
            r = softplus(raw_r) + eps
        For large raw_k, softplus is ≈ linear, so k stays comparable in
        magnitude to raw_k. With Kaiming-init weights × normalized encoder
        features, the head can only realistically reach k of order 10²–10³.
        Consequence on bright crystallography data: the noise-to-signal
        ratio std/mu = 1/sqrt(k) floors at ~1/√(10⁴) = 1e-2 — *worse* than
        Poisson at high intensity (CRLB says σ/μ should be 1/√I, e.g.
        3e-3 at I=10⁵). The Laplace approximation isn't satisfied.

      * ``"log"`` (recommended for high-dynamic-range Poisson data):
            k = exp(raw_k) + k_min
            r = exp(raw_r) + eps
        raw_k spans log k, so the linear head produces k spanning many
        decades while weights stay O(1). Matches gammaB's mu-side
        exponential range while keeping gammaA's structural advantage —
        k is parameterized directly with a hard floor at k_min, so the
        ``mu/fano → 0`` rsample-NaN failure path that breaks gammaB on
        bright-tail data is structurally absent.

    Bias initialization:
      * ``k_init``: target k at step 0. Bias is set so the chosen
        activation evaluates to k_init. Default 1.0 reproduces the legacy
        gammaA init.
      * ``r_init``: target r at step 0. ``None`` (default) leaves r's
        bias at PyTorch's Kaiming-uniform default (legacy behavior).

    ``zero_head_weights=True`` zeroes ``linear_k.weight`` (and
    ``linear_r.weight``, or ``fc.weight``) at construction so initial
    (k, r) is identical across reflections at step 0 — eliminates the
    seed-dependent per-reflection variance that random Kaiming weights
    × encoder features inject. Recommended on bright-tail data; default
    False for backward compat.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        paraeterization: str = "softplus",
        k_init: float = 1.0,
        r_init: float | None = None,
        zero_head_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        if parameterization not in ("softplus", "log"):
            raise ValueError(
                f"parameterization must be 'softplus' or 'log'; "
                f"got {parameterization!r}"
            )
        self.parameterization = parameterization

        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self._init_k_head_bias(self.linear_k, k_init)
        self._init_r_head_bias(self.linear_r, r_init)

        if zero_head_weights:
            with torch.no_grad():
                self.linear_k.weight.zero_()
                self.linear_r.weight.zero_()

    def _k_bias(self, target: float) -> float:
        """Bias value so the k head evaluates to `target` at init."""
        delta = max(target - self.k_min, 1e-12)
        if self.parameterization == "log":
            return math.log(delta)
        # softplus path: solve softplus(b) = target - k_min, falling back
        # to the linear approximation for large delta to avoid expm1
        # overflow.
        if delta > 30.0:
            return float(delta)
        return math.log(math.expm1(delta))

    def _r_bias(self, target: float) -> float:
        """Bias value so the r head evaluates to `target` at init."""
        delta = max(target - self.eps, 1e-12)
        if self.parameterization == "log":
            return math.log(delta)
        if delta > 30.0:
            return float(delta)
        return math.log(math.expm1(delta))

    def _init_k_head_bias(self, linear: nn.Linear, target: float) -> None:
        if linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._k_bias(target))

    def _init_r_head_bias(
        self, linear: nn.Linear, target: float | None
    ) -> None:
        if target is None or linear.bias is None:
            return
        with torch.no_grad():
            linear.bias.fill_(self._r_bias(target))

    def forward(self, x: torch.Tensor, x_: torch.Tensor | None = None):
        raw_k = self.linear_k(x)
        raw_r = self.linear_r(x_ if x_ is not None else x)

        if self.parameterization == "log":
            k = torch.exp(raw_k) + self.k_min
            r = torch.exp(raw_r) + self.eps
        else:
            k = F.softplus(raw_k) + self.k_min
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
        mean_init: float | None = None,
        fano_init: float = 1.0,
        mu_parameterization: str = "softplus",
        floor_k_min: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.floor_k_min = (
            float(floor_k_min) if floor_k_min is not None else None
        )
        if mu_parameterization not in ("softplus", "log"):
            raise ValueError(
                f"mu_parameterization must be 'softplus' or 'log', got {mu_parameterization!r}"
            )
        self.mu_parameterization = mu_parameterization

        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)
        self._init_mu_head(self.linear_mu, mean_init)
        self._init_fano_head(self.linear_fano, fano_init)

        if mean_init is not None:
            with torch.no_grad():
                self.linear_mu.weight.zero_()
                self.linear_fano.weight.zero_()

    def _mu_bias(self, target: float) -> float:
        """Bias value so the mu head evaluates to `target` at init."""
        if self.mu_parameterization == "log":
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

    def forward(self, x: torch.Tensor, x_: torch.Tensor | None = None):
        raw_mu = self.linear_mu(x)
        raw_fano = self.linear_fano(x_ if x_ is not None else x)

        if self.mu_parameterization == "log":
            mu = torch.exp(raw_mu)
        else:
            mu = F.softplus(raw_mu) + self.eps
        fano = F.softplus(raw_fano) + self.eps

        r = 1.0 / fano
        k = mu * r

        if self.floor_k_min is not None:
            k = F.softplus(k - self.floor_k_min) + self.floor_k_min
            r = k / mu

        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamE(nn.Module):
    """Squared Nakagami parameterization: Gamma(m, m / Omega)."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        m_min: float = 0.1,
        mean_init: float | None = None,
        m_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.m_min = m_min

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

    def forward(self, x: torch.Tensor, x_: torch.Tensor | None = None):
        raw_m = self.linear_m(x)
        raw_omega = self.linear_omega(x_ if x_ is not None else x)

        m = _bound_k(raw_m, self.m_min)
        omega = F.softplus(raw_omega) + self.eps

        r = m / omega

        return Gamma(concentration=m.flatten(), rate=r.flatten())
