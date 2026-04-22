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
    """Gamma via (mu, fano): k = mu/fano, r = 1/fano.

    (mu, fano) is the natural statistical parameterization — mu is the
    Gamma mean, fano = var/mean is the Fano factor. Converges slower
    than RepamA's direct (k, r) because ``k = mu/fano`` couples the two
    heads through a non-diagonal Jacobian, but settles into more
    physically meaningful solutions.

    To accelerate convergence, pass ``mean_init`` (approximate expected
    mean of the data the surrogate will model) and optionally
    ``fano_init`` (default 1.0 = Poisson-like). These initialize the
    linear layer biases so ``softplus(raw_mu) ≈ mean_init`` and
    ``softplus(raw_fano) ≈ fano_init`` at step 0. Without these, the
    mu head starts at ~0.7 and has to grow many orders of magnitude
    before matching the data scale.

    ``mu_parameterization`` controls the output activation for mu only —
    fano stays in softplus space, so the (mu, fano) statistical
    decomposition is preserved in both modes:
      * ``"softplus"`` (default): ``mu = softplus(raw_mu) + eps``.
      * ``"log"``: ``mu = exp(raw_mu)``. The linear head spans log mu
        rather than mu, so weights stay O(1) even when target
        intensities span 0 to ~3e5. Recommended when mean_init is
        large (thousands+).
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        separate_inputs: bool = False,
        mean_init: float | None = None,
        fano_init: float = 1.0,
        mu_parameterization: str = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self.separate_inputs = separate_inputs
        if mu_parameterization not in ("softplus", "log"):
            raise ValueError(
                f"mu_parameterization must be 'softplus' or 'log', got {mu_parameterization!r}"
            )
        self.mu_parameterization = mu_parameterization

        if separate_inputs:
            self.linear_mu = nn.Linear(in_features, 1)
            self.linear_fano = nn.Linear(in_features, 1)
            self._init_mu_head(self.linear_mu, mean_init)
            self._init_fano_head(self.linear_fano, fano_init)
        else:
            self.fc = nn.Linear(in_features, 2)
            self._init_fc_biases(mean_init, fano_init)

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

    def _init_fc_biases(
        self,
        mean_init: float | None,
        fano_init: float | None,
    ) -> None:
        if self.fc.bias is None:
            return
        with torch.no_grad():
            if mean_init is not None:
                self.fc.bias[0] = self._mu_bias(mean_init)
            if fano_init is not None:
                self.fc.bias[1] = _softplus_inverse_shifted(
                    fano_init, self.eps
                )

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

        if self.mu_parameterization == "log":
            mu = torch.exp(raw_mu)
        else:
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
