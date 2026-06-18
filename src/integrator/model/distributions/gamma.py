import math

import torch
import torch.nn as nn
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


def _resolve_positive_constraint(
    positive_constraint: str | tuple[str, str] | list,
) -> tuple[str, str]:
    """Resolve the `positive_constraint` arg to per-head constraint names.

    Args:
        positive_constraint: A constraint name, or a `[head1, head2]` pair of names.

    Returns:
        The `(head1, head2)` constraint names.
    """
    if isinstance(positive_constraint, str):
        return positive_constraint, positive_constraint
    seq = tuple(positive_constraint)
    if len(seq) != 2:
        raise ValueError(
            "positive_constraint must be a string or a 2-element sequence "
            f"[head1, head2], got {positive_constraint!r}"
        )
    return str(seq[0]), str(seq[1])


# %%
class GammaDistributionRepamA(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.01,
        positive_constraint: str | tuple[str, str] = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min

        k_name, r_name = _resolve_positive_constraint(positive_constraint)
        self.k_constrain = get_positive_constraint(k_name)
        self.k_constraint_name = k_name

        self.r_constrain = get_positive_constraint(r_name)
        self.r_constraint_name = r_name

        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        k = self.k_constrain(self.linear_k(x)) + self.k_min
        r = self.r_constrain(self.linear_r(x_)) + self.eps
        return Gamma(concentration=k.flatten(), rate=r.flatten())


# %%
class GammaDistributionRepamB(nn.Module):
    """Gamma via (mu, fano): k = mu/fano, r = 1/fano."""

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        positive_constraint: str | tuple[str, str] = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min

        # The two heads of this reparameterization are (mu, fano).
        mu_name, fano_name = _resolve_positive_constraint(positive_constraint)
        self._mu_constrain = get_positive_constraint(mu_name)
        self._mu_constraint_name = mu_name

        self.fano_constrain = get_positive_constraint(fano_name)
        self.fano_constraint_name = fano_name

        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        mu = self._mu_constrain(self.linear_mu(x))
        if self._mu_constraint_name == "softplus":
            mu = mu + self.eps
        fano = self.fano_constrain(self.linear_fano(x_))
        if self.fano_constraint_name == "softplus":
            fano = fano + self.eps

        r = 1.0 / fano
        k = (mu * r) + self.k_min

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

        # The two heads of this reparameterization are (m, Omega).
        m_name, omega_name = _resolve_positive_constraint(positive_constraint)
        self._m_constrain = get_positive_constraint(m_name)
        self._m_constraint_name = m_name
        self._omega_constrain = get_positive_constraint(omega_name)
        self._omega_constraint_name = omega_name

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
        m = self._m_constrain(self.linear_m(x)) + self.m_min
        omega = self._omega_constrain(self.linear_omega(x_)) + self.eps
        r = m / omega
        return Gamma(concentration=m.flatten(), rate=r.flatten())


# Reparameterization names, all selected under the single registry
# key `gamma`. The name describes what the two heads predict.
GAMMA_REPARAMETERIZATIONS = {
    "shape_rate": GammaDistributionRepamA,  # heads predict shape k and rate r
    "mean_fano": GammaDistributionRepamB,  # heads predict mean and Fano factor
    "nakagami": GammaDistributionRepamE,  # squared-Nakagami Gamma(m, m/Omega)
}

# Short aliases matching the historical RepamA/RepamB/RepamE suffixes.
_GAMMA_ALIASES = {"a": "shape_rate", "b": "mean_fano", "e": "nakagami"}


def build_gamma(reparameterization: str = "shape_rate", **kwargs) -> nn.Module:
    """Construct a Gamma intensity/background surrogate by reparameterization name.

    The YAML surrogate `name` is always `gamma`; `reparameterization` selects how the
    two heads parameterize the predicted `Gamma(concentration, rate)`:

        shape_rate (default): heads predict the shape `k` and rate `r` directly.
        mean_fano: heads predict the mean and Fano factor; `k = mu / fano`, `r = 1 / fano`.
        nakagami: squared-Nakagami form `Gamma(m, m / Omega)`.

    Aliases `a`, `b`, `e` map to `shape_rate`, `mean_fano`, `nakagami` respectively.

    Args:
        reparameterization: Canonical name or single-letter alias of the parameterization.
        **kwargs: Constructor arguments forwarded to the chosen reparameterization module.

    Returns:
        The instantiated reparameterization module.
    """
    key = str(reparameterization).lower()
    key = _GAMMA_ALIASES.get(key, key)
    impl = GAMMA_REPARAMETERIZATIONS.get(key)
    if impl is None:
        valid = ", ".join(sorted(GAMMA_REPARAMETERIZATIONS))
        raise ValueError(
            f"Unknown gamma reparameterization {reparameterization!r}. "
            f"Valid options: {valid} (aliases: a, b, e)."
        )
    _reject_unknown_gamma_args(kwargs)
    return impl(**kwargs)


def _gamma_valid_args() -> set[str]:
    """Args valid for any reparameterization (keeps swaps tolerant, catches typos)."""
    import inspect

    valid: set[str] = set()
    for cls in GAMMA_REPARAMETERIZATIONS.values():
        for name, p in inspect.signature(cls.__init__).parameters.items():
            if name != "self" and p.kind is not p.VAR_KEYWORD:
                valid.add(name)
    return valid


def _reject_unknown_gamma_args(kwargs: dict) -> None:
    valid = _gamma_valid_args()
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"Unknown gamma surrogate arg(s): {sorted(unknown)}. "
            f"Valid args: {sorted(valid)}."
        )


# the dispatcher's **kwargs hide the real args; expose them for the factory
build_gamma.arg_names = _gamma_valid_args() | {"reparameterization"}  # type: ignore[attr-defined]
