import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma, Normal


def _softplus_inv(target: float, shift: float) -> float:
    delta = max(target - shift, 1e-6)
    if delta > 30.0:
        return delta
    return math.log(math.expm1(delta))


class HKLLookupTable(nn.Module):
    """Per-HKL Gamma variational parameters as an embedding table.

    Uses the GammaB parameterization: (mu, fano) -> Gamma(k, rate) where
    k = mu/fano + k_min, rate = 1/fano.

    mu uses exp constraint (log-space embedding) so that small additive
    updates produce multiplicative changes — necessary because F^2 spans
    orders of magnitude.  fano uses softplus since it stays O(1).

    Observations provide a precomputed ``asu_id`` that indexes into the
    table.
    """

    def __init__(
        self,
        n_hkl: int,
        init_mu: float = 1.0,
        init_fano: float = 1.0,
        eps: float = 1e-6,
        k_min: float = 0.1,
        fano_min: float = 0.0,
        mu_positive_constraint: str = "exp",
    ):
        super().__init__()
        self.n_hkl = n_hkl
        self.eps = eps
        self.k_min = k_min
        self.fano_min = fano_min
        self.mu_positive_constraint = mu_positive_constraint

        self.raw_mu = nn.Embedding(n_hkl, 1, sparse=True)
        self.raw_fano = nn.Embedding(n_hkl, 1, sparse=True)

        if mu_positive_constraint == "exp":
            nn.init.constant_(self.raw_mu.weight, math.log(max(init_mu, 1e-12)))
        else:
            nn.init.constant_(self.raw_mu.weight, _softplus_inv(init_mu, eps))
        nn.init.constant_(self.raw_fano.weight, _softplus_inv(init_fano, eps))

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Gamma, Tensor]:
        """Index into the table, build Gamma, and sample F^2.

        Returns:
            qi: Gamma distribution with batch shape (B,).
            F_sq: (S, B) structure factor squared samples.
        """
        raw = self.raw_mu(asu_ids).squeeze(-1)
        if self.mu_positive_constraint == "exp":
            mu = torch.exp(raw)
        else:
            mu = F.softplus(raw) + self.eps
        fano = F.softplus(self.raw_fano(asu_ids).squeeze(-1)) + self.eps + self.fano_min

        rate = 1.0 / fano
        k = mu * rate + self.k_min

        qi = Gamma(concentration=k, rate=rate)
        F_sq = qi.rsample([mc_samples])
        return qi, F_sq


class HKLLookupTableA(nn.Module):
    """Per-HKL Gamma variational parameters using GammaA parameterization.

    Directly learns (k, rate) instead of deriving them from (mu, fano).
    k has a floor via k_min, rate has a floor via eps. No coupling
    between concentration and rate through a shared mu.

    Gamma mean = k/rate, Gamma variance = k/rate^2.
    """

    def __init__(
        self,
        n_hkl: int,
        init_k: float = 1.0,
        init_rate: float = 1.0,
        eps: float = 1e-6,
        k_min: float = 0.1,
    ):
        super().__init__()
        self.n_hkl = n_hkl
        self.eps = eps
        self.k_min = k_min

        self.raw_k = nn.Embedding(n_hkl, 1, sparse=True)
        self.raw_rate = nn.Embedding(n_hkl, 1, sparse=True)

        nn.init.constant_(self.raw_k.weight, _softplus_inv(init_k, k_min))
        nn.init.constant_(self.raw_rate.weight, _softplus_inv(init_rate, eps))

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Gamma, Tensor]:
        k = F.softplus(self.raw_k(asu_ids).squeeze(-1)) + self.k_min
        rate = F.softplus(self.raw_rate(asu_ids).squeeze(-1)) + self.eps

        qi = Gamma(concentration=k, rate=rate)
        F_sq = qi.rsample([mc_samples])
        return qi, F_sq


class HKLAmplitudeTable(nn.Module):
    """Per-HKL amplitude variational parameters as an embedding table.

    Models the structure factor amplitude F via either:
    - ``"normal"``: X ~ N(mu, sigma), F^2 = X^2 (pathwise gradients)
    - ``"folded_normal"``: F ~ FoldedNormal(mu, sigma), F^2 = F^2
      (implicit reparameterization via rs-distributions)

    Both use the same Wilson prior in X-space: p(X) = N(0, sigma_w^2)
    which induces the Rayleigh distribution on |X| = F.  The KL is
    closed-form Normal-Normal in both cases.

    mu is stored in log-space (mu = exp(raw_mu)) so Adam produces
    multiplicative updates — matching the factory/abismal convention
    and handling the orders-of-magnitude range of F values.

    Supports Wilson-mean initialization via ``init_from_wilson`` to
    start each HKL near its resolution-appropriate expected amplitude.
    """

    def __init__(
        self,
        n_hkl: int,
        amplitude_type: str = "normal",
        init_mu: float = 1.0,
        init_sigma_frac: float = 0.05,
        eps: float = 1e-6,
        init_from_wilson: str | None = None,
    ):
        super().__init__()
        self.n_hkl = n_hkl
        self.amplitude_type = amplitude_type
        self.eps = eps

        self.raw_mu = nn.Embedding(n_hkl, 1, sparse=True)
        self.raw_sigma = nn.Embedding(n_hkl, 1, sparse=True)

        if init_from_wilson is not None:
            import torch as _torch

            wilson_data = _torch.load(
                init_from_wilson, weights_only=False, map_location="cpu"
            )
            if isinstance(wilson_data, dict):
                wilson_mu = wilson_data["wilson_F_mean"]
            else:
                wilson_mu = wilson_data
            wilson_mu = wilson_mu.float().clamp(min=1e-6)
            nn.init.constant_(self.raw_mu.weight, 0.0)
            with _torch.no_grad():
                self.raw_mu.weight[:len(wilson_mu)] = (
                    wilson_mu.log().unsqueeze(-1)
                )
            sigma_init = wilson_mu * init_sigma_frac
            nn.init.constant_(self.raw_sigma.weight, 0.0)
            with _torch.no_grad():
                self.raw_sigma.weight[:len(wilson_mu)] = (
                    _torch.tensor(
                        [_softplus_inv(float(s), eps) for s in sigma_init]
                    ).unsqueeze(-1)
                )
        else:
            nn.init.constant_(
                self.raw_mu.weight, math.log(max(init_mu, 1e-12))
            )
            sigma_val = max(init_mu, 1e-12) * init_sigma_frac
            nn.init.constant_(
                self.raw_sigma.weight, _softplus_inv(sigma_val, eps)
            )

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Index into the table, sample F^2.

        Returns:
            F_sq: (S, B) structure factor squared samples.
            mu: (B,) posterior loc (always positive via exp).
            sigma: (B,) posterior scale.
        """
        mu = torch.exp(self.raw_mu(asu_ids).squeeze(-1))
        sigma = F.softplus(self.raw_sigma(asu_ids).squeeze(-1)) + self.eps

        if self.amplitude_type == "folded_normal":
            from rs_distributions import FoldedNormal

            dist = FoldedNormal(mu, sigma)
            F_samples = dist.rsample([mc_samples])
            F_sq = F_samples.pow(2)
        else:
            X = Normal(mu, sigma).rsample([mc_samples])
            F_sq = X.pow(2)

        return F_sq, mu, sigma
