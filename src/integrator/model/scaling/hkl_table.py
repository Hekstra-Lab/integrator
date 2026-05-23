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
    ):
        super().__init__()
        self.n_hkl = n_hkl
        self.eps = eps
        self.k_min = k_min

        self.raw_mu = nn.Embedding(n_hkl, 1, sparse=True)
        self.raw_fano = nn.Embedding(n_hkl, 1, sparse=True)

        nn.init.constant_(self.raw_mu.weight, math.log(max(init_mu, 1e-12)))
        nn.init.constant_(self.raw_fano.weight, _softplus_inv(init_fano, eps))

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Gamma, Tensor]:
        """Index into the table, build Gamma, and sample F^2.

        Returns:
            qi: Gamma distribution with batch shape (B,).
            F_sq: (S, B) structure factor squared samples.
        """
        mu = torch.exp(self.raw_mu(asu_ids).squeeze(-1))
        fano = F.softplus(self.raw_fano(asu_ids).squeeze(-1)) + self.eps

        rate = 1.0 / fano
        k = mu * rate + self.k_min

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

    The ``"normal"`` variant is cheaper (pathwise gradients) but samples
    can be negative (only X^2 enters the rate, so this is harmless).
    The ``"folded_normal"`` variant samples are always non-negative but
    uses implicit reparameterization (more expensive).
    """

    def __init__(
        self,
        n_hkl: int,
        amplitude_type: str = "normal",
        init_mu: float = 1.0,
        init_sigma: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_hkl = n_hkl
        self.amplitude_type = amplitude_type
        self.eps = eps

        self.raw_mu = nn.Embedding(n_hkl, 1, sparse=True)
        self.raw_sigma = nn.Embedding(n_hkl, 1, sparse=True)

        nn.init.constant_(self.raw_mu.weight, init_mu)
        nn.init.constant_(
            self.raw_sigma.weight, _softplus_inv(init_sigma, eps)
        )

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Index into the table, sample F^2.

        Returns:
            F_sq: (S, B) structure factor squared samples.
            mu: (B,) posterior mean of signed amplitude.
            sigma: (B,) posterior std of signed amplitude.
        """
        mu = self.raw_mu(asu_ids).squeeze(-1)
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
