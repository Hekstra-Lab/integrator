import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma


def _softplus_inv(target: float, shift: float) -> float:
    delta = max(target - shift, 1e-6)
    if delta > 30.0:
        return delta
    return math.log(math.expm1(delta))


class HKLLookupTable(nn.Module):
    """Per-HKL Gamma variational parameters as an embedding table.

    Uses the GammaB parameterization: (mu, fano) -> Gamma(k, rate) where
    k = mu/fano + k_min, rate = 1/fano. Each unique reflection in the
    asymmetric unit gets its own (mu, fano) pair, stored as unconstrained
    embeddings with softplus activation.

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

        raw_mu_init = _softplus_inv(init_mu, eps)
        raw_fano_init = _softplus_inv(init_fano, eps)
        nn.init.constant_(self.raw_mu.weight, raw_mu_init)
        nn.init.constant_(self.raw_fano.weight, raw_fano_init)

    def forward(
        self, asu_ids: Tensor, mc_samples: int = 1
    ) -> tuple[Gamma, Tensor]:
        """Index into the table, build Gamma, and sample F^2.

        Args:
            asu_ids: (B,) integer ASU reflection IDs.
            mc_samples: number of Monte-Carlo samples.

        Returns:
            qi: Gamma distribution with batch shape (B,).
            F_sq: (S, B) structure factor squared samples.
        """
        mu = F.softplus(self.raw_mu(asu_ids).squeeze(-1)) + self.eps
        fano = F.softplus(self.raw_fano(asu_ids).squeeze(-1)) + self.eps

        rate = 1.0 / fano
        k = mu * rate + self.k_min

        qi = Gamma(concentration=k, rate=rate)
        F_sq = qi.rsample([mc_samples])
        return qi, F_sq
