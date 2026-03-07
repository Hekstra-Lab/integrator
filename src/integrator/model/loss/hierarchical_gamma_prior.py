"""
Hierarchical Gamma Prior — learns the Gamma prior parameters from data.

Implements:
    I_j ~ Gamma(alpha_I, beta_I)
    alpha_I ~ LogNormal(0, sigma)   (hyperprior)
    beta_I  ~ LogNormal(0, sigma)   (hyperprior)

The parameters alpha_I (concentration) and beta_I (rate) are stored in
log-space as learnable ``nn.Parameter`` tensors and optimized jointly with
the rest of the model.  A LogNormal hyperprior prevents them from
collapsing to degenerate values.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Gamma, LogNormal


class HierarchicalGammaPrior(nn.Module):
    """Learnable Gamma prior with LogNormal hyperprior on its parameters.

    Parameters
    ----------
    init_concentration : float
        Initial value for the Gamma concentration (alpha).
    init_rate : float
        Initial value for the Gamma rate (beta).
    hyperprior_scale : float
        Scale (sigma) of the LogNormal(0, sigma) hyperprior placed on
        both concentration and rate.  Larger values make the hyperprior
        more diffuse (less informative).
    """

    def __init__(
        self,
        init_concentration: float = 1.0,
        init_rate: float = 0.5,
        hyperprior_scale: float = 2.0,
    ):
        super().__init__()
        self.log_concentration = nn.Parameter(
            torch.tensor(math.log(init_concentration))
        )
        self.log_rate = nn.Parameter(torch.tensor(math.log(init_rate)))
        self.hyperprior_scale = hyperprior_scale

    # -- properties ----------------------------------------------------------

    @property
    def concentration(self) -> torch.Tensor:
        return self.log_concentration.exp()

    @property
    def rate(self) -> torch.Tensor:
        return self.log_rate.exp()

    # -- distribution constructors -------------------------------------------

    def prior_distribution(self) -> Gamma:
        """Return ``Gamma(alpha_I, beta_I)`` with current learned params."""
        return Gamma(self.concentration, self.rate)

    def hyperprior_log_prob(self) -> torch.Tensor:
        """Log p(alpha_I, beta_I) under the LogNormal hyperprior."""
        hp = LogNormal(0.0, self.hyperprior_scale)
        return hp.log_prob(self.concentration) + hp.log_prob(self.rate)

    # -- KL helpers ----------------------------------------------------------

    def kl_divergence(
        self,
        q: torch.distributions.Distribution,
        mc_samples: int = 100,
    ) -> torch.Tensor:
        """KL(q || Gamma(alpha_I, beta_I)), analytical or MC-estimated."""
        p = self.prior_distribution()
        try:
            return torch.distributions.kl.kl_divergence(q, p)
        except NotImplementedError:
            samples = q.rsample(torch.Size([mc_samples]))
            return (q.log_prob(samples) - p.log_prob(samples)).mean(dim=0)

    # -- repr ----------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"concentration={self.concentration.item():.4f}, "
            f"rate={self.rate.item():.4f}, "
            f"hyperprior_scale={self.hyperprior_scale}"
        )
